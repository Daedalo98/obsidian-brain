from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Safely import BM25 (Community)
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.retrievers import BM25Retriever

from langchain_community.tools import DuckDuckGoSearchRun
from core.config_loader import ConfigLoader
from core.database import DatabaseManager
from core.ollama_client import OllamaClient
from utils.logger import logger, log_step, log_brain

# --- CUSTOM ENSEMBLE RETRIEVER ---
# This ensures it works regardless of LangChain version
class SimpleEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        all_docs = []
        for i, retriever in enumerate(self.retrievers):
            try:
                docs = retriever.invoke(query)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Retriever {i} failed: {e}")
        
        # Deduplicate
        unique_docs = []
        seen_hashes = set()
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)
        return unique_docs
# ---------------------------------

class RetrievalEngine:
    def __init__(self, config: ConfigLoader, db: DatabaseManager, ollama: OllamaClient):
        self.config = config
        self.llm = ollama.get_llm()
        self.web_search = DuckDuckGoSearchRun()
        self.vector_retriever = db.get_retriever(k=config.get('retrieval', 'top_k'))
        self.db = db

    def hybrid_search(self, query: str) -> List[Document]:
        log_step("Performing Hybrid Search (BM25 + Vector)...")
        docs = self.db.get_all_documents()
        if not docs: 
            return []
        
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.config.get('retrieval', 'top_k')
        
        ensemble = SimpleEnsembleRetriever(
            retrievers=[bm25, self.vector_retriever], 
            weights=[0.5, 0.5]
        )
        return ensemble.invoke(query)

    def hyde_search(self, query: str) -> List[Document]:
        log_step("Executing HyDE (Hypothetical Document Embeddings)...")
        template = "Write a passage that answers: {question}"
        hypothetical = (PromptTemplate.from_template(template) | self.llm | StrOutputParser()).invoke({"question": query})
        log_brain("Generated hypothetical answer for embedding alignment.")
        return self.vector_retriever.invoke(hypothetical)

    def execute_retrieval(self, query: str) -> List[Document]:
        strategy = self.config.get('retrieval', 'strategy')
        
        try:
            if strategy == "hybrid": docs = self.hybrid_search(query)
            elif strategy == "hyde": docs = self.hyde_search(query)
            else: docs = self.vector_retriever.invoke(query)
        except Exception as e:
            logger.error(f"Retrieval strategy '{strategy}' failed: {e}")
            docs = self.vector_retriever.invoke(query)
        
        if not docs and self.config.get('retrieval', 'web_fallback'):
            log_step("⚠️ Local context missing. Triggering Web Search...")
            try:
                web_res = self.web_search.invoke(query)
                return [Document(page_content=web_res, metadata={"source": "DuckDuckGo", "filename": "Web"})]
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                return []
            
        return docs