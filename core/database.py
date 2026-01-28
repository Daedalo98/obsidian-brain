from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # <--- THIS WAS MISSING
from core.config_loader import ConfigLoader
from core.ollama_client import OllamaClient
from core.document_parser import DocumentParser
from utils.logger import logger, log_step, log_brain

class DatabaseManager:
    def __init__(self, config: ConfigLoader, ollama: OllamaClient):
        self.persist_directory = config.get('system', 'chroma_path')
        self.embedding_function = ollama.get_embeddings()
        self.parser = DocumentParser(config.get('system', 'vault_path'))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('system', 'chunk_size'),
            chunk_overlap=config.get('system', 'chunk_overlap')
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            collection_name="obsidian_vault"
        )

    def index_vault(self):
        log_step("Starting incremental indexing...")
        disk_docs = self.parser.load_vault()
        
        # Fetch existing hashes to avoid re-embedding
        existing_data = self.vector_store.get()
        existing_hashes = set()
        if existing_data and 'metadatas' in existing_data:
            for meta in existing_data['metadatas']:
                if meta and 'hash' in meta: existing_hashes.add(meta['hash'])
        
        new_docs = [d for d in disk_docs if d.metadata['hash'] not in existing_hashes]
        
        if not new_docs:
            log_step("✨ No changes detected. Database up to date.")
            return

        log_step(f"⚡ Processing {len(new_docs)} new/modified documents...")
        splits = self.splitter.split_documents(new_docs)
        
        if splits:
            batch_size = 100
            for i in range(0, len(splits), batch_size):
                self.vector_store.add_documents(splits[i:i+batch_size])
                log_brain(f"Indexed batch {i//batch_size + 1}")
            
        log_step("✅ Indexing complete.")

    def get_retriever(self, k=5):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
        
    def get_all_documents(self):
        """Helper for BM25 (fetch all docs text)."""
        # This function caused the error because Document wasn't imported
        data = self.vector_store.get() 
        if not data['documents']:
            return []
        return [Document(page_content=t, metadata=m) for t, m in zip(data['documents'], data['metadatas'])]