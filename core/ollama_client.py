from langchain_ollama import ChatOllama, OllamaEmbeddings
from core.config_loader import ConfigLoader

class OllamaClient:
    def __init__(self, config: ConfigLoader):
        self.base_url = config.get('system', 'ollama_url')
        self.llm_model = config.get('system', 'llm_model')
        self.embed_model_name = config.get('system', 'embed_model')

    def get_llm(self, temperature=0):
        return ChatOllama(base_url=self.base_url, model=self.llm_model, temperature=temperature)

    def get_embeddings(self):
        return OllamaEmbeddings(base_url=self.base_url, model=self.embed_model_name)