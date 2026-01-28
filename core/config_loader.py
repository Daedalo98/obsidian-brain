import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()

class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file missing at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Env var overrides
        cfg['system']['vault_path'] = os.getenv("OBSIDIAN_VAULT_PATH", "./vault")
        cfg['system']['ollama_url'] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cfg['system']['chroma_path'] = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        return cfg

    def get(self, section, key=None):
        if key: return self.config.get(section, {}).get(key)
        return self.config.get(section, {})