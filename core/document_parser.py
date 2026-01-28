import re
import hashlib
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from utils.logger import logger

class DocumentParser:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.wikilink_pattern = re.compile(r'\[\[(.*?)\]\]')

    def calculate_file_hash(self, file_path: Path) -> str:
        """MD5 hash for change detection."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def parse_file(self, file_path: Path) -> List[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple Metadata & Link Extraction
            links = [l.split('|')[0] for l in self.wikilink_pattern.findall(content)]
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "hash": self.calculate_file_hash(file_path),
                "links": ",".join(links) # Store as string for Chroma compatibility
            }
            return [Document(page_content=content, metadata=metadata)]
        except Exception as e:
            logger.warning(f"Failed to parse {file_path.name}: {e}")
            return []

    def load_vault(self) -> List[Document]:
        docs = []
        for file_path in self.vault_path.rglob("*.md"):
            docs.extend(self.parse_file(file_path))
        return docs