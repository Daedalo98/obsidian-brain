# 🧠 Obsidian-Brain

> **Your Local Second Brain.**
> A production-ready, local-first RAG (Retrieval-Augmented Generation) system specifically designed for [Obsidian.md](https://obsidian.md/) users.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/AI-Ollama-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Obsidian-Brain** allows you to chat with your markdown notes using local LLMs. It runs entirely on your machine (Ubuntu/Linux/Mac), ensuring 100% privacy. It uses **"Observed Retrieval"** to show you exactly how it thinks, retrieves, and answers.

---

## 🏗️ Architecture

The system uses a modular **Strategy Pattern** for retrieval:

```mermaid
graph LR
    User[User Query] --> Brain[Retrieval Engine]
    Brain -->|Strategy A| Hybrid[Hybrid Search\n(BM25 + Vector)]
    Brain -->|Strategy B| HyDE[HyDE\n(Hypothetical Answer)]
    Hybrid --> Chroma[(ChromaDB\nVector Store)]
    HyDE --> Chroma
    Chroma --> Context[Context Window]
    Context --> LLM[Ollama (Llama3)]
    LLM --> Answer
```
✨ Key Features

    🔒 100% Local & Private: Powered by Ollama. Your data never leaves your SSD.

    ⚡ Incremental Indexing: Uses MD5 hashing to only re-index files that have changed.

    🔎 Hybrid Search: Combines keyword matching (BM25) with semantic search (Embeddings) for high precision.

    🧠 HyDE (Hypothetical Document Embeddings): Generates a fake answer first to find semantically similar notes (great for abstract questions).

    🌐 Web Fallback: Automatically searches DuckDuckGo if your notes don't contain the answer.

    🔗 Link Aware: (In Progress) Parses [[Wikilinks]] to understand note connections.

🛠️ Prerequisites

    Python 3.10+

    Ollama running locally.

Pull the required models:
```Bash
ollama pull llama3
ollama pull nomic-embed-text
```

🚀 Installation

Clone the repo
```Bash
git clone [https://github.com/yourusername/obsidian-brain.git](https://github.com/yourusername/obsidian-brain.git)
cd obsidian-brain
```

Set up Virtual Environment
```Bash
python3 -m venv venv
source venv/bin/activate
```
Install Dependencies
```Bash
pip install -r requirements.txt
```
Configure Environment Create a .env file:
```Bash
cp .env.example .env
```
Edit .env to point to your Vault:
```
OBSIDIAN_VAULT_PATH="/home/user/Documents/ObsidianVault"
CHROMA_DB_PATH="./chroma_db"
OLLAMA_BASE_URL="http://localhost:11434"
```

🖥️ Usage
1. Build the Brain (Indexing)

Run this whenever you add new notes. It's fast because it skips unchanged files.
Bash

python main.py index

2. Chat with your Notes

The default mode uses Hybrid Search.
Bash

python main.py ask "What are my notes on Docker?"

3. Advanced Strategies

HyDE Strategy: Best for complex concepts or "how-to" questions.
Bash

python main.py ask "How do I optimize a database?" --strategy hyde

⚙️ Configuration (config.yaml)

You can tune the RAG parameters without changing code:
```YAML

system:
  chunk_size: 1000       # Size of text chunks
  chunk_overlap: 200     # Context overlap

retrieval:
  top_k: 5               # Number of notes to retrieve
  web_fallback: true     # Enable DuckDuckGo search
  strategy: "hybrid"     # Default strategy
```