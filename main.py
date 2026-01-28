import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from core.config_loader import ConfigLoader
from core.ollama_client import OllamaClient
from core.database import DatabaseManager
from core.retrieval_engine import RetrievalEngine
from utils.logger import log_step

console = Console()

@click.group()
def cli():
    """Obsidian-Brain: Local RAG System"""
    pass

@cli.command()
def index():
    """Index the Obsidian Vault."""
    cfg = ConfigLoader()
    db = DatabaseManager(cfg, OllamaClient(cfg))
    db.index_vault()

@cli.command()
@click.argument('query')
@click.option('--strategy', default=None, help='Override retrieval strategy')
def ask(query, strategy):
    """Ask a question to your brain."""
    cfg = ConfigLoader()
    if strategy: cfg.config['retrieval']['strategy'] = strategy

    ollama = OllamaClient(cfg)
    engine = RetrievalEngine(cfg, DatabaseManager(cfg, ollama), ollama)

    log_step(f"Query: [bold cyan]{query}[/bold cyan]")
    docs = engine.execute_retrieval(query)
    
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get('filename', 'Unknown') for d in docs]))
    log_step(f"Retrieved {len(docs)} docs from: {sources}")

    log_step("Generating answer...")
    prompt = f"Answer strictly using this Context:\n{context}\n\nQuestion: {query}"
    response = ollama.get_llm().invoke(prompt)
    
    console.print(Panel(Markdown(response.content), title="Obsidian-Brain Answer", border_style="green"))

if __name__ == "__main__":
    cli()