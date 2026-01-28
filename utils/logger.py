import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "step": "bold green",
    "brain": "magenta"
})

console = Console(theme=custom_theme)

def setup_logger():
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )
    return logging.getLogger("obsidian_brain")

logger = setup_logger()

def log_step(message: str):
    """Log a major processing step."""
    console.print(f"[step]➜[/step] {message}")

def log_brain(message: str):
    """Log internal reasoning/thought process."""
    console.print(f"[brain]🧠 {message}[/brain]")