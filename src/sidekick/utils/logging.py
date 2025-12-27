"""Structured logging for Sidekick."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from sidekick.core.paths import get_paths

# Module-level console
console = Console(stderr=True)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file to log to
        verbose: If True, use DEBUG level
    """
    if verbose:
        level = logging.DEBUG

    # Create root logger
    root_logger = logging.getLogger("sidekick")
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    if not name.startswith("sidekick"):
        name = f"sidekick.{name}"
    return logging.getLogger(name)


# Set up default logging on import
setup_logging()
