"""Initialize Sidekick."""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from sidekick.core.config import Settings
from sidekick.core.paths import get_paths

console = Console()


def init_cmd(
    base_model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Base model to use (e.g., qwen2.5-1.5b, phi-3-mini)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Reinitialize even if already initialized",
    ),
) -> None:
    """Initialize Sidekick.

    Creates the ~/.sidekick/ directory structure and configuration.
    """
    paths = get_paths()

    if paths.is_initialized() and not force:
        console.print(
            "[yellow]Sidekick is already initialized.[/]\n"
            "Use [bold]--force[/] to reinitialize."
        )
        raise typer.Exit(1)

    # Create directory structure
    console.print("Creating directory structure...")
    paths.ensure_all()

    # Create default config
    settings = Settings()
    if base_model:
        settings.model.base = base_model

    settings.save()

    console.print(
        Panel.fit(
            f"[green]Sidekick initialized successfully![/]\n\n"
            f"[dim]Config:[/] {paths.config_file}\n"
            f"[dim]Data:[/]   {paths.data_dir}\n"
            f"[dim]Models:[/] {paths.models_dir}\n\n"
            f"[bold]Next steps:[/]\n"
            f"  1. Add your data:  [cyan]sidekick add ./notes/[/]\n"
            f"  2. Train model:    [cyan]sidekick train[/]\n"
            f"  3. Ask questions:  [cyan]sidekick ask \"What do I work on?\"[/]",
            title="[bold blue]Sidekick[/]",
            border_style="blue",
        )
    )
