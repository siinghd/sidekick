"""Configuration commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax

from sidekick.core.config import get_settings
from sidekick.core.paths import get_paths

console = Console()


def config_cmd(
    key: Optional[str] = typer.Argument(
        None,
        help="Config key to get (e.g., model.base)",
    ),
    value: Optional[str] = typer.Argument(
        None,
        help="Value to set",
    ),
) -> None:
    """View or modify Sidekick configuration.

    Examples:
        sidekick config                    # View all config
        sidekick config model.base         # View specific key
        sidekick config model.base phi-3   # Set value
    """
    paths = get_paths()

    if not paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    settings = get_settings()

    # No args - show full config
    if key is None:
        if paths.config_file.exists():
            content = paths.config_file.read_text()
            syntax = Syntax(content, "toml", theme="monokai", line_numbers=True)
            console.print(syntax)
        else:
            console.print("[dim]No configuration file found.[/]")
        return

    # Key only - show value
    if value is None:
        try:
            val = settings.get(key)
            console.print(f"{key} = {val}")
        except (KeyError, AttributeError) as e:
            console.print(f"[red]Unknown key:[/] {key}")
            raise typer.Exit(1)
        return

    # Key and value - set it
    try:
        # Attempt type conversion
        old_val = settings.get(key)
        if isinstance(old_val, bool):
            new_val = value.lower() in ("true", "1", "yes")
        elif isinstance(old_val, int):
            new_val = int(value)
        elif isinstance(old_val, float):
            new_val = float(value)
        else:
            new_val = value

        settings.set(key, new_val)
        settings.save()
        console.print(f"[green]Set[/] {key} = {new_val}")
    except (KeyError, AttributeError):
        console.print(f"[red]Unknown key:[/] {key}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Invalid value:[/] {e}")
        raise typer.Exit(1)
