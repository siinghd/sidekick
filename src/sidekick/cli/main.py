"""Main CLI entry point for Sidekick."""

import typer
from rich.console import Console

from sidekick import __version__

# Create the main app
app = typer.Typer(
    name="sidekick",
    help="Your AI's AI - A personal memory model for LLMs",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]Sidekick[/] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Sidekick - Your AI's AI.

    A personal memory model that knows YOU. Train it on your notes, chats,
    and documents. Query it from any LLM.
    """
    pass


# Import and register subcommands
from sidekick.cli.init import init_cmd
from sidekick.cli.add import add_cmd
from sidekick.cli.list import list_cmd
from sidekick.cli.train import train_cmd
from sidekick.cli.ask import ask_cmd
from sidekick.cli.serve import serve_cmd
from sidekick.cli.config_cmd import config_cmd
from sidekick.cli.export import export_cmd
from sidekick.cli.watch import watch_cmd
from sidekick.cli.setup import setup_cmd

app.command(name="init")(init_cmd)
app.command(name="add")(add_cmd)
app.command(name="list")(list_cmd)
app.command(name="train")(train_cmd)
app.command(name="ask")(ask_cmd)
app.command(name="serve")(serve_cmd)
app.command(name="config")(config_cmd)
app.command(name="export")(export_cmd)
app.command(name="watch")(watch_cmd)
app.command(name="setup")(setup_cmd)


if __name__ == "__main__":
    app()
