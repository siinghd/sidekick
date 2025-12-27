"""Export model or data."""

from typing import Optional

import typer
from rich.console import Console

from sidekick.core.paths import get_paths

console = Console()


def export_cmd(
    format: str = typer.Option(
        "gguf",
        "--format",
        "-f",
        help="Export format: gguf",
    ),
    quantize: str = typer.Option(
        "q4_k_m",
        "--quantize",
        "-q",
        help="Quantization: q4_k_m, q8_0, f16",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path",
    ),
    data: Optional[str] = typer.Option(
        None,
        "--data",
        "-d",
        help="Export data instead: qa, raw",
    ),
) -> None:
    """Export the trained model or data.

    Examples:
        sidekick export
        sidekick export --format gguf --quantize q8_0
        sidekick export --data qa -o ./my_qa.jsonl
    """
    paths = get_paths()

    if not paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    if data:
        # Export data
        if data == "qa":
            if paths.qa_dataset_file.exists():
                console.print(f"[green]QA dataset:[/] {paths.qa_dataset_file}")
            else:
                console.print("[yellow]No QA dataset found.[/]")
        elif data == "raw":
            console.print(f"[green]Raw data:[/] {paths.raw_dir}")
        else:
            console.print(f"[red]Unknown data type:[/] {data}")
            console.print("Valid options: qa, raw")
            raise typer.Exit(1)
    else:
        # Export model
        console.print(f"[bold]Export Configuration[/]")
        console.print(f"  Format: {format}")
        console.print(f"  Quantization: {quantize}")
        console.print(f"  Output: {output or paths.merged_dir}")
        console.print()

        # TODO: Implement actual export
        console.print("[yellow]Export not yet implemented.[/]")
