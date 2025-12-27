"""List ingested sources."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from sidekick.core.paths import get_paths
from sidekick.ingest import get_manifest_manager

console = Console()


def list_cmd(
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Filter by tag",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        "-s",
        help="Show statistics only",
    ),
) -> None:
    """List all ingested sources.

    Examples:
        sidekick list
        sidekick list --tag work
        sidekick list --verbose
        sidekick list --stats
    """
    paths = get_paths()

    if not paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    manifest = get_manifest_manager()

    # Show stats only
    if stats:
        _show_stats(manifest)
        return

    # Get sources
    if tag:
        sources = manifest.get_sources_by_tag(tag)
    else:
        sources = manifest.get_all_sources()

    if not sources:
        console.print("[dim]No sources ingested yet.[/]")
        console.print("Use [bold]sidekick add[/] to add files or text.")
        return

    # Build table
    table = Table(title=f"Ingested Sources ({len(sources)})")
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Type", style="cyan")
    table.add_column("Source", max_width=40)
    table.add_column("Tags")
    table.add_column("Status", justify="center")

    if verbose:
        table.add_column("Size", justify="right")
        table.add_column("Added")

    for source in sorted(sources, key=lambda s: s.added_at, reverse=True):
        # Source name
        if source.path:
            source_name = source.path.name
        else:
            source_name = "[dim]<text>[/]"

        # Tags
        tags_str = ", ".join(source.tags) if source.tags else "[dim]-[/]"

        # Status
        if source.processed:
            status = "[green]✓[/]"
        else:
            status = "[yellow]pending[/]"

        row = [
            source.id[:8],
            source.source_type.value,
            source_name,
            tags_str,
            status,
        ]

        if verbose:
            # Size
            size_kb = source.size_bytes / 1024
            if size_kb < 1:
                size_str = f"{source.size_bytes}B"
            elif size_kb < 1024:
                size_str = f"{size_kb:.1f}KB"
            else:
                size_str = f"{size_kb/1024:.1f}MB"

            # Date
            date_str = source.added_at.strftime("%Y-%m-%d %H:%M")

            row.extend([size_str, date_str])

        table.add_row(*row)

    console.print(table)

    # Show summary
    console.print()
    _show_stats(manifest)


def _show_stats(manifest: "ManifestManager") -> None:  # type: ignore
    """Show statistics about ingested sources."""
    stats = manifest.get_stats()

    total = stats["total_sources"]
    processed = stats["processed"]
    unprocessed = stats["unprocessed"]
    qa_pairs = stats["qa_pairs"]

    # Format size
    total_bytes = stats["total_bytes"]
    if total_bytes < 1024:
        size_str = f"{total_bytes}B"
    elif total_bytes < 1024 * 1024:
        size_str = f"{total_bytes/1024:.1f}KB"
    else:
        size_str = f"{total_bytes/1024/1024:.1f}MB"

    console.print(f"[bold]Statistics:[/]")
    console.print(f"  Sources: {total} ({size_str})")
    console.print(f"  Processed: {processed} / {total}")
    if qa_pairs > 0:
        console.print(f"  QA Pairs: {qa_pairs}")
