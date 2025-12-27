"""Add data to Sidekick."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from sidekick.core.paths import get_paths
from sidekick.ingest import get_ingester, get_manifest_manager
from sidekick.ingest.base import IngestResult

console = Console()


def add_cmd(
    sources: list[str] = typer.Argument(
        ...,
        help="Files, directories, or text to add",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Tag to apply to added sources",
    ),
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read from stdin",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help="Recursively add directories",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-ingest even if source already exists",
    ),
) -> None:
    """Add files, directories, or text to Sidekick.

    Examples:
        sidekick add ./notes/
        sidekick add document.pdf --tag work
        sidekick add "I work at Acme Corp"
        cat notes.txt | sidekick add --stdin
    """
    paths = get_paths()

    if not paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    tags = [tag] if tag else []
    ingester = get_ingester()
    manifest = get_manifest_manager()

    added_count = 0
    skipped_count = 0

    # Handle stdin
    if stdin:
        content = sys.stdin.read()
        if content.strip():
            result = ingester.ingest_text(content, tags)
            if _save_result(result, manifest):
                added_count += 1
                _print_added(result, "stdin")
            else:
                skipped_count += 1
        _print_summary(added_count, skipped_count)
        return

    for source in sources:
        path = Path(source)

        if path.exists():
            if path.is_file():
                result = _add_file(path, tags, ingester, manifest, force)
                if result:
                    added_count += 1
                else:
                    skipped_count += 1
            elif path.is_dir():
                a, s = _add_directory(path, tags, recursive, ingester, manifest, force)
                added_count += a
                skipped_count += s
        else:
            # Treat as raw text
            result = ingester.ingest_text(source, tags)
            if _save_result(result, manifest, force):
                added_count += 1
                _print_added(result, "text")
            else:
                skipped_count += 1

    _print_summary(added_count, skipped_count)


def _add_file(
    path: Path,
    tags: list[str],
    ingester: "AutoIngester",  # type: ignore
    manifest: "ManifestManager",  # type: ignore
    force: bool = False,
) -> bool:
    """Add a single file.

    Returns:
        True if added, False if skipped (duplicate)
    """
    try:
        result = ingester.ingest_file(path, tags)
        if _save_result(result, manifest, force):
            _print_added(result, str(path))
            return True
        else:
            console.print(f"  [dim]Skipped (duplicate):[/] {path}")
            return False
    except Exception as e:
        console.print(f"  [red]Error:[/] {path}: {e}")
        return False


def _add_directory(
    path: Path,
    tags: list[str],
    recursive: bool,
    ingester: "AutoIngester",  # type: ignore
    manifest: "ManifestManager",  # type: ignore
    force: bool = False,
) -> tuple[int, int]:
    """Add all files in a directory.

    Returns:
        Tuple of (added_count, skipped_count)
    """
    added = 0
    skipped = 0
    pattern = "**/*" if recursive else "*"

    for file_path in sorted(path.glob(pattern)):
        if file_path.is_file() and not file_path.name.startswith("."):
            if _add_file(file_path, tags, ingester, manifest, force):
                added += 1
            else:
                skipped += 1

    return added, skipped


def _save_result(result: IngestResult, manifest: "ManifestManager", force: bool = False) -> bool:  # type: ignore
    """Save an ingestion result to manifest and disk.

    Returns:
        True if saved (new or forced), False if duplicate
    """
    paths = get_paths()

    # Check for duplicate (skip if force)
    if manifest.source_exists(result.source.content_hash):
        if not force:
            return False
        # Force: remove old source and re-add
        manifest.remove_source(result.source.content_hash)

    # Save extracted text
    processed_file = paths.processed_dir / f"{result.source.id}.txt"
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    processed_file.write_text(result.text, encoding="utf-8")

    # Add to manifest
    manifest.add_source(result.source)

    return True


def _print_added(result: IngestResult, source_name: str) -> None:
    """Print info about an added source."""
    source = result.source
    size_kb = source.size_bytes / 1024
    type_str = source.source_type.value

    if size_kb < 1:
        size_str = f"{source.size_bytes}B"
    elif size_kb < 1024:
        size_str = f"{size_kb:.1f}KB"
    else:
        size_str = f"{size_kb/1024:.1f}MB"

    console.print(f"  [green]+[/] {source_name} [dim]({type_str}, {size_str})[/]")


def _print_summary(added: int, skipped: int) -> None:
    """Print summary of add operation."""
    console.print()
    if added > 0:
        console.print(f"[green]Added {added} source(s)[/]")
    if skipped > 0:
        console.print(f"[dim]Skipped {skipped} duplicate(s)[/]")
    if added == 0 and skipped == 0:
        console.print("[yellow]No sources added[/]")
