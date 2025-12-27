"""Watch files for changes and update hot memory."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from sidekick.core.config import get_settings
from sidekick.core.paths import get_paths

console = Console()


def watch_cmd(
    paths: Optional[list[Path]] = typer.Argument(
        None,
        help="Directories to watch (default: current directory)",
    ),
    patterns: Optional[str] = typer.Option(
        None,
        "--patterns",
        "-p",
        help="Comma-separated file patterns (e.g., '*.md,*.txt')",
    ),
    train_interval: int = typer.Option(
        3600,
        "--train-interval",
        "-t",
        help="Seconds between incremental training (0 to disable)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed status",
    ),
) -> None:
    """Watch files for changes and learn incrementally.

    Monitors specified directories for file changes, extracts facts
    from modified files, and periodically trains the model.

    Examples:
        sidekick watch                      # Watch current directory
        sidekick watch ~/notes ~/docs       # Watch multiple directories
        sidekick watch -p "*.md,*.org"      # Custom patterns
        sidekick watch --train-interval 0   # Disable auto-training
    """
    sidekick_paths = get_paths()
    settings = get_settings()

    if not sidekick_paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    # Default to current directory
    watch_paths = list(paths) if paths else [Path.cwd()]

    # Validate paths exist
    for p in watch_paths:
        if not p.exists():
            console.print(f"[red]Path does not exist:[/] {p}")
            raise typer.Exit(1)
        if not p.is_dir():
            console.print(f"[red]Not a directory:[/] {p}")
            raise typer.Exit(1)

    # Parse patterns
    file_patterns = (
        patterns.split(",") if patterns
        else settings.watch.patterns
    )

    console.print("[bold]Starting watch mode...[/]")
    console.print(f"  Directories: {', '.join(str(p) for p in watch_paths)}")
    console.print(f"  Patterns: {', '.join(file_patterns)}")
    console.print(f"  Train interval: {train_interval}s" if train_interval else "  Auto-training: disabled")
    console.print("\n[dim]Press Ctrl+C to stop[/]\n")

    # Run async watcher
    try:
        asyncio.run(_watch_async(
            watch_paths=watch_paths,
            patterns=file_patterns,
            settings=settings,
            sidekick_dir=sidekick_paths.root,
            train_interval=train_interval,
            verbose=verbose,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Watch stopped.[/]")


async def _watch_async(
    watch_paths: list[Path],
    patterns: list[str],
    settings,
    sidekick_dir: Path,
    train_interval: int,
    verbose: bool,
) -> None:
    """Async watch implementation."""
    try:
        from watchfiles import awatch, Change
    except ImportError:
        console.print(
            "[red]watchfiles not installed.[/]\n"
            "Install with: [bold]pip install watchfiles[/]"
        )
        return

    from sidekick.watch import (
        BackgroundTrainer,
        FileWatcher,
        HotMemory,
    )

    # Initialize components
    memory = HotMemory(
        max_entries=settings.hot_memory.max_entries,
        max_bytes=settings.hot_memory.max_bytes,
        max_age_hours=settings.hot_memory.max_age_hours,
        context_cache_ttl=settings.hot_memory.context_cache_ttl,
    )

    trainer = BackgroundTrainer(sidekick_dir)

    # Stats
    stats = {
        "files_processed": 0,
        "facts_extracted": 0,
        "batches_processed": 0,
        "last_train": None,
    }

    async def process_batch(changes):
        """Process a batch of file changes."""
        from sidekick.qa import get_generator

        stats["batches_processed"] += 1

        # Combine file contents for batch LLM call
        combined_text = "\n\n---\n\n".join(
            f"[{c.path.name}]\n{c.content[:2000]}"
            for c in changes
        )

        try:
            # Use QA generator to extract facts
            generator = get_generator()

            # Simple fact extraction prompt
            prompt = f"""Extract key facts from the following content.
Output each fact on a new line, starting with "- ".

{combined_text}

Facts:"""

            # This is simplified - in production you'd use proper fact extraction
            # For now we just add the content summary as facts
            for change in changes:
                # Extract simple facts from the content
                lines = change.content.split("\n")
                for line in lines[:10]:  # First 10 lines
                    line = line.strip()
                    if line and len(line) > 20 and len(line) < 500:
                        memory.add(line, str(change.path.name))
                        stats["facts_extracted"] += 1

                stats["files_processed"] += 1

            if verbose:
                console.print(f"[green]+{len(changes)}[/] files processed")

        except Exception as e:
            if verbose:
                console.print(f"[red]Error processing batch:[/] {e}")

    # Create watcher
    watcher = FileWatcher(
        paths=watch_paths,
        patterns=patterns,
        ignore=settings.watch.ignore,
        max_file_size_kb=settings.watch.max_file_size_kb,
        debounce_ms=settings.watch.debounce_ms,
        rate_limit_per_minute=settings.watch.rate_limit_per_minute,
        batch_size=settings.watch.batch_size,
        batch_wait_ms=settings.watch.batch_wait_ms,
    )

    # Training timer
    last_train_time = asyncio.get_event_loop().time()

    async def maybe_train():
        nonlocal last_train_time

        if train_interval <= 0:
            return

        now = asyncio.get_event_loop().time()
        if now - last_train_time < train_interval:
            return

        if trainer.is_training:
            return

        entries = memory.flush()
        if entries:
            console.print(f"[blue]Starting incremental training ({len(entries)} facts)...[/]")

            def on_status(status):
                if verbose:
                    console.print(f"[dim]  {status.message}[/]")

            await trainer.train(entries, on_status=on_status)
            stats["last_train"] = now

        last_train_time = now

    # Status display
    async def show_status():
        while True:
            await asyncio.sleep(30)
            await maybe_train()

            if verbose:
                mem_stats = memory.stats()
                console.print(
                    f"[dim]Stats: {stats['files_processed']} files, "
                    f"{stats['facts_extracted']} facts, "
                    f"{mem_stats['entries']} in memory[/]"
                )

    # Start status task
    status_task = asyncio.create_task(show_status())

    try:
        await watcher.watch(on_batch=process_batch)
    finally:
        status_task.cancel()
        trainer.cleanup()

        # Final stats
        console.print(f"\n[bold]Session summary:[/]")
        console.print(f"  Files processed: {stats['files_processed']}")
        console.print(f"  Facts extracted: {stats['facts_extracted']}")
        console.print(f"  Batches: {stats['batches_processed']}")
