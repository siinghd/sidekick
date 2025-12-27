"""Train the Sidekick model."""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from sidekick.core.config import TrainingMode, get_settings
from sidekick.core.paths import get_paths
from sidekick.ingest import get_manifest_manager
from sidekick.qa import get_generator, get_qa_dataset
from sidekick.train import (
    TrainingBackend,
    TrainingConfig,
    check_backend_available,
    detect_hardware,
    get_trainer,
)

console = Console()


def train_cmd(
    mode: Optional[TrainingMode] = typer.Option(
        None,
        "--mode",
        "-m",
        help="Training mode: auto, local, cloud, or prompt",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model to use (e.g., qwen2.5-1.5b)",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Only process new sources (default) or reprocess all",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume training from existing adapter (incremental model training)",
    ),
    generate_only: bool = typer.Option(
        False,
        "--generate-only",
        "-g",
        help="Only generate QA pairs, don't train model",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="QA generation backend: local, openai, anthropic, ollama",
    ),
    skip_generate: bool = typer.Option(
        False,
        "--skip-generate",
        help="Skip QA generation, use existing dataset",
    ),
) -> None:
    """Train the Sidekick model on ingested data.

    Examples:
        sidekick train                      # Auto-detect and train
        sidekick train --resume             # Continue training from existing adapter
        sidekick train --generate-only      # Just generate QA pairs
        sidekick train --backend local      # Use local model for QA
        sidekick train --mode prompt        # Use prompt-tuned mode
        sidekick train --skip-generate      # Train on existing QA data
    """
    paths = get_paths()
    settings = get_settings()

    if not paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    # Apply overrides
    if base:
        settings.model.base = base
    if epochs:
        settings.training.epochs = epochs

    manifest = get_manifest_manager()
    qa_dataset = get_qa_dataset()

    # ─────────────────────────────────────────────────────────────
    # Step 1: QA Generation
    # ─────────────────────────────────────────────────────────────
    if not skip_generate:
        # Get sources to process
        if incremental:
            sources = manifest.get_unprocessed_sources()
        else:
            sources = manifest.get_all_sources()

        if sources:
            console.print("[bold]Step 1: QA Generation[/]")
            console.print(f"  Sources to process: {len(sources)}")

            # Initialize QA generator
            try:
                generator = get_generator(backend)
                console.print(f"  Backend: {generator.backend_name}")
            except Exception as e:
                console.print(f"[red]Error:[/] {e}")
                raise typer.Exit(1)

            console.print()

            # Generate QA pairs
            total_generated = 0
            total_added = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating...", total=len(sources))

                for source in sources:
                    processed_file = paths.processed_dir / f"{source.id}.txt"
                    if not processed_file.exists():
                        progress.advance(task)
                        continue

                    text = processed_file.read_text()
                    if not text.strip():
                        progress.advance(task)
                        continue

                    name = source.path.name if source.path else source.id[:8]
                    text_size = len(text)
                    size_info = f"({text_size:,} chars)"

                    # Show size for large files
                    if text_size > 12000:
                        num_chunks = (text_size // 12000) + 1
                        progress.update(
                            task,
                            description=f"Processing {name} {size_info} ~{num_chunks} chunks..."
                        )
                    else:
                        progress.update(task, description=f"Processing {name}...")

                    result = generator.generate(text, source.id)

                    if result.error:
                        console.print(f"  [red]Error:[/] {name}: {result.error}")
                    elif result.qa_pairs:
                        added = qa_dataset.add_pairs(result.qa_pairs, dedupe=True)
                        total_generated += len(result.qa_pairs)
                        total_added += added
                        if added > 0:
                            console.print(f"  [green]+{added}[/] from {name}")

                    manifest.mark_processed(source.id)
                    progress.advance(task)

            manifest.update_qa_count(qa_dataset.count())

            console.print()
            console.print(f"  Generated: {total_generated} | Added: {total_added}")
        else:
            console.print("[dim]No new sources to process.[/]")

    # Check if we have data to train on
    total_qa = qa_dataset.count()
    console.print(f"\n[bold]QA Dataset:[/] {total_qa} pairs")

    if total_qa == 0:
        console.print("[yellow]No QA pairs available for training.[/]")
        console.print("Add sources with [bold]sidekick add[/] first.")
        raise typer.Exit(0)

    if generate_only:
        console.print(f"\n[dim]Dataset saved to: {qa_dataset.path}[/]")
        return

    # ─────────────────────────────────────────────────────────────
    # Step 2: Hardware Detection
    # ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 2: Hardware Detection[/]")

    hw_info = detect_hardware()
    console.print(f"  {hw_info.details}")

    # Determine training backend
    if mode == TrainingMode.PROMPT:
        train_backend = TrainingBackend.PROMPT_TUNED
    elif mode == TrainingMode.LOCAL:
        train_backend = hw_info.recommended_backend
    else:
        train_backend = hw_info.recommended_backend

    console.print(f"  Training backend: {train_backend.value}")

    # Check if backend is available
    available, msg = check_backend_available(train_backend)
    if not available:
        console.print(f"[yellow]{msg}[/]")
        if train_backend != TrainingBackend.PROMPT_TUNED:
            console.print("Falling back to prompt-tuned mode...")
            train_backend = TrainingBackend.PROMPT_TUNED

    # ─────────────────────────────────────────────────────────────
    # Step 3: Model Training
    # ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 3: Model Training[/]")

    # Check for existing adapter to resume from
    resume_from = None
    if resume:
        mlx_adapter = paths.adapters_dir / "mlx_adapter"
        if mlx_adapter.exists() and (mlx_adapter / "adapters.safetensors").exists():
            resume_from = mlx_adapter
            console.print(f"  [cyan]Resuming from:[/] {mlx_adapter}")
        else:
            console.print("  [yellow]No existing adapter found, training from scratch[/]")

    config = TrainingConfig(
        base_model=settings.model.base,
        epochs=settings.training.epochs,
        learning_rate=settings.training.learning_rate,
        batch_size=settings.training.batch_size,
        lora_rank=settings.training.lora_rank,
        lora_alpha=settings.training.lora_alpha,
        output_dir=paths.adapters_dir,
        quantization=settings.model.quantization,
        resume_from=resume_from,
    )

    console.print(f"  Base model: {config.base_model}")
    console.print(f"  Epochs: {config.epochs}")
    console.print(f"  QA pairs: {total_qa}")

    if total_qa < 10:
        console.print(
            "\n[yellow]Warning: Small dataset may not train well.[/]"
        )

    console.print()

    # Get trainer and run
    trainer = get_trainer(train_backend, config)
    qa_pairs = qa_dataset.get_all()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Training with {trainer.name}...", total=None)

        result = trainer.train(qa_pairs)

    # ─────────────────────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────────────────────
    console.print()

    if result.success:
        time_str = f"{result.training_time_seconds:.1f}s"

        console.print(
            Panel.fit(
                f"[green]Training complete![/]\n\n"
                f"Backend: {trainer.name}\n"
                f"Time: {time_str}\n"
                f"QA pairs: {total_qa}\n"
                + (f"Output: {result.model_path or result.adapter_path}" if result.model_path or result.adapter_path else ""),
                title="[bold]Success[/]",
                border_style="green",
            )
        )

        if train_backend == TrainingBackend.PROMPT_TUNED:
            console.print(
                "\n[dim]Prompt-tuned mode: Context file created. "
                "Use 'sidekick ask' to query.[/]"
            )
    else:
        console.print(f"[red]Training failed:[/] {result.error}")
        raise typer.Exit(1)
