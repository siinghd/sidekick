"""Query the Sidekick model."""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from sidekick.core.paths import get_paths
from sidekick.inference import (
    InferenceConfig,
    detect_inference_mode,
    get_inference_backend,
)

console = Console()


def ask_cmd(
    query: str = typer.Argument(
        ...,
        help="Question to ask",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Inference backend: llamacpp, ollama, prompt_tuned (auto-detected if not specified)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show backend, latency, and token stats",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature (0.0-2.0)",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        "-m",
        help="Maximum tokens to generate",
    ),
) -> None:
    """Ask the Sidekick model a question.

    Uses your trained personal model to answer questions about you.

    Examples:
        sidekick ask "What projects am I working on?"
        sidekick ask "Who is my photographer friend?" --verbose
        sidekick ask "What's my tech stack?" --json
        sidekick ask "Summarize my career" --max-tokens 1000
    """
    paths = get_paths()

    if not paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    # Detect inference mode
    mode, model_path = detect_inference_mode()

    if mode == "none":
        console.print(
            "[yellow]No trained model found.[/]\n"
            "Run [bold]sidekick train[/] first."
        )
        raise typer.Exit(1)

    # Show what we're using if verbose
    if verbose and not json_output:
        if mode == "gguf":
            console.print(f"[dim]Using GGUF model: {model_path.name}[/]")
        else:
            console.print(f"[dim]Using {mode} mode[/]")

    # Configure inference
    config = InferenceConfig(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        # Get the appropriate backend
        inference = get_inference_backend(backend=backend, config=config)

        # Check availability
        available, msg = inference.is_available()
        if not available:
            console.print(f"[red]Backend not available:[/] {msg}")
            raise typer.Exit(1)

        # Generate response
        if verbose and not json_output:
            console.print(f"[dim]Backend: {inference.name}[/]")
            console.print()

        result = inference.generate(query)

        if result.success:
            if json_output:
                output = {
                    "query": query,
                    "response": result.response,
                    "backend": inference.name,
                    "latency_ms": round(result.latency_ms, 2),
                    "tokens_generated": result.tokens_generated,
                    "tokens_per_second": round(result.tokens_per_second, 2),
                }
                console.print(json.dumps(output, indent=2))
            else:
                console.print(
                    Panel(
                        result.response,
                        title="[bold]Response[/]",
                        border_style="green",
                    )
                )

                if verbose:
                    console.print()
                    console.print(f"[dim]Latency: {result.latency_ms:.0f}ms[/]")
                    if result.tokens_generated > 0:
                        console.print(
                            f"[dim]Tokens: {result.tokens_generated} "
                            f"({result.tokens_per_second:.1f} tok/s)[/]"
                        )
        else:
            if json_output:
                output = {
                    "query": query,
                    "error": result.error,
                    "success": False,
                }
                console.print(json.dumps(output, indent=2))
            else:
                console.print(f"[red]Error:[/] {result.error}")
            raise typer.Exit(1)

    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
