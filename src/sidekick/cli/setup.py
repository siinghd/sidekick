"""Setup optional dependencies."""

import os
import platform
import shutil
import subprocess
import sys

import typer
from rich.console import Console

console = Console()


def _get_pip_command() -> list[str]:
    """Get the appropriate pip command for the current environment.

    Prefers uv pip for uv-created venvs, falls back to python -m pip.
    """
    # Check if uv is available and we're in a venv
    if shutil.which("uv") and sys.prefix != sys.base_prefix:
        return ["uv", "pip", "install"]

    # Fall back to python -m pip
    return [sys.executable, "-m", "pip", "install"]


def setup_cmd(
    backend: str = typer.Argument(
        "auto",
        help="Backend to install: auto, llama, ollama, mlx",
    ),
    cpu_only: bool = typer.Option(
        False,
        "--cpu-only",
        help="Install without GPU acceleration",
    ),
) -> None:
    """Install optional backends for inference and training.

    This command installs the necessary dependencies based on your hardware.

    Examples:
        sidekick setup              # Auto-detect and install
        sidekick setup llama        # Install llama-cpp-python
        sidekick setup mlx          # Install MLX (Apple Silicon)
        sidekick setup --cpu-only   # Install without GPU support
    """
    system = platform.system()
    machine = platform.machine()

    is_apple_silicon = system == "Darwin" and machine == "arm64"
    is_linux = system == "Linux"

    if backend == "auto":
        if is_apple_silicon:
            _install_llama(metal=True)
            console.print("\n[dim]Tip: You can also install MLX for training:[/]")
            console.print("  [cyan]sidekick setup mlx[/]")
        elif is_linux:
            # Check for NVIDIA GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5
                )
                has_nvidia = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                has_nvidia = False

            if has_nvidia and not cpu_only:
                _install_llama(cuda=True)
            else:
                _install_llama()
        else:
            _install_llama(cpu_only=cpu_only)

    elif backend == "llama":
        if is_apple_silicon and not cpu_only:
            _install_llama(metal=True)
        else:
            _install_llama(cpu_only=cpu_only)

    elif backend == "mlx":
        if not is_apple_silicon:
            console.print("[red]MLX is only available on Apple Silicon Macs.[/]")
            raise typer.Exit(1)
        _install_mlx()

    elif backend == "ollama":
        _setup_ollama()

    else:
        console.print(f"[red]Unknown backend:[/] {backend}")
        console.print("Available: auto, llama, mlx, ollama")
        raise typer.Exit(1)


def _install_llama(metal: bool = False, cuda: bool = False, cpu_only: bool = False) -> None:
    """Install llama-cpp-python with appropriate flags."""
    console.print("[bold]Installing llama-cpp-python...[/]")

    env = os.environ.copy()
    if metal:
        console.print("  [dim]With Metal acceleration (Apple Silicon)[/]")
        env["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
    elif cuda:
        console.print("  [dim]With CUDA acceleration (NVIDIA GPU)[/]")
        env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
    else:
        console.print("  [dim]CPU only[/]")

    pip_cmd = _get_pip_command()
    console.print(f"  [dim]Using: {' '.join(pip_cmd)}[/]")

    try:
        result = subprocess.run(
            [*pip_cmd, "llama-cpp-python"],
            env=env,
            check=False,
        )

        if result.returncode == 0:
            console.print("\n[green]llama-cpp-python installed successfully![/]")
            console.print("\nYou can now run:")
            console.print("  [cyan]sidekick train[/]")
        else:
            console.print("\n[red]Installation failed.[/]")
            console.print("\nTry installing manually:")
            if metal:
                console.print('  [cyan]CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install llama-cpp-python[/]')
            elif cuda:
                console.print('  [cyan]CMAKE_ARGS="-DLLAMA_CUBLAS=on" uv pip install llama-cpp-python[/]')
            else:
                console.print("  [cyan]uv pip install llama-cpp-python[/]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


def _install_mlx() -> None:
    """Install MLX for Apple Silicon training."""
    console.print("[bold]Installing MLX...[/]")

    pip_cmd = _get_pip_command()
    console.print(f"  [dim]Using: {' '.join(pip_cmd)}[/]")

    try:
        result = subprocess.run(
            [*pip_cmd, "mlx", "mlx-lm"],
            check=False,
        )

        if result.returncode == 0:
            console.print("\n[green]MLX installed successfully![/]")
            console.print("\nYou can now train with MLX:")
            console.print("  [cyan]sidekick train[/]")
        else:
            console.print("\n[red]Installation failed.[/]")
            console.print("\nTry installing manually:")
            console.print("  [cyan]uv pip install mlx mlx-lm[/]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


def _setup_ollama() -> None:
    """Provide instructions for Ollama setup."""
    console.print("[bold]Ollama Setup[/]\n")

    # Check if ollama is installed
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, timeout=5
        )
        ollama_installed = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ollama_installed = False

    if ollama_installed:
        console.print("[green]Ollama is installed![/]\n")
        console.print("Pull a model:")
        console.print("  [cyan]ollama pull qwen3:4b[/]\n")
        console.print("Then train with:")
        console.print("  [cyan]sidekick train --backend ollama[/]")
    else:
        console.print("Ollama is not installed.\n")
        console.print("Install from: [cyan]https://ollama.ai[/]\n")
        console.print("Or on macOS:")
        console.print("  [cyan]brew install ollama[/]\n")
        console.print("Then pull a model:")
        console.print("  [cyan]ollama pull qwen3:4b[/]")
