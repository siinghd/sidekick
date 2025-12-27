"""Start the Sidekick server."""

from typing import Optional

import typer
from rich.console import Console

from sidekick.core.config import get_settings
from sidekick.core.paths import get_paths
from sidekick.inference import detect_inference_mode

console = Console()


def serve_cmd(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to bind to",
    ),
    mcp: bool = typer.Option(
        False,
        "--mcp",
        help="Start MCP server instead of HTTP",
    ),
    proxy: bool = typer.Option(
        False,
        "--proxy",
        help="Start OpenAI-compatible proxy",
    ),
    upstream: str = typer.Option(
        "https://api.openai.com",
        "--upstream",
        help="Upstream API URL for proxy mode",
    ),
) -> None:
    """Start the Sidekick server.

    Modes:
    - Default: HTTP API server with /ask and OpenAI-compatible endpoints
    - MCP: Model Context Protocol server for Claude and MCP clients
    - Proxy: OpenAI-compatible proxy that injects personal context

    Examples:
        sidekick serve                    # Start HTTP API
        sidekick serve --port 8080        # Custom port
        sidekick serve --mcp              # MCP server (stdin/stdout)
        sidekick serve --proxy            # Proxy to OpenAI
        sidekick serve --proxy --upstream https://api.anthropic.com/v1
    """
    paths = get_paths()
    settings = get_settings()

    if not paths.is_initialized():
        console.print(
            "[red]Sidekick is not initialized.[/]\n"
            "Run [bold]sidekick init[/] first."
        )
        raise typer.Exit(1)

    # Check for trained model (except proxy mode which uses context injection)
    mode, model_path = detect_inference_mode()
    if mode == "none" and not proxy:
        console.print(
            "[yellow]No trained model found.[/]\n"
            "Run [bold]sidekick train[/] first."
        )
        raise typer.Exit(1)

    # Apply overrides
    actual_host = host or settings.server.host
    actual_port = port or settings.server.port

    if mcp:
        # MCP server uses stdin/stdout, ignore host/port
        console.print("[bold]Starting MCP server...[/]", err=True)
        console.print("[dim]Reading from stdin, writing to stdout[/]", err=True)

        from sidekick.server import run_mcp_server
        run_mcp_server()

    elif proxy:
        console.print("[bold]Starting proxy server...[/]")
        console.print(f"  Upstream: {upstream}")
        console.print(f"  Listen: http://{actual_host}:{actual_port}")

        if mode == "none":
            console.print("[yellow]Warning: No context file found. Proxy will not inject personal context.[/]")
            console.print("Run [bold]sidekick train[/] to generate context.\n")

        console.print("\n[dim]Press Ctrl+C to stop[/]\n")

        from sidekick.server import run_proxy_server
        run_proxy_server(actual_host, actual_port, upstream)

    else:
        console.print("[bold]Starting HTTP server...[/]")
        console.print(f"  Listen: http://{actual_host}:{actual_port}")

        if mode == "gguf":
            console.print(f"  Model: {model_path.name}")
        else:
            console.print(f"  Mode: {mode}")

        console.print("\n[bold]Endpoints:[/]")
        console.print(f"  GET  http://{actual_host}:{actual_port}/           - API info")
        console.print(f"  GET  http://{actual_host}:{actual_port}/health     - Health check")
        console.print(f"  GET  http://{actual_host}:{actual_port}/ask?q=...  - Ask a question")
        console.print(f"  POST http://{actual_host}:{actual_port}/ask        - Ask a question")
        console.print(f"  POST http://{actual_host}:{actual_port}/v1/chat/completions  - OpenAI compat")

        console.print("\n[dim]Press Ctrl+C to stop[/]\n")

        from sidekick.server import run_server
        run_server(actual_host, actual_port)
