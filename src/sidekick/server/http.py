"""HTTP server for Sidekick API."""

import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from sidekick.inference import InferenceConfig, get_inference_backend
from sidekick.utils.fast_json import dumps, loads


class SidekickHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Sidekick API."""

    # Shared inference backend (set by server)
    inference_backend = None

    def log_message(self, format: str, *args: Any) -> None:
        """Override to suppress default logging."""
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON response."""
        body = dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        """Send an error response."""
        self._send_json({"error": message, "success": False}, status)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._handle_root()
        elif path == "/health":
            self._handle_health()
        elif path == "/v1/models":
            self._handle_models()
        elif path == "/ask":
            query = parse_qs(parsed.query)
            q = query.get("q", [""])[0]
            if q:
                self._handle_ask(q)
            else:
                self._send_error("Missing 'q' parameter", 400)
        else:
            self._send_error("Not found", 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else ""

        try:
            data = loads(body) if body else {}
        except (ValueError, TypeError):
            self._send_error("Invalid JSON", 400)
            return

        if path == "/ask":
            query = data.get("query") or data.get("q")
            if query:
                self._handle_ask(query, data)
            else:
                self._send_error("Missing 'query' field", 400)
        elif path == "/v1/chat/completions":
            self._handle_chat_completions(data)
        else:
            self._send_error("Not found", 404)

    def _handle_root(self) -> None:
        """Handle root endpoint."""
        self._send_json({
            "name": "sidekick",
            "version": "0.1.0",
            "description": "Personal memory model API",
            "endpoints": [
                {"path": "/health", "method": "GET"},
                {"path": "/ask", "methods": ["GET", "POST"]},
                {"path": "/v1/models", "method": "GET"},
                {"path": "/v1/chat/completions", "method": "POST"},
            ],
        })

    def _handle_health(self) -> None:
        """Handle health check endpoint."""
        backend_name = self.inference_backend.name if self.inference_backend else "none"
        available, msg = self.inference_backend.is_available() if self.inference_backend else (False, "No backend")

        self._send_json({
            "status": "healthy" if available else "degraded",
            "backend": backend_name,
            "message": msg,
        })

    def _handle_models(self) -> None:
        """Handle OpenAI-compatible models endpoint."""
        self._send_json({
            "object": "list",
            "data": [
                {
                    "id": "sidekick",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sidekick",
                },
            ],
        })

    def _handle_ask(self, query: str, options: dict = None) -> None:
        """Handle ask endpoint."""
        options = options or {}

        if not self.inference_backend:
            self._send_error("No inference backend available", 503)
            return

        # Generate response
        result = self.inference_backend.generate(query)

        if result.success:
            self._send_json({
                "success": True,
                "query": query,
                "response": result.response,
                "backend": self.inference_backend.name,
                "latency_ms": round(result.latency_ms, 2),
                "tokens_generated": result.tokens_generated,
            })
        else:
            self._send_error(result.error or "Generation failed", 500)

    def _handle_chat_completions(self, data: dict) -> None:
        """Handle OpenAI-compatible chat completions endpoint."""
        messages = data.get("messages", [])
        if not messages:
            self._send_error("Missing 'messages' field", 400)
            return

        # Extract the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            self._send_error("No user message found", 400)
            return

        if not self.inference_backend:
            self._send_error("No inference backend available", 503)
            return

        # Generate response
        result = self.inference_backend.generate(user_message)

        if result.success:
            self._send_json({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "sidekick",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.response,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # We don't track this
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": result.tokens_generated,
                },
            })
        else:
            self._send_error(result.error or "Generation failed", 500)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    config: InferenceConfig | None = None,
) -> HTTPServer:
    """Create a Sidekick HTTP server.

    Args:
        host: Host to bind to
        port: Port to bind to
        config: Inference configuration

    Returns:
        HTTPServer instance
    """
    config = config or InferenceConfig()

    # Initialize inference backend
    try:
        backend = get_inference_backend(config=config)
        SidekickHandler.inference_backend = backend
    except ValueError as e:
        raise RuntimeError(f"Cannot initialize inference backend: {e}")

    server = HTTPServer((host, port), SidekickHandler)
    return server


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    config: InferenceConfig | None = None,
) -> None:
    """Run the Sidekick HTTP server.

    Args:
        host: Host to bind to
        port: Port to bind to
        config: Inference configuration
    """
    server = create_server(host, port, config)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
