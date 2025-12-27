"""OpenAI-compatible proxy server with context injection.

This proxy sits between clients and upstream APIs (OpenAI, Anthropic, etc.)
and automatically injects the user's personal context into requests.
"""

import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from sidekick.train.prompt_tuned import get_context_prompt
from sidekick.utils.fast_json import dumps, loads


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the OpenAI-compatible proxy."""

    # Configuration (set by server)
    upstream_url: str = "https://api.openai.com"
    context: str | None = None

    def log_message(self, format: str, *args: Any) -> None:
        """Override to suppress default logging."""
        pass

    def _forward_headers(self) -> dict[str, str]:
        """Extract headers to forward to upstream."""
        headers = {}
        for key, value in self.headers.items():
            # Forward authentication and content headers
            key_lower = key.lower()
            if key_lower in ("authorization", "content-type", "x-api-key"):
                headers[key] = value
        return headers

    def _inject_context(self, messages: list[dict]) -> list[dict]:
        """Inject personal context into the messages.

        Args:
            messages: Original messages list

        Returns:
            Messages with context injected
        """
        if not self.context:
            return messages

        # Find if there's already a system message
        has_system = any(m.get("role") == "system" for m in messages)

        context_text = f"""You have access to personal context about the user. Use this information to personalize your responses:

{self.context}

Remember to use this context when answering questions about the user, their preferences, projects, or history."""

        if has_system:
            # Prepend context to existing system message
            result = []
            for msg in messages:
                if msg.get("role") == "system":
                    original = msg.get("content", "")
                    msg = dict(msg)
                    msg["content"] = f"{context_text}\n\n{original}"
                result.append(msg)
            return result
        else:
            # Add a new system message with context
            return [
                {"role": "system", "content": context_text},
                *messages,
            ]

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON response."""
        body = dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        """Send an error response."""
        self._send_json({"error": {"message": message, "type": "proxy_error"}}, status)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Api-Key")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/":
            self._send_json({
                "name": "sidekick-proxy",
                "version": "0.1.0",
                "description": "OpenAI-compatible proxy with personal context injection",
                "upstream": self.upstream_url,
                "context_loaded": self.context is not None,
            })
        elif self.path == "/v1/models":
            # Forward to upstream
            self._proxy_request()
        else:
            self._send_error("Not found", 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path.startswith("/v1/"):
            # Forward other v1 requests
            self._proxy_request()
        else:
            self._send_error("Not found", 404)

    def _handle_chat_completions(self) -> None:
        """Handle chat completions with context injection."""
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else ""

        try:
            data = loads(body) if body else {}
        except (ValueError, TypeError):
            self._send_error("Invalid JSON", 400)
            return

        # Inject context into messages
        if "messages" in data:
            data["messages"] = self._inject_context(data["messages"])

        # Forward to upstream
        self._proxy_request(dumps(data).encode("utf-8"))

    def _proxy_request(self, body: bytes | None = None) -> None:
        """Proxy a request to the upstream API.

        Args:
            body: Request body (if None, reads from input)
        """
        if body is None:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else None

        # Build upstream URL
        upstream = f"{self.upstream_url}{self.path}"

        # Prepare request
        headers = self._forward_headers()
        method = self.command

        try:
            request = urllib.request.Request(
                upstream,
                data=body,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                # Forward response
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()

                # Stream response body
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        except urllib.error.HTTPError as e:
            # Forward error response
            self.send_response(e.code)
            for key, value in e.headers.items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(e.read())

        except urllib.error.URLError as e:
            self._send_error(f"Cannot connect to upstream: {e}", 502)

        except Exception as e:
            self._send_error(f"Proxy error: {e}", 500)


def create_proxy_server(
    host: str = "127.0.0.1",
    port: int = 8766,
    upstream: str = "https://api.openai.com",
) -> HTTPServer:
    """Create a Sidekick proxy server.

    Args:
        host: Host to bind to
        port: Port to bind to
        upstream: Upstream API URL

    Returns:
        HTTPServer instance
    """
    # Load context
    context = get_context_prompt()

    # Configure handler
    ProxyHandler.upstream_url = upstream.rstrip("/")
    ProxyHandler.context = context

    server = HTTPServer((host, port), ProxyHandler)
    return server


def run_proxy_server(
    host: str = "127.0.0.1",
    port: int = 8766,
    upstream: str = "https://api.openai.com",
) -> None:
    """Run the Sidekick proxy server.

    Args:
        host: Host to bind to
        port: Port to bind to
        upstream: Upstream API URL
    """
    server = create_proxy_server(host, port, upstream)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
