"""Tests for proxy server module."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from sidekick.server.proxy import ProxyHandler


class MockRequest:
    """Mock HTTP request for testing."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/",
        body: bytes = b"",
        headers: dict = None,
    ):
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers or {}


class MockProxyHandler(ProxyHandler):
    """Mock proxy handler for testing without network."""

    def __init__(self, request: MockRequest):
        self.request = request
        self.path = request.path
        self.command = request.method
        self.headers = request.headers
        self.rfile = BytesIO(request.body)
        self.wfile = BytesIO()
        self._response_code = None
        self._response_headers = {}

    def send_response(self, code: int) -> None:
        self._response_code = code

    def send_header(self, key: str, value: str) -> None:
        self._response_headers[key] = value

    def end_headers(self) -> None:
        pass

    def get_response(self) -> tuple[int, dict, bytes]:
        """Get the response data."""
        return (
            self._response_code,
            self._response_headers,
            self.wfile.getvalue(),
        )

    def get_json_response(self) -> tuple[int, dict]:
        """Get the response as JSON."""
        code, headers, body = self.get_response()
        return code, json.loads(body.decode("utf-8"))


class TestProxyHandler:
    """Tests for ProxyHandler."""

    def test_root_endpoint(self) -> None:
        """Test root endpoint returns proxy info."""
        handler = MockProxyHandler(MockRequest("GET", "/"))
        handler.upstream_url = "https://api.openai.com"
        handler.context = None
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["name"] == "sidekick-proxy"
        assert data["upstream"] == "https://api.openai.com"
        assert data["context_loaded"] is False

    def test_root_with_context(self) -> None:
        """Test root endpoint shows context loaded."""
        handler = MockProxyHandler(MockRequest("GET", "/"))
        handler.upstream_url = "https://api.openai.com"
        handler.context = "# Personal Context"
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["context_loaded"] is True

    def test_inject_context_no_system(self) -> None:
        """Test context injection without existing system message."""
        handler = MockProxyHandler(MockRequest())
        handler.context = "User name is John."

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        result = handler._inject_context(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "John" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_inject_context_with_system(self) -> None:
        """Test context injection with existing system message."""
        handler = MockProxyHandler(MockRequest())
        handler.context = "User name is John."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        result = handler._inject_context(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        # Context should be prepended to existing system message
        assert "John" in result[0]["content"]
        assert "helpful assistant" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_inject_context_no_context(self) -> None:
        """Test context injection when no context available."""
        handler = MockProxyHandler(MockRequest())
        handler.context = None

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        result = handler._inject_context(messages)

        # Should return messages unchanged
        assert result == messages

    def test_forward_headers(self) -> None:
        """Test header forwarding."""
        handler = MockProxyHandler(MockRequest(
            headers={
                "Authorization": "Bearer sk-xxx",
                "Content-Type": "application/json",
                "X-Api-Key": "key123",
                "User-Agent": "test",  # Should not be forwarded
            }
        ))

        forwarded = handler._forward_headers()

        assert forwarded.get("Authorization") == "Bearer sk-xxx"
        assert forwarded.get("Content-Type") == "application/json"
        assert forwarded.get("X-Api-Key") == "key123"
        assert "User-Agent" not in forwarded

    def test_cors_options(self) -> None:
        """Test CORS preflight handling."""
        handler = MockProxyHandler(MockRequest("OPTIONS", "/v1/chat/completions"))
        handler.do_OPTIONS()

        assert handler._response_code == 200
        assert handler._response_headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in handler._response_headers.get("Access-Control-Allow-Methods", "")

    def test_not_found(self) -> None:
        """Test 404 for unknown endpoint."""
        handler = MockProxyHandler(MockRequest("GET", "/unknown"))
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 404
        assert "error" in data
