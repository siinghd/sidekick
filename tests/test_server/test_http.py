"""Tests for HTTP server module."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from sidekick.inference import InferenceResult
from sidekick.server.http import SidekickHandler


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


class MockHandler(SidekickHandler):
    """Mock handler for testing without network."""

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


class TestSidekickHandler:
    """Tests for SidekickHandler."""

    def test_root_endpoint(self) -> None:
        """Test root endpoint returns API info."""
        handler = MockHandler(MockRequest("GET", "/"))
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["name"] == "sidekick"
        assert "endpoints" in data

    def test_health_endpoint_no_backend(self) -> None:
        """Test health endpoint with no backend."""
        handler = MockHandler(MockRequest("GET", "/health"))
        handler.inference_backend = None
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["status"] == "degraded"
        assert data["backend"] == "none"

    def test_health_endpoint_with_backend(self) -> None:
        """Test health endpoint with backend."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.is_available.return_value = (True, "Available")

        handler = MockHandler(MockRequest("GET", "/health"))
        handler.inference_backend = mock_backend
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["status"] == "healthy"
        assert data["backend"] == "test"

    def test_models_endpoint(self) -> None:
        """Test OpenAI-compatible models endpoint."""
        handler = MockHandler(MockRequest("GET", "/v1/models"))
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "sidekick"

    def test_ask_get_endpoint(self) -> None:
        """Test ask endpoint with GET."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.generate.return_value = InferenceResult(
            success=True,
            response="Test response",
            latency_ms=100.0,
            tokens_generated=5,
        )

        handler = MockHandler(MockRequest("GET", "/ask?q=Hello"))
        handler.inference_backend = mock_backend
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["success"] is True
        assert data["response"] == "Test response"
        assert data["query"] == "Hello"

    def test_ask_get_missing_query(self) -> None:
        """Test ask endpoint with missing query."""
        handler = MockHandler(MockRequest("GET", "/ask"))
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 400
        assert "error" in data

    def test_ask_post_endpoint(self) -> None:
        """Test ask endpoint with POST."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.generate.return_value = InferenceResult(
            success=True,
            response="Post response",
            latency_ms=150.0,
            tokens_generated=7,
        )

        body = json.dumps({"query": "What is my name?"}).encode()
        handler = MockHandler(MockRequest("POST", "/ask", body, {"Content-Length": str(len(body))}))
        handler.inference_backend = mock_backend
        handler.do_POST()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["success"] is True
        assert data["response"] == "Post response"

    def test_chat_completions_endpoint(self) -> None:
        """Test OpenAI-compatible chat completions endpoint."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.generate.return_value = InferenceResult(
            success=True,
            response="Chat response",
            latency_ms=200.0,
            tokens_generated=10,
        )

        body = json.dumps({
            "model": "sidekick",
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
        }).encode()
        handler = MockHandler(MockRequest("POST", "/v1/chat/completions", body, {"Content-Length": str(len(body))}))
        handler.inference_backend = mock_backend
        handler.do_POST()

        code, data = handler.get_json_response()

        assert code == 200
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Chat response"
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completions_missing_messages(self) -> None:
        """Test chat completions with missing messages."""
        body = json.dumps({"model": "sidekick"}).encode()
        handler = MockHandler(MockRequest("POST", "/v1/chat/completions", body, {"Content-Length": str(len(body))}))
        handler.do_POST()

        code, data = handler.get_json_response()

        assert code == 400
        assert "error" in data

    def test_not_found(self) -> None:
        """Test 404 for unknown endpoint."""
        handler = MockHandler(MockRequest("GET", "/unknown"))
        handler.do_GET()

        code, data = handler.get_json_response()

        assert code == 404
        assert "error" in data

    def test_cors_options(self) -> None:
        """Test CORS preflight handling."""
        handler = MockHandler(MockRequest("OPTIONS", "/ask"))
        handler.do_OPTIONS()

        assert handler._response_code == 200
        assert handler._response_headers.get("Access-Control-Allow-Origin") == "*"
