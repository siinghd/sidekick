"""Tests for MCP server module."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

from sidekick.inference import InferenceResult
from sidekick.server.mcp import MCPServer


class TestMCPServer:
    """Tests for MCPServer."""

    def test_initialization(self) -> None:
        """Test MCP server initialization."""
        server = MCPServer()

        assert server._backend is None
        assert server.config is not None

    def test_handle_initialize(self) -> None:
        """Test initialize request handling."""
        server = MCPServer()
        responses = []

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

            server.handle_request({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 1
        assert error is None
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "sidekick"
        assert "tools" in result["capabilities"]

    def test_handle_tools_list(self) -> None:
        """Test tools/list request handling."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 2
        assert error is None
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "ask_sidekick"

    def test_handle_tools_call_success(self) -> None:
        """Test successful tools/call request handling."""
        server = MCPServer()
        responses = []

        mock_backend = MagicMock()
        mock_backend.generate.return_value = InferenceResult(
            success=True,
            response="The answer is 42.",
        )
        server._backend = mock_backend

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ask_sidekick",
                "arguments": {"question": "What is the answer?"},
            },
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 3
        assert error is None
        assert result["content"][0]["text"] == "The answer is 42."
        assert result.get("isError") is None

    def test_handle_tools_call_no_backend(self) -> None:
        """Test tools/call when no backend available."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        with patch.object(server, "_get_backend", return_value=None):
            server.handle_request({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "ask_sidekick",
                    "arguments": {"question": "Hello"},
                },
            })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 4
        assert result["isError"] is True
        assert "train" in result["content"][0]["text"].lower()

    def test_handle_tools_call_missing_question(self) -> None:
        """Test tools/call with missing question argument."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "ask_sidekick",
                "arguments": {},
            },
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 5
        assert error is not None
        assert error["code"] == -32602

    def test_handle_tools_call_unknown_tool(self) -> None:
        """Test tools/call with unknown tool."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {},
            },
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 6
        assert error is not None
        assert "unknown" in error["message"].lower()

    def test_handle_resources_list(self) -> None:
        """Test resources/list request handling."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "resources/list",
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 7
        assert error is None
        assert result["resources"] == []

    def test_handle_prompts_list(self) -> None:
        """Test prompts/list request handling."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "prompts/list",
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 8
        assert error is None
        assert result["prompts"] == []

    def test_handle_unknown_method(self) -> None:
        """Test handling of unknown method."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "unknown/method",
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 9
        assert error is not None
        assert error["code"] == -32601

    def test_handle_shutdown(self) -> None:
        """Test shutdown request handling."""
        server = MCPServer()
        responses = []

        server._send_response = lambda id, result=None, error=None: responses.append((id, result, error))

        server.handle_request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "shutdown",
        })

        assert len(responses) == 1
        id, result, error = responses[0]
        assert id == 10
        assert error is None
        assert result == {}
