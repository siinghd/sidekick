"""MCP (Model Context Protocol) server for Sidekick.

This allows Sidekick to be used as a tool by Claude and other
MCP-compatible clients.

MCP specification: https://modelcontextprotocol.io/
"""

import sys
from typing import Any

from sidekick.inference import InferenceConfig, get_inference_backend
from sidekick.utils.fast_json import dumps, loads


class MCPServer:
    """Simple MCP server implementation.

    Communicates via stdin/stdout using JSON-RPC 2.0.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        """Initialize the MCP server.

        Args:
            config: Inference configuration
        """
        self.config = config or InferenceConfig()
        self._backend = None

    def _get_backend(self):
        """Lazily initialize the inference backend."""
        if self._backend is None:
            try:
                self._backend = get_inference_backend(config=self.config)
            except ValueError:
                self._backend = None
        return self._backend

    def _send_response(self, id: Any, result: Any = None, error: dict | None = None) -> None:
        """Send a JSON-RPC response."""
        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": id,
        }

        if error:
            response["error"] = error
        else:
            response["result"] = result

        sys.stdout.write(dumps(response) + "\n")
        sys.stdout.flush()

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification."""
        notification: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        sys.stdout.write(dumps(notification) + "\n")
        sys.stdout.flush()

    def handle_request(self, request: dict) -> None:
        """Handle an incoming JSON-RPC request.

        Args:
            request: The parsed JSON-RPC request
        """
        method = request.get("method", "")
        params = request.get("params", {})
        id = request.get("id")

        if method == "initialize":
            self._handle_initialize(id, params)
        elif method == "initialized":
            # Notification, no response needed
            pass
        elif method == "tools/list":
            self._handle_tools_list(id)
        elif method == "tools/call":
            self._handle_tools_call(id, params)
        elif method == "resources/list":
            self._handle_resources_list(id)
        elif method == "prompts/list":
            self._handle_prompts_list(id)
        elif method == "shutdown":
            self._send_response(id, {})
        else:
            self._send_response(id, error={
                "code": -32601,
                "message": f"Method not found: {method}",
            })

    def _handle_initialize(self, id: Any, params: dict) -> None:
        """Handle initialize request."""
        self._send_response(id, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "sidekick",
                "version": "0.1.0",
            },
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
            },
        })

    def _handle_tools_list(self, id: Any) -> None:
        """Handle tools/list request."""
        self._send_response(id, {
            "tools": [
                {
                    "name": "ask_sidekick",
                    "description": "Ask a question about the user's personal information, preferences, projects, or history. Use this when you need to know something specific about the user.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask about the user",
                            },
                        },
                        "required": ["question"],
                    },
                },
            ],
        })

    def _handle_tools_call(self, id: Any, params: dict) -> None:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "ask_sidekick":
            question = arguments.get("question", "")
            if not question:
                self._send_response(id, error={
                    "code": -32602,
                    "message": "Missing 'question' argument",
                })
                return

            backend = self._get_backend()
            if not backend:
                self._send_response(id, {
                    "content": [
                        {
                            "type": "text",
                            "text": "Sidekick is not trained yet. Please run 'sidekick train' first.",
                        }
                    ],
                    "isError": True,
                })
                return

            result = backend.generate(question)

            if result.success:
                self._send_response(id, {
                    "content": [
                        {
                            "type": "text",
                            "text": result.response,
                        }
                    ],
                })
            else:
                self._send_response(id, {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {result.error}",
                        }
                    ],
                    "isError": True,
                })
        else:
            self._send_response(id, error={
                "code": -32601,
                "message": f"Unknown tool: {tool_name}",
            })

    def _handle_resources_list(self, id: Any) -> None:
        """Handle resources/list request."""
        self._send_response(id, {"resources": []})

    def _handle_prompts_list(self, id: Any) -> None:
        """Handle prompts/list request."""
        self._send_response(id, {"prompts": []})

    def run(self) -> None:
        """Run the MCP server, reading from stdin."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = loads(line)
                self.handle_request(request)
            except (ValueError, TypeError):
                self._send_response(None, error={
                    "code": -32700,
                    "message": "Parse error",
                })


def run_mcp_server(config: InferenceConfig | None = None) -> None:
    """Run the Sidekick MCP server.

    Args:
        config: Inference configuration
    """
    server = MCPServer(config)
    server.run()
