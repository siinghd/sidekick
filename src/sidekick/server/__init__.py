"""Server module for Sidekick.

Provides HTTP, MCP, and proxy servers for the Sidekick API.
"""

from sidekick.server.http import create_server, run_server
from sidekick.server.mcp import MCPServer, run_mcp_server
from sidekick.server.proxy import create_proxy_server, run_proxy_server

__all__ = [
    "create_server",
    "run_server",
    "MCPServer",
    "run_mcp_server",
    "create_proxy_server",
    "run_proxy_server",
]
