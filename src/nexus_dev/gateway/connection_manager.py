"""MCP Connection Manager for Gateway Mode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..mcp_config import MCPServerConfig


@dataclass
class MCPConnection:
    """Active connection to an MCP server."""

    name: str
    config: MCPServerConfig
    session: ClientSession | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _cleanup_stack: list[Any] = field(default_factory=list)

    async def connect(self) -> ClientSession:
        """Get or create connection."""
        async with self._lock:
            if self.session is None:
                try:
                    # 1. Start stdio client
                    server_params = StdioServerParameters(
                        command=self.config.command,
                        args=self.config.args,
                        env=self.config.env,
                    )
                    transport_cm = stdio_client(server_params)
                    read, write = await transport_cm.__aenter__()
                    self._cleanup_stack.append(transport_cm)

                    # 2. Start Request/Response ClientSession
                    session_cm = ClientSession(read, write)
                    self.session = await session_cm.__aenter__()
                    self._cleanup_stack.append(session_cm)

                    # 3. Initialize
                    await self.session.initialize()

                except Exception:
                    await self.disconnect()
                    raise

            return self.session

    async def disconnect(self) -> None:
        """Close connection."""
        async with self._lock:
            # Cleanup in reverse order
            while self._cleanup_stack:
                cm = self._cleanup_stack.pop()
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    # Log error but continue cleanup
                    pass
            self.session = None


class ConnectionManager:
    """Manages pool of MCP connections."""

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._lock = asyncio.Lock()

    async def get_connection(self, name: str, config: MCPServerConfig) -> ClientSession:
        """Get connection for a named server, creating if needed."""
        async with self._lock:
            if name not in self._connections:
                self._connections[name] = MCPConnection(name=name, config=config)

            connection = self._connections[name]

        # Connect outside the manager lock to avoid blocking other requests
        return await connection.connect()

    async def disconnect_all(self) -> None:
        """Close all active connections."""
        async with self._lock:
            coros = [conn.disconnect() for conn in self._connections.values()]
            if coros:
                await asyncio.gather(*coros, return_exceptions=True)
            self._connections.clear()

    async def invoke_tool(
        self, name: str, config: MCPServerConfig, tool: str, arguments: dict[str, Any]
    ) -> Any:
        """Invoke a tool on a backend MCP server.

        Args:
            name: Server name for connection pooling.
            config: Server configuration.
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result (MCP CallToolResult).
        """
        session = await self.get_connection(name, config)
        return await session.call_tool(tool, arguments)
