"""MCP Connection Manager for Gateway Mode."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..mcp_config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Failed to connect to MCP server."""

    pass


class MCPTimeoutError(Exception):
    """Tool invocation timed out."""

    pass


@dataclass
class MCPConnection:
    """Active connection to an MCP server."""

    name: str
    config: MCPServerConfig
    session: ClientSession | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _cleanup_stack: list[Any] = field(default_factory=list)

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds (base delay for exponential backoff)

    @property
    def timeout(self) -> float:
        """Get tool execution timeout from config."""
        return self.config.timeout

    @property
    def connect_timeout(self) -> float:
        """Get connection timeout from config."""
        return self.config.connect_timeout

    async def connect(self) -> ClientSession:
        """Get or create connection with retry logic."""
        async with self._lock:
            # Check if existing session is still alive
            if self.session is not None:
                try:
                    await self.session.send_ping()
                    return self.session
                except Exception:
                    logger.warning("Connection to %s lost, reconnecting...", self.name)
                    await self._cleanup()

            # Try to connect with retries
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    return await self._do_connect()
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Connection attempt %d/%d to %s failed: %s",
                        attempt + 1,
                        self.max_retries,
                        self.name,
                        e,
                    )
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2**attempt)  # Exponential backoff
                        await asyncio.sleep(delay)

            raise MCPConnectionError(
                f"Failed to connect to {self.name} after {self.max_retries} attempts"
            ) from last_error

    async def _do_connect(self) -> ClientSession:
        """Perform actual connection to MCP server."""
        try:
            return await asyncio.wait_for(
                self._do_connect_impl(),
                timeout=self.connect_timeout,
            )
        except TimeoutError as e:
            raise MCPConnectionError(
                f"Connection to {self.name} timed out after {self.connect_timeout}s"
            ) from e

    async def _do_connect_impl(self) -> ClientSession:
        """Internal connection implementation."""
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        transport_cm = stdio_client(server_params)
        read, write = await transport_cm.__aenter__()
        self._cleanup_stack.append(transport_cm)

        session_cm = ClientSession(read, write)
        self.session = await session_cm.__aenter__()
        self._cleanup_stack.append(session_cm)

        await self.session.initialize()
        logger.info("Connected to MCP server: %s", self.name)
        return self.session

    async def _cleanup(self) -> None:
        """Clean up connection resources (called with lock held)."""
        while self._cleanup_stack:
            cm = self._cleanup_stack.pop()
            try:
                await cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Cleanup error for %s: %s", self.name, e)
        self.session = None

    async def disconnect(self) -> None:
        """Close connection."""
        async with self._lock:
            await self._cleanup()
            logger.info("Disconnected from MCP server: %s", self.name)

    async def invoke_with_timeout(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool with timeout protection.

        Args:
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            MCPTimeoutError: If tool invocation times out.
            MCPConnectionError: If connection fails.
        """
        session = await self.connect()
        try:
            return await asyncio.wait_for(
                session.call_tool(tool, arguments),
                timeout=self.timeout,
            )
        except TimeoutError as e:
            logger.error(
                "Tool %s on %s timed out after %.1fs",
                tool,
                self.name,
                self.timeout,
            )
            raise MCPTimeoutError(f"Tool '{tool}' timed out after {self.timeout}s") from e


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
        """Invoke a tool on a backend MCP server with timeout.

        Args:
            name: Server name for connection pooling.
            config: Server configuration.
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result (MCP CallToolResult).

        Raises:
            MCPConnectionError: If connection fails after retries.
            MCPTimeoutError: If tool invocation times out.
        """
        async with self._lock:
            if name not in self._connections:
                self._connections[name] = MCPConnection(name=name, config=config)
            connection = self._connections[name]

        return await connection.invoke_with_timeout(tool, arguments)
