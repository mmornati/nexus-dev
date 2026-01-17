"""MCP Connection Manager for Gateway Mode."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

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
            # For HTTP transport, always create fresh connections to avoid
            # anyio TaskGroup conflicts with streamable_http_client
            if self.config.transport == "http":
                if self.session is not None:
                    logger.debug("[%s] Cleaning up previous HTTP session", self.name)
                    await self._cleanup()
            elif self.session is not None:
                # Check if existing session is still alive (non-HTTP transports only)
                try:
                    logger.debug("[%s] Pinging existing session", self.name)
                    await self.session.send_ping()
                    logger.debug("[%s] Existing session is alive", self.name)
                    return self.session
                except Exception as e:
                    logger.warning(
                        "[%s] Connection lost or ping failed, reconnecting... Error: %s",
                        self.name,
                        e,
                    )
                    await self._cleanup()

            # Try to connect with retries within total connect_timeout
            last_error: Exception | None = None
            try:
                async with asyncio.timeout(self.connect_timeout):
                    for attempt in range(self.max_retries):
                        try:
                            logger.info(
                                "[%s] Connection attempt %d/%d",
                                self.name,
                                attempt + 1,
                                self.max_retries,
                            )
                            return await self._do_connect()
                        except Exception as e:
                            last_error = e
                            logger.warning(
                                "[%s] Connection attempt %d/%d failed: %s",
                                self.name,
                                attempt + 1,
                                self.max_retries,
                                e,
                            )
                            if attempt < self.max_retries - 1:
                                delay = self.retry_delay * (2**attempt)
                                logger.debug("[%s] Retrying in %.1fs...", self.name, delay)
                                await asyncio.sleep(delay)
            except asyncio.TimeoutError:
                logger.error("[%s] Connection timed out after %.1fs", self.name, self.connect_timeout)
                raise MCPConnectionError(
                    f"Failed to connect to {self.name} due to timeout after {self.connect_timeout}s"
                ) from last_error

            logger.error("[%s] All connection attempts failed", self.name)
            raise MCPConnectionError(
                f"Failed to connect to {self.name} after {self.max_retries} attempts"
            ) from last_error

    async def _do_connect(self) -> ClientSession:
        """Perform actual connection to MCP server.

        Note: We don't use asyncio.wait_for() here because anyio-based transports
        (like streamable_http_client) use their own cancel scopes which conflict
        with asyncio's cancellation. The httpx client has its own timeout configured.
        """
        logger.debug("[%s] Connecting...", self.name)
        try:
            result = await self._do_connect_impl()
            logger.info("[%s] Connection successful", self.name)
            return result
        except Exception as e:
            logger.error("[%s] Connection failed: %s", self.name, e)
            raise

    async def _do_connect_impl(self) -> ClientSession:
        """Internal connection implementation for SSE and stdio transports.

        Note: HTTP transport does NOT use this method - it uses _invoke_http instead
        to properly handle anyio's structured concurrency requirements.
        """
        if self.config.transport == "sse":
            if not self.config.url:
                raise ValueError(f"URL required for SSE transport: {self.name}")

            logger.debug("[%s] Using SSE transport to %s", self.name, self.config.url)
            transport_cm = sse_client(
                url=self.config.url,
                headers=self.config.headers,
            )
        elif self.config.transport == "stdio":
            logger.debug(
                "[%s] Using stdio transport with command: %s", self.name, self.config.command
            )
            server_params = StdioServerParameters(
                command=self.config.command,  # type: ignore
                args=self.config.args,
                env=self.config.env,
            )
            transport_cm = stdio_client(server_params)
        else:
            raise ValueError(f"Unsupported transport for pooling: {self.config.transport}")

        logger.debug("[%s] Entering transport context manager", self.name)
        read, write = await transport_cm.__aenter__()
        self._cleanup_stack.append(transport_cm)

        logger.debug("[%s] Creating client session", self.name)
        session_cm = ClientSession(read, write)
        self.session = await session_cm.__aenter__()
        self._cleanup_stack.append(session_cm)

        logger.debug("[%s] Initializing session", self.name)
        await self.session.initialize()
        logger.info("[%s] Connected to MCP server successfully", self.name)
        return self.session

        logger.debug("[%s] Creating client session", self.name)
        session_cm = ClientSession(read, write)
        self.session = await session_cm.__aenter__()
        self._cleanup_stack.append(session_cm)

        logger.debug("[%s] Initializing session", self.name)
        await self.session.initialize()
        logger.info("[%s] Connected to MCP server successfully", self.name)
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

    async def _invoke_http(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool using HTTP transport with proper context manager handling.

        For HTTP transport, we must use proper async with blocks because
        streamable_http_client uses anyio TaskGroups internally that conflict
        with manual __aenter__/__aexit__ calls.
        """
        logger.debug("[%s] Using HTTP transport for tool: %s", self.name, tool)

        if not self.config.url:
            raise ValueError(f"URL required for HTTP transport: {self.name}")

        async with (
            httpx.AsyncClient(
                headers=self.config.headers,
                timeout=httpx.Timeout(self.config.timeout),
            ) as http_client,
            streamable_http_client(
                url=self.config.url,
                http_client=http_client,
                terminate_on_close=True,
            ) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            logger.debug("[%s] HTTP session initialized, calling tool: %s", self.name, tool)
            result = await session.call_tool(tool, arguments)
            logger.debug("[%s] Tool call completed: %s", self.name, tool)
            return result

    async def _invoke_impl(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Internal implementation of tool invocation.

        Args:
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        # For HTTP transport, use dedicated method with proper async with handling
        if self.config.transport == "http":
            return await self._invoke_http(tool, arguments)

        # For other transports (stdio, sse), use connection pooling
        logger.debug("[%s] Getting connection for tool: %s", self.name, tool)
        session = await self.connect()
        logger.debug("[%s] Connection established, calling tool: %s", self.name, tool)
        result = await session.call_tool(tool, arguments)
        logger.debug("[%s] Tool call completed: %s", self.name, tool)
        return result

    async def invoke_with_timeout(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool with timeout protection.

        Note: We don't use asyncio.wait_for() here because anyio-based transports
        (like streamable_http_client) use their own cancel scopes which conflict
        with asyncio's cancellation. The httpx client has its own timeout configured.

        Args:
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            MCPTimeoutError: If tool invocation times out.
            MCPConnectionError: If connection fails.
        """
        logger.info("[%s] Starting invoke_tool: %s with args: %s", self.name, tool, arguments)

        try:
            if self.config.transport == "http":
                # HTTP has its own timeout in httpx
                result = await self._invoke_impl(tool, arguments)
            else:
                # Use asyncio.wait_for for stdio/sse as they don't have built-in timeout
                result = await asyncio.wait_for(
                    self._invoke_impl(tool, arguments), timeout=self.timeout
                )
            logger.info("[%s] Tool invocation successful: %s", self.name, tool)
            return result
        except asyncio.TimeoutError:
            logger.error("[%s] Tool invocation timed out: %s", self.name, tool)
            raise MCPTimeoutError(f"Tool '{tool}' timed out after {self.timeout}s")
        except Exception as e:
            logger.error("[%s] Tool invocation failed: %s - %s", self.name, tool, e)
            # Cleanup on any error (only for non-HTTP transports)
            if self.config.transport != "http":
                async with self._lock:
                    await self._cleanup()
            raise


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
