"""MCP Connection Manager for Gateway Mode.

This module provides connection management for MCP gateway mode, handling
concurrent tool invocations with per-call connection isolation and graceful
shutdown support.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
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
    """Manages connections to an MCP server with concurrency control.

    Each tool invocation gets its own isolated connection to avoid
    conflicts between concurrent calls. A semaphore limits the number
    of concurrent connections per server.
    """

    name: str
    config: MCPServerConfig
    max_concurrent: int = 5

    # Internal state
    _semaphore: asyncio.Semaphore = field(default=None)  # type: ignore
    _active_count: int = field(default=0, repr=False)
    _shutdown_event: asyncio.Event = field(default=None)  # type: ignore

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds (base delay for exponential backoff)

    def __post_init__(self) -> None:
        """Initialize fields that can't be set in default_factory."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()

    @property
    def timeout(self) -> float:
        """Get tool execution timeout from config."""
        return self.config.timeout

    @property
    def connect_timeout(self) -> float:
        """Get connection timeout from config."""
        return self.config.connect_timeout

    @property
    def active_invocations(self) -> int:
        """Get count of currently active invocations."""
        return self._active_count

    async def list_tools(self) -> Any:
        """List available tools on this MCP server.

        Returns:
            MCP ListToolsResult with available tools.

        Raises:
            MCPConnectionError: If connection fails.
        """
        logger.debug("[%s] Listing tools", self.name)
        try:
            async with asyncio.timeout(self.connect_timeout):
                async with self._scoped_session() as session:
                    return await session.list_tools()
        except TimeoutError as e:
            raise MCPConnectionError(
                f"Connection to {self.name} timed out after {self.connect_timeout}s"
            ) from e
        except Exception as e:
            raise MCPConnectionError(f"Failed to connect to {self.name}: {e}") from e

    @asynccontextmanager
    async def _scoped_stdio_session(self) -> AsyncGenerator[ClientSession]:
        """Create a scoped stdio session for a single invocation.

        The session is automatically cleaned up when the context exits.
        """
        logger.debug("[%s] Creating scoped stdio session", self.name)
        server_params = StdioServerParameters(
            command=self.config.command,  # type: ignore
            args=self.config.args,
            env=self.config.env,
        )

        async with stdio_client(server_params) as (read, write):  # noqa: SIM117
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.debug("[%s] Stdio session initialized", self.name)
                yield session

    @asynccontextmanager
    async def _scoped_sse_session(self) -> AsyncGenerator[ClientSession]:
        """Create a scoped SSE session for a single invocation."""
        logger.debug("[%s] Creating scoped SSE session to %s", self.name, self.config.url)

        if not self.config.url:
            raise ValueError(f"URL required for SSE transport: {self.name}")

        async with sse_client(  # noqa: SIM117
            url=self.config.url,
            headers=self.config.headers,
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.debug("[%s] SSE session initialized", self.name)
                yield session

    @asynccontextmanager
    async def _scoped_http_session(self) -> AsyncGenerator[ClientSession]:
        """Create a scoped HTTP session for a single invocation."""
        logger.debug("[%s] Creating scoped HTTP session to %s", self.name, self.config.url)

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
            logger.debug("[%s] HTTP session initialized", self.name)
            yield session

    @asynccontextmanager
    async def _scoped_session(self) -> AsyncGenerator[ClientSession]:
        """Create a scoped session appropriate for the transport type."""
        if self.config.transport == "stdio":
            async with self._scoped_stdio_session() as session:
                yield session
        elif self.config.transport == "sse":
            async with self._scoped_sse_session() as session:
                yield session
        elif self.config.transport == "http":
            async with self._scoped_http_session() as session:
                yield session
        else:
            raise ValueError(f"Unsupported transport: {self.config.transport}")

    async def _invoke_with_retry(
        self, session: ClientSession, tool: str, arguments: dict[str, Any]
    ) -> Any:
        """Invoke a tool with timeout protection.

        Args:
            session: Active MCP session.
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            MCPTimeoutError: If tool invocation times out.
        """
        try:
            # Use asyncio.wait_for only for stdio/sse as HTTP has built-in timeout
            if self.config.transport == "http":
                return await session.call_tool(tool, arguments)
            else:
                return await asyncio.wait_for(
                    session.call_tool(tool, arguments),
                    timeout=self.timeout,
                )
        except TimeoutError:
            logger.error("[%s] Tool invocation timed out: %s", self.name, tool)
            raise MCPTimeoutError(f"Tool '{tool}' timed out after {self.timeout}s") from None

    async def invoke_with_timeout(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool with concurrency control and timeout protection.

        Each invocation gets its own isolated connection. A semaphore limits
        the number of concurrent invocations per server.

        Args:
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result (MCP CallToolResult).

        Raises:
            MCPConnectionError: If connection fails after retries.
            MCPTimeoutError: If tool invocation times out.
        """
        logger.info("[%s] Queuing invoke_tool: %s", self.name, tool)

        # Wait for semaphore (limits concurrent connections)
        async with self._semaphore:
            self._active_count += 1
            try:
                logger.info(
                    "[%s] Starting invoke_tool: %s (active: %d/%d)",
                    self.name,
                    tool,
                    self._active_count,
                    self.max_concurrent,
                )

                # Retry connection with exponential backoff
                last_error: Exception | None = None
                for attempt in range(self.max_retries):
                    try:
                        async with asyncio.timeout(self.connect_timeout):
                            async with self._scoped_session() as session:
                                result = await self._invoke_with_retry(session, tool, arguments)
                                logger.info("[%s] Tool invocation successful: %s", self.name, tool)
                                return result
                    except MCPTimeoutError:
                        # Don't retry on tool timeout - the connection was fine
                        raise
                    except TimeoutError:
                        last_error = MCPConnectionError(
                            f"Connection to {self.name} timed out after {self.connect_timeout}s"
                        )
                        logger.warning(
                            "[%s] Connection attempt %d/%d timed out",
                            self.name,
                            attempt + 1,
                            self.max_retries,
                        )
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

                logger.error("[%s] All connection attempts failed for tool: %s", self.name, tool)
                raise MCPConnectionError(
                    f"Failed to connect to {self.name} after {self.max_retries} attempts"
                ) from last_error

            finally:
                self._active_count -= 1
                logger.debug(
                    "[%s] Completed invoke_tool: %s (remaining active: %d)",
                    self.name,
                    tool,
                    self._active_count,
                )

    async def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """Wait for all active invocations to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if all invocations completed, False if timeout expired.
        """
        if self._active_count == 0:
            return True

        logger.info(
            "[%s] Waiting for %d active invocation(s) to complete (timeout: %.1fs)",
            self.name,
            self._active_count,
            timeout,
        )

        try:
            async with asyncio.timeout(timeout):
                while self._active_count > 0:
                    await asyncio.sleep(0.1)
            logger.info("[%s] All invocations completed gracefully", self.name)
            return True
        except TimeoutError:
            logger.warning(
                "[%s] Timeout waiting for %d active invocation(s)",
                self.name,
                self._active_count,
            )
            return False

    async def disconnect(self) -> None:
        """Signal shutdown and cleanup.

        For per-call connections, this just logs the disconnect.
        Active invocations will complete naturally or be cancelled by their callers.
        """
        logger.info("[%s] Disconnect called (active: %d)", self.name, self._active_count)
        self._shutdown_event.set()


class ConnectionManager:
    """Manages pool of MCP connections with graceful shutdown support."""

    def __init__(self, default_max_concurrent: int = 5, shutdown_timeout: float = 5.0) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._lock = asyncio.Lock()
        self._default_max_concurrent = default_max_concurrent
        self._shutdown_timeout = shutdown_timeout

    def _get_max_concurrent(self, config: MCPServerConfig) -> int:
        """Get max concurrent setting for a server (per-server or default)."""
        if config.max_concurrent is not None:
            return config.max_concurrent
        return self._default_max_concurrent

    async def get_connection(self, name: str, config: MCPServerConfig) -> MCPConnection:
        """Get or create an MCPConnection for a named server.

        Note: This returns the MCPConnection object, not a ClientSession.
        For tool invocation, use invoke_tool() instead.
        """
        async with self._lock:
            if name not in self._connections:
                max_concurrent = self._get_max_concurrent(config)
                self._connections[name] = MCPConnection(
                    name=name,
                    config=config,
                    max_concurrent=max_concurrent,
                )
                logger.info(
                    "[%s] Created connection handler (max_concurrent: %d)",
                    name,
                    max_concurrent,
                )
            return self._connections[name]

    async def disconnect_all(self, graceful: bool = True) -> None:
        """Close all active connections.

        Args:
            graceful: If True, wait for active invocations to complete.
        """
        async with self._lock:
            if not self._connections:
                return

            logger.info("Disconnecting all MCP connections (graceful: %s)", graceful)

            if graceful:
                # Wait for all active invocations to complete
                wait_tasks = [
                    conn.wait_for_completion(self._shutdown_timeout)
                    for conn in self._connections.values()
                ]
                results = await asyncio.gather(*wait_tasks, return_exceptions=True)

                incomplete = sum(1 for r in results if r is False or isinstance(r, Exception))
                if incomplete > 0:
                    logger.warning(
                        "%d connection(s) had pending invocations at shutdown", incomplete
                    )

            # Signal disconnect to all connections
            disconnect_tasks = [conn.disconnect() for conn in self._connections.values()]
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

            self._connections.clear()
            logger.info("All MCP connections disconnected")

    async def invoke_tool(
        self, name: str, config: MCPServerConfig, tool: str, arguments: dict[str, Any]
    ) -> Any:
        """Invoke a tool on a backend MCP server with timeout and concurrency control.

        Each invocation gets its own isolated connection. This method is safe
        for concurrent calls to the same server.

        Args:
            name: Server name for connection management.
            config: Server configuration.
            tool: Tool name to invoke.
            arguments: Tool arguments.

        Returns:
            Tool execution result (MCP CallToolResult).

        Raises:
            MCPConnectionError: If connection fails after retries.
            MCPTimeoutError: If tool invocation times out.
        """
        connection = await self.get_connection(name, config)
        return await connection.invoke_with_timeout(tool, arguments)
