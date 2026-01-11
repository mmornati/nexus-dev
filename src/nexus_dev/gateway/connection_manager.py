"""MCP Connection Manager for Gateway Mode."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import anyio
import httpx
from anyio.abc import TaskStatus
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.message import SessionMessage

from ..mcp_config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Failed to connect to MCP server."""

    pass


@asynccontextmanager
async def github_sse_client(
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = 5,
    sse_read_timeout: float = 60 * 5,
    auth: httpx.Auth | None = None,
) -> AsyncIterator[tuple[MemoryObjectReceiveStream[Any], MemoryObjectSendStream[Any]]]:
    """Custom SSE client for GitHub that uses POST with ping payload.

    This client avoids aconnect_sse because api.githubcopilot.com returns 400
    if 'Accept: text/event-stream' is present in the initial POST.
    """
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(100)

    # Prepare headers, ensuring Accept is NOT text/event-stream
    effective_headers = (headers or {}).copy()
    effective_headers["Accept"] = "*/*"
    if "User-Agent" not in effective_headers:
        effective_headers["User-Agent"] = "nexus-dev/1.0"

    async with anyio.create_task_group() as tg:
        try:
            async with (
                httpx.AsyncClient(
                    http2=True,
                    headers=effective_headers,
                    auth=auth,
                    timeout=httpx.Timeout(timeout, read=sse_read_timeout),
                ) as client,
                client.stream(
                    "POST",
                    url,
                    json={"jsonrpc": "2.0", "method": "ping", "id": 999},
                ) as response,
            ):
                response.raise_for_status()
                from httpx_sse import EventSource

                event_source = EventSource(response)

                async def sse_reader(
                    task_status: TaskStatus[str] = anyio.TASK_STATUS_IGNORED,
                ) -> None:
                    started = False
                    try:
                        async for sse in event_source.aiter_sse():
                            if not started and sse.event != "endpoint":
                                task_status.started(url)
                                started = True

                            match sse.event:
                                case "endpoint":
                                    endpoint_url = urljoin(url, sse.data)
                                    if not started:
                                        task_status.started(endpoint_url)
                                        started = True
                                case "message":
                                    if not sse.data:
                                        continue
                                    msg = types.JSONRPCMessage.model_validate_json(sse.data)
                                    await read_stream_writer.send(SessionMessage(msg))
                    except anyio.get_cancelled_exc_class():
                        pass
                    except Exception as exc:
                        if not started:
                            task_status.started(url)
                            started = True
                        await read_stream_writer.send(exc)
                    finally:
                        if not started:
                            task_status.started(url)

                async def post_writer(endpoint_url: str) -> None:
                    try:
                        # Use a separate client for POSTs to avoid potential multiplexing issues
                        # with the long-lived SSE stream.
                        async with (
                            httpx.AsyncClient(
                                http2=True,
                                headers=effective_headers,
                                auth=auth,
                                timeout=httpx.Timeout(timeout),
                            ) as post_client,
                            write_stream_reader,
                        ):
                            async for session_message in write_stream_reader:
                                resp = await post_client.post(
                                    endpoint_url,
                                    json=session_message.message.model_dump(
                                        by_alias=True, mode="json", exclude_none=True
                                    ),
                                )
                                resp.raise_for_status()

                                # GitHub hybrid behavior: response may be in the POST body
                                # formatted as SSE: data: {...}
                                body = resp.text
                                if "data: " in body:
                                    for line in body.splitlines():
                                        if line.startswith("data: "):
                                            data = line[len("data: ") :].strip()
                                            if data:
                                                msg = types.JSONRPCMessage.model_validate_json(data)
                                                await read_stream_writer.send(SessionMessage(msg))
                    except anyio.get_cancelled_exc_class():
                        pass
                    except Exception as exc:
                        logger.error("Error in post_writer for %s: %s", endpoint_url, exc)
                    finally:
                        await write_stream.aclose()

                with anyio.move_on_after(5.0):
                    endpoint_url = await tg.start(sse_reader)

                if endpoint_url is None:
                    endpoint_url = url

                tg.start_soon(post_writer, endpoint_url)
                yield read_stream, write_stream
        finally:
            # Cancel all background tasks in the group before exiting
            tg.cancel_scope.cancel()
            await read_stream_writer.aclose()
            await write_stream.aclose()


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
        if self.config.transport in ("sse", "http"):
            if not self.config.url:
                raise ValueError(
                    f"URL required for {self.config.transport.upper()} transport: {self.name}"
                )

            client_factory = github_sse_client if self.config.transport == "http" else sse_client
            transport_cm = client_factory(
                url=self.config.url,
                headers=self.config.headers,
            )
        else:
            server_params = StdioServerParameters(
                command=self.config.command,  # type: ignore
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
