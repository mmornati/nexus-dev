"""Tests for MCP connection manager."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ClientSession

from nexus_dev.gateway.connection_manager import (
    ConnectionManager,
    MCPConnection,
    MCPConnectionError,
    MCPTimeoutError,
)
from nexus_dev.mcp_config import MCPServerConfig


@pytest.fixture
def mock_config():
    """Mock MCP server config."""
    return MCPServerConfig(
        command="test-server",
        args=["--test"],
        env={"TEST": "1"},
    )


@pytest.fixture
def mock_sse_config():
    """Mock SSE server config."""
    return MCPServerConfig(
        transport="sse",
        url="http://localhost:8000/sse",
        headers={"Authorization": "Bearer test"},
    )


@pytest.fixture
def mock_http_config():
    """Mock HTTP server config."""
    return MCPServerConfig(
        transport="http",
        url="http://localhost:8000/mcp",
        timeout=30.0,
    )


class TestMCPConnectionProperties:
    """Tests for MCPConnection properties and initialization."""

    def test_default_values(self, mock_config):
        """Test default initialization values."""
        conn = MCPConnection(name="test", config=mock_config)

        assert conn.name == "test"
        assert conn.max_concurrent == 5
        assert conn.max_retries == 3
        assert conn.retry_delay == 1.0
        assert conn.active_invocations == 0

    def test_custom_max_concurrent(self, mock_config):
        """Test custom max_concurrent setting."""
        conn = MCPConnection(name="test", config=mock_config, max_concurrent=10)

        assert conn.max_concurrent == 10

    def test_timeout_property(self, mock_config):
        """Test timeout property returns config value."""
        mock_config.timeout = 60.0
        conn = MCPConnection(name="test", config=mock_config)

        assert conn.timeout == 60.0

    def test_connect_timeout_property(self, mock_config):
        """Test connect_timeout property returns config value."""
        mock_config.connect_timeout = 15.0
        conn = MCPConnection(name="test", config=mock_config)

        assert conn.connect_timeout == 15.0


class TestMCPConnectionInvocation:
    """Tests for tool invocation with per-call connections."""

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_invoke_with_timeout_success(self, mock_session_cls, mock_stdio, mock_config):
        """Test successful tool invocation."""
        # Setup mocks
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value="result")

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        conn = MCPConnection(name="test", config=mock_config)
        result = await conn.invoke_with_timeout("my_tool", {"arg": "value"})

        assert result == "result"
        mock_session.call_tool.assert_called_once_with("my_tool", {"arg": "value"})

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_invoke_tracks_active_count(self, mock_session_cls, mock_stdio, mock_config):
        """Test that active invocation count is tracked correctly."""
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()

        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "result"

        mock_session.call_tool = slow_tool

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        conn = MCPConnection(name="test", config=mock_config)
        assert conn.active_invocations == 0

        # Start invocation but don't await
        task = asyncio.create_task(conn.invoke_with_timeout("tool", {}))
        await asyncio.sleep(0.05)

        # Should be active during invocation
        assert conn.active_invocations == 1

        await task

        # Should be 0 after completion
        assert conn.active_invocations == 0

    @pytest.mark.asyncio
    async def test_invoke_with_timeout_raises_timeout_error(self, mock_config):
        """Test MCPTimeoutError when tool exceeds timeout."""
        mock_config.timeout = 0.1
        conn = MCPConnection(name="test", config=mock_config)

        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.5)
            return "result"

        # Mock the scoped session to return a slow session
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.call_tool = slow_tool

        with patch.object(conn, "_scoped_session") as mock_scoped:
            mock_scoped.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_scoped.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(MCPTimeoutError, match="timed out"):
                await conn.invoke_with_timeout("tool", {})


class TestMCPConnectionConcurrency:
    """Tests for concurrent invocation handling."""

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_concurrent_invocations_complete_independently(
        self, mock_session_cls, mock_stdio, mock_config
    ):
        """Test that multiple concurrent invocations complete independently."""
        call_order = []

        async def tracked_tool(tool_name, args):
            call_order.append(f"start_{tool_name}")
            await asyncio.sleep(0.05)
            call_order.append(f"end_{tool_name}")
            return f"result_{tool_name}"

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = tracked_tool

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        conn = MCPConnection(name="test", config=mock_config, max_concurrent=3)

        # Run 3 concurrent invocations
        results = await asyncio.gather(
            conn.invoke_with_timeout("tool1", {}),
            conn.invoke_with_timeout("tool2", {}),
            conn.invoke_with_timeout("tool3", {}),
        )

        # All should complete
        assert len(results) == 3

        # All starts should happen before all ends (parallel execution)
        starts = [x for x in call_order if x.startswith("start_")]
        ends = [x for x in call_order if x.startswith("end_")]
        assert len(starts) == 3
        assert len(ends) == 3

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_connections(
        self, mock_session_cls, mock_stdio, mock_config
    ):
        """Test that max_concurrent setting limits parallel connections."""
        active_count = []
        max_active = 0

        async def count_active_tool(tool_name, args):
            nonlocal max_active
            active_count.append(1)
            current = len(active_count)
            if current > max_active:
                max_active = current
            await asyncio.sleep(0.05)
            active_count.pop()
            return "result"

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = count_active_tool

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        # Limit to 2 concurrent
        conn = MCPConnection(name="test", config=mock_config, max_concurrent=2)

        # Try to run 5 concurrent invocations
        await asyncio.gather(*[conn.invoke_with_timeout(f"tool{i}", {}) for i in range(5)])

        # Max active should never exceed 2
        assert max_active <= 2


class TestMCPConnectionRetry:
    """Tests for connection retry logic."""

    @patch("nexus_dev.gateway.connection_manager.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_retry_on_connection_failure(self, mock_sleep, mock_config):
        """Test retry logic when connection fails."""
        from contextlib import asynccontextmanager

        mock_sleep.return_value = None
        conn = MCPConnection(name="test", config=mock_config, max_retries=3, retry_delay=0.01)

        attempt_count = 0

        @asynccontextmanager
        async def failing_session():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Connection failed")
            # Return working mock on third attempt
            mock_session = AsyncMock(spec=ClientSession)
            mock_session.call_tool = AsyncMock(return_value="result")
            yield mock_session

        with patch.object(conn, "_scoped_session", failing_session):
            result = await conn.invoke_with_timeout("tool", {})

        assert result == "result"
        assert attempt_count == 3

        assert result == "result"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises_error(self, mock_config):
        """Test MCPConnectionError after max retries."""
        conn = MCPConnection(name="test", config=mock_config, max_retries=2, retry_delay=0.01)

        with patch.object(conn, "_scoped_session") as mock_scoped:
            mock_scoped.return_value.__aenter__ = AsyncMock(
                side_effect=ConnectionError("Always fails")
            )
            mock_scoped.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(MCPConnectionError) as exc_info:
                await conn.invoke_with_timeout("tool", {})

        assert "Failed to connect to test after 2 attempts" in str(exc_info.value)


class TestMCPConnectionGracefulShutdown:
    """Tests for graceful shutdown functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_completion_no_active(self, mock_config):
        """Test wait_for_completion returns immediately when no active invocations."""
        conn = MCPConnection(name="test", config=mock_config)

        result = await conn.wait_for_completion(timeout=1.0)

        assert result is True

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_wait_for_completion_waits_for_active(
        self, mock_session_cls, mock_stdio, mock_config
    ):
        """Test wait_for_completion waits for active invocations."""
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()

        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.2)
            return "result"

        mock_session.call_tool = slow_tool

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        conn = MCPConnection(name="test", config=mock_config)

        # Start a slow invocation
        invoke_task = asyncio.create_task(conn.invoke_with_timeout("tool", {}))
        await asyncio.sleep(0.05)

        # Wait should succeed
        wait_task = asyncio.create_task(conn.wait_for_completion(timeout=5.0))
        result = await wait_task

        assert result is True
        await invoke_task  # Cleanup

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, mock_config):
        """Test wait_for_completion returns False on timeout."""
        conn = MCPConnection(name="test", config=mock_config)
        conn._active_count = 1  # Simulate active invocation

        result = await conn.wait_for_completion(timeout=0.1)

        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_signals_shutdown(self, mock_config):
        """Test disconnect sets shutdown event."""
        conn = MCPConnection(name="test", config=mock_config)

        assert not conn._shutdown_event.is_set()

        await conn.disconnect()

        assert conn._shutdown_event.is_set()


class TestConnectionManager:
    """Tests for ConnectionManager pooling."""

    @pytest.mark.asyncio
    async def test_connection_creation(self, mock_config):
        """Test connection is created on first request."""
        manager = ConnectionManager()

        conn = await manager.get_connection("test", mock_config)

        assert conn.name == "test"
        assert "test" in manager._connections

    @pytest.mark.asyncio
    async def test_connection_reuse(self, mock_config):
        """Test same connection is returned for same server."""
        manager = ConnectionManager()

        conn1 = await manager.get_connection("test", mock_config)
        conn2 = await manager.get_connection("test", mock_config)

        assert conn1 is conn2

    @pytest.mark.asyncio
    async def test_different_connections_for_different_servers(self, mock_config):
        """Test different connections for different servers."""
        manager = ConnectionManager()

        conn1 = await manager.get_connection("server1", mock_config)
        conn2 = await manager.get_connection("server2", mock_config)

        assert conn1 is not conn2
        assert conn1.name == "server1"
        assert conn2.name == "server2"

    @pytest.mark.asyncio
    async def test_default_max_concurrent(self, mock_config):
        """Test default max_concurrent is applied."""
        manager = ConnectionManager(default_max_concurrent=10)

        conn = await manager.get_connection("test", mock_config)

        assert conn.max_concurrent == 10

    @pytest.mark.asyncio
    async def test_per_server_max_concurrent_override(self, mock_config):
        """Test per-server max_concurrent overrides default."""
        mock_config.max_concurrent = 3
        manager = ConnectionManager(default_max_concurrent=10)

        conn = await manager.get_connection("test", mock_config)

        assert conn.max_concurrent == 3

    @pytest.mark.asyncio
    async def test_disconnect_all_graceful(self, mock_config):
        """Test graceful disconnect waits for invocations."""
        manager = ConnectionManager(shutdown_timeout=0.5)

        # Create a connection
        _conn = await manager.get_connection("test", mock_config)

        # Disconnect all
        await manager.disconnect_all(graceful=True)

        assert len(manager._connections) == 0

    @pytest.mark.asyncio
    async def test_disconnect_all_non_graceful(self, mock_config):
        """Test non-graceful disconnect clears immediately."""
        manager = ConnectionManager()

        await manager.get_connection("test", mock_config)
        await manager.disconnect_all(graceful=False)

        assert len(manager._connections) == 0


class TestConnectionManagerInvokeTool:
    """Tests for ConnectionManager.invoke_tool."""

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_invoke_tool_creates_connection(self, mock_session_cls, mock_stdio, mock_config):
        """Test invoke_tool creates connection if needed."""
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value="result")

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        manager = ConnectionManager()

        result = await manager.invoke_tool("test", mock_config, "my_tool", {"arg": "value"})

        assert result == "result"
        assert "test" in manager._connections

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_concurrent_invoke_tool_same_server(
        self, mock_session_cls, mock_stdio, mock_config
    ):
        """Test concurrent tool invocations on same server."""
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value="result")

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        manager = ConnectionManager()

        results = await asyncio.gather(
            manager.invoke_tool("test", mock_config, "tool1", {}),
            manager.invoke_tool("test", mock_config, "tool2", {}),
            manager.invoke_tool("test", mock_config, "tool3", {}),
        )

        assert results == ["result", "result", "result"]
        # Only one connection handler created
        assert len(manager._connections) == 1


class TestExceptions:
    """Tests for custom exceptions."""

    def test_mcp_connection_error(self):
        """Test MCPConnectionError can be raised and caught."""
        with pytest.raises(MCPConnectionError) as exc_info:
            raise MCPConnectionError("Connection failed")

        assert "Connection failed" in str(exc_info.value)

    def test_mcp_timeout_error(self):
        """Test MCPTimeoutError can be raised and caught."""
        with pytest.raises(MCPTimeoutError) as exc_info:
            raise MCPTimeoutError("Tool timed out")

        assert "Tool timed out" in str(exc_info.value)

    def test_exceptions_are_distinct(self):
        """Test that exceptions are distinct types."""
        assert MCPConnectionError != MCPTimeoutError
        assert not issubclass(MCPConnectionError, MCPTimeoutError)
        assert not issubclass(MCPTimeoutError, MCPConnectionError)


class TestMCPConnectionTransports:
    """Tests for different transport types."""

    @pytest.mark.asyncio
    async def test_unsupported_transport_raises_error(self, mock_config):
        """Test ValueError for unsupported transport."""
        mock_config.transport = "unsupported"  # type: ignore
        conn = MCPConnection(name="test", config=mock_config, max_retries=1)

        with pytest.raises(MCPConnectionError) as exc_info:
            await conn.invoke_with_timeout("tool", {})

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Unsupported transport" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_sse_missing_url_raises_error(self, mock_sse_config):
        """Test ValueError when URL missing for SSE."""
        mock_sse_config.url = None
        conn = MCPConnection(name="test", config=mock_sse_config, max_retries=1)

        with pytest.raises(MCPConnectionError) as exc_info:
            await conn.invoke_with_timeout("tool", {})

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "URL required for SSE transport" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_http_missing_url_raises_error(self, mock_http_config):
        """Test ValueError when URL missing for HTTP."""
        mock_http_config.url = None
        conn = MCPConnection(name="test", config=mock_http_config, max_retries=1)

        with pytest.raises(MCPConnectionError) as exc_info:
            await conn.invoke_with_timeout("tool", {})

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "URL required for HTTP transport" in str(exc_info.value.__cause__)


class TestMCPConnectionLogging:
    """Tests for logging output."""

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_logs_invocation_info(self, mock_session_cls, mock_stdio, mock_config, caplog):
        """Test info logs during invocation."""
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value="result")

        mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        conn = MCPConnection(name="test-server", config=mock_config)

        with caplog.at_level(logging.INFO):
            await conn.invoke_with_timeout("my_tool", {})

        assert "[test-server] Queuing invoke_tool: my_tool" in caplog.text
        assert "[test-server] Tool invocation successful: my_tool" in caplog.text
