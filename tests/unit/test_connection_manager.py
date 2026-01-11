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


class TestMCPConnectionLifecycle:
    """Tests for MCPConnection connect/disconnect lifecycle."""

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_session_cls, mock_stdio, mock_config):
        """Test connection establishment and teardown."""
        # Mock context managers
        mock_transport_cm = AsyncMock()
        mock_stdio.return_value = mock_transport_cm
        mock_transport_cm.__aenter__.return_value = (MagicMock(), MagicMock())

        mock_session_cm = AsyncMock()
        mock_session_cls.return_value = mock_session_cm
        mock_session = AsyncMock(spec=ClientSession)
        mock_session_cm.__aenter__.return_value = mock_session

        conn = MCPConnection(name="test", config=mock_config)

        # Test connect
        session = await conn.connect()

        assert session == mock_session
        assert mock_transport_cm.__aenter__.called
        assert mock_session_cm.__aenter__.called
        assert mock_session.initialize.called

        # Test disconnect
        await conn.disconnect()

        assert mock_session_cm.__aexit__.called
        assert mock_transport_cm.__aexit__.called
        assert conn.session is None


class TestMCPConnectionPing:
    """Tests for connection health check via ping."""

    @pytest.mark.asyncio
    async def test_ping_healthy_session_reused(self, mock_config):
        """Test that healthy session is reused after ping."""
        conn = MCPConnection(name="test", config=mock_config)
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.send_ping = AsyncMock()
        conn.session = mock_session

        # Connect should return existing session after successful ping
        result = await conn.connect()

        assert result == mock_session
        mock_session.send_ping.assert_called_once()

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_ping_lost_connection_reconnects(self, mock_session_cls, mock_stdio, mock_config):
        """Test that lost connection triggers reconnection."""
        # Setup mock for reconnection
        mock_transport_cm = AsyncMock()
        mock_stdio.return_value = mock_transport_cm
        mock_transport_cm.__aenter__.return_value = (MagicMock(), MagicMock())

        mock_session_cm = AsyncMock()
        mock_session_cls.return_value = mock_session_cm
        new_session = AsyncMock(spec=ClientSession)
        mock_session_cm.__aenter__.return_value = new_session

        # Setup connection with a session that will fail ping
        conn = MCPConnection(name="test", config=mock_config)
        old_session = AsyncMock(spec=ClientSession)
        old_session.send_ping = AsyncMock(side_effect=Exception("Connection lost"))
        conn.session = old_session

        # Connect should detect lost connection and reconnect
        result = await conn.connect()

        assert result == new_session
        old_session.send_ping.assert_called_once()


class TestMCPConnectionRetry:
    """Tests for connection retry logic."""

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_retry_success_on_second_attempt(self, mock_session_cls, mock_stdio, mock_config):
        """Test successful connection after initial failure."""
        # First attempt fails, second succeeds
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.side_effect = [
            Exception("First attempt failed"),
            (MagicMock(), MagicMock()),
        ]
        mock_stdio.return_value = mock_transport_cm

        mock_session_cm = AsyncMock()
        mock_session_cls.return_value = mock_session_cm
        mock_session = AsyncMock(spec=ClientSession)
        mock_session_cm.__aenter__.return_value = mock_session

        conn = MCPConnection(name="test", config=mock_config, retry_delay=0.01)

        session = await conn.connect()

        assert session == mock_session
        assert mock_transport_cm.__aenter__.call_count == 2

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises_error(self, mock_stdio, mock_config):
        """Test MCPConnectionError after max retries."""
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.side_effect = Exception("Connection failed")
        mock_stdio.return_value = mock_transport_cm

        conn = MCPConnection(name="test", config=mock_config, max_retries=3, retry_delay=0.01)

        with pytest.raises(MCPConnectionError) as exc_info:
            await conn.connect()

        assert "Failed to connect to test after 3 attempts" in str(exc_info.value)
        assert mock_transport_cm.__aenter__.call_count == 3

    @patch("nexus_dev.gateway.connection_manager.asyncio.sleep")
    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, mock_stdio, mock_sleep, mock_config):
        """Test that retry delays use exponential backoff."""
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.side_effect = Exception("Connection failed")
        mock_stdio.return_value = mock_transport_cm
        mock_sleep.return_value = None

        conn = MCPConnection(name="test", config=mock_config, max_retries=4, retry_delay=1.0)

        with pytest.raises(MCPConnectionError):
            await conn.connect()

        # Check exponential backoff: 1.0, 2.0, 4.0 (3 sleeps for 4 attempts)
        assert mock_sleep.call_count == 3
        calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert calls == [1.0, 2.0, 4.0]


class TestMCPConnectionTimeout:
    """Tests for timeout functionality."""

    @pytest.mark.asyncio
    async def test_invoke_with_timeout_success(self, mock_config):
        """Test successful tool invocation within timeout."""
        mock_config.timeout = 5.0
        conn = MCPConnection(name="test", config=mock_config)
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.send_ping = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value="result")
        conn.session = mock_session

        result = await conn.invoke_with_timeout("my_tool", {"arg": "value"})

        assert result == "result"
        mock_session.call_tool.assert_called_once_with("my_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_invoke_with_timeout_raises_timeout_error(self, mock_config):
        """Test MCPTimeoutError when tool exceeds timeout."""
        mock_config.timeout = 0.1
        connection = MCPConnection(name="test", config=mock_config)

        # Create properly async mock for session
        mock_session = MagicMock(spec=ClientSession)
        mock_session.send_ping = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock(return_value=None)

        # The tool call itself needs to hang
        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.5)
            return "result"

        mock_session.call_tool = MagicMock(side_effect=slow_tool)

        # Mock connect to return our session
        connection.connect = AsyncMock(return_value=mock_session)
        connection.session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out"):
            await connection.invoke_with_timeout("tool", {})

    @pytest.mark.asyncio
    async def test_connect_uses_config_timeout(self, mock_config):
        """Test that connect uses the configured connect_timeout."""
        # Set a very short connect timeout
        mock_config.connect_timeout = 0.05
        connection = MCPConnection(name="test", config=mock_config)
        connection.retry_delay = 0.01  # Speed up retries

        async def slow_connect():
            await asyncio.sleep(0.2)
            return MagicMock()

        # Mock _do_connect_impl to be slow
        connection._do_connect_impl = AsyncMock(side_effect=slow_connect)

        # We anticipate connection failure due to timeout
        with pytest.raises(MCPConnectionError) as excinfo:
            await connection.connect()

        assert "after 3 attempts" in str(excinfo.value)


class TestMCPConnectionLogging:
    """Tests for logging output."""

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @pytest.mark.asyncio
    async def test_logs_warning_on_retry(self, mock_stdio, mock_config, caplog):
        """Test warning logs during retry attempts."""
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.side_effect = Exception("Connection failed")
        mock_stdio.return_value = mock_transport_cm

        conn = MCPConnection(
            name="test-server", config=mock_config, max_retries=2, retry_delay=0.01
        )

        with caplog.at_level(logging.WARNING), pytest.raises(MCPConnectionError):
            await conn.connect()

        assert "Connection attempt 1/2 to test-server failed" in caplog.text
        assert "Connection attempt 2/2 to test-server failed" in caplog.text

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @patch("nexus_dev.gateway.connection_manager.ClientSession")
    @pytest.mark.asyncio
    async def test_logs_info_on_connect(self, mock_session_cls, mock_stdio, mock_config, caplog):
        """Test info log on successful connection."""
        mock_transport_cm = AsyncMock()
        mock_stdio.return_value = mock_transport_cm
        mock_transport_cm.__aenter__.return_value = (MagicMock(), MagicMock())

        mock_session_cm = AsyncMock()
        mock_session_cls.return_value = mock_session_cm
        mock_session = AsyncMock(spec=ClientSession)
        mock_session_cm.__aenter__.return_value = mock_session

        conn = MCPConnection(name="test-server", config=mock_config)

        with caplog.at_level(logging.INFO):
            await conn.connect()

        assert "Connected to MCP server: test-server" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_warning_on_lost_connection(self, mock_config, caplog):
        """Test warning log when connection is lost."""
        conn = MCPConnection(name="test-server", config=mock_config, max_retries=1)
        mock_session = AsyncMock(spec=ClientSession)
        mock_session.send_ping = AsyncMock(side_effect=Exception("Connection lost"))
        conn.session = mock_session

        with caplog.at_level(logging.WARNING), pytest.raises(MCPConnectionError):
            await conn.connect()

        assert "Connection to test-server lost, reconnecting" in caplog.text


class TestConnectionManager:
    """Tests for ConnectionManager pooling."""

    @patch("nexus_dev.gateway.connection_manager.MCPConnection")
    @pytest.mark.asyncio
    async def test_connection_reuse(self, mock_conn_cls, mock_config):
        """Test connection reuse."""
        manager = ConnectionManager()
        mock_conn_instance = AsyncMock()
        mock_conn_instance.connect.return_value = AsyncMock(spec=ClientSession)
        mock_conn_cls.return_value = mock_conn_instance

        # First call - creates new connection
        s1 = await manager.get_connection("s1", mock_config)
        assert mock_conn_cls.call_count == 1

        # Second call - reuses connection
        s2 = await manager.get_connection("s1", mock_config)
        assert mock_conn_cls.call_count == 1
        assert s1 == s2

    @pytest.mark.asyncio
    async def test_disconnect_all(self, mock_config):
        """Test disconnecting all servers."""
        manager = ConnectionManager()

        mock_conn1 = AsyncMock()
        mock_conn2 = AsyncMock()

        manager._connections["c1"] = mock_conn1
        manager._connections["c2"] = mock_conn2

        await manager.disconnect_all()

        assert mock_conn1.disconnect.called
        assert mock_conn2.disconnect.called
        assert len(manager._connections) == 0

    @patch("nexus_dev.gateway.connection_manager.MCPConnection")
    @pytest.mark.asyncio
    async def test_invoke_tool(self, mock_conn_cls, mock_config):
        """Test invoke_tool uses connection and calls tool with timeout."""
        manager = ConnectionManager()

        mock_conn_instance = AsyncMock()
        mock_conn_instance.invoke_with_timeout = AsyncMock(return_value="result")
        mock_conn_cls.return_value = mock_conn_instance

        result = await manager.invoke_tool("test", mock_config, "my_tool", {"arg": "value"})

        assert result == "result"
        mock_conn_instance.invoke_with_timeout.assert_called_once_with("my_tool", {"arg": "value"})


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


class TestConcurrentConnections:
    """Tests for concurrent connection handling."""

    @patch("nexus_dev.gateway.connection_manager.MCPConnection")
    @pytest.mark.asyncio
    async def test_concurrent_get_connections(self, mock_conn_cls, mock_config):
        """Test multiple concurrent get_connection calls."""
        manager = ConnectionManager()
        mock_conn_instance = AsyncMock()
        mock_conn_instance.connect.return_value = AsyncMock(spec=ClientSession)
        mock_conn_cls.return_value = mock_conn_instance

        # Simulate concurrent requests for same server
        results = await asyncio.gather(
            manager.get_connection("s1", mock_config),
            manager.get_connection("s1", mock_config),
            manager.get_connection("s1", mock_config),
        )

        # Should only create one connection
        assert mock_conn_cls.call_count == 1
        assert all(r is not None for r in results)

    @patch("nexus_dev.gateway.connection_manager.MCPConnection")
    @pytest.mark.asyncio
    async def test_concurrent_different_servers(self, mock_conn_cls, mock_config):
        """Test concurrent connections to different servers."""
        manager = ConnectionManager()
        mock_conn_instance = AsyncMock()
        mock_conn_instance.connect.return_value = AsyncMock(spec=ClientSession)
        mock_conn_cls.return_value = mock_conn_instance

        # Concurrent requests for different servers
        await asyncio.gather(
            manager.get_connection("s1", mock_config),
            manager.get_connection("s2", mock_config),
            manager.get_connection("s3", mock_config),
        )

        # Should create 3 separate connections
        assert mock_conn_cls.call_count == 3

    @patch("nexus_dev.gateway.connection_manager.MCPConnection")
    @pytest.mark.asyncio
    async def test_concurrent_invoke_tool_same_server(self, mock_conn_cls, mock_config):
        """Test concurrent tool invocations on same server."""
        manager = ConnectionManager()
        mock_conn_instance = AsyncMock()
        mock_conn_instance.invoke_with_timeout.return_value = "result"
        mock_conn_cls.return_value = mock_conn_instance

        results = await asyncio.gather(
            manager.invoke_tool("s1", mock_config, "tool1", {}),
            manager.invoke_tool("s1", mock_config, "tool2", {}),
        )

        # Both should succeed
        assert results == ["result", "result"]
        # Only one connection created
        assert mock_conn_cls.call_count == 1


class TestConnectionEdgeCases:
    """Tests for edge cases in connection handling."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_session(self, mock_config):
        """Test that disconnect properly clears session."""
        conn = MCPConnection(name="test", config=mock_config)
        mock_session = AsyncMock(spec=ClientSession)
        conn.session = mock_session
        conn._cleanup_stack = []

        await conn.disconnect()

        assert conn.session is None

    @patch("nexus_dev.gateway.connection_manager.stdio_client")
    @pytest.mark.asyncio
    async def test_connect_while_cleanup_in_progress(self, mock_stdio, mock_config):
        """Test connect behavior when previous session was stale."""
        conn = MCPConnection(name="test", config=mock_config, max_retries=1, retry_delay=0.01)

        # Simulate stale session that fails ping
        old_session = AsyncMock(spec=ClientSession)
        old_session.send_ping = AsyncMock(side_effect=Exception("Stale"))
        conn.session = old_session

        # Mock transport context manager to fail (no real server)
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.side_effect = Exception("No server")
        mock_stdio.return_value = mock_transport_cm

        with pytest.raises(MCPConnectionError):
            await conn.connect()

        # Old session should be cleared
        assert conn.session is None
