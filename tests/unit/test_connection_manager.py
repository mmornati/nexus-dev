"""Tests for MCP connection manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ClientSession

from nexus_dev.gateway.connection_manager import ConnectionManager, MCPConnection
from nexus_dev.mcp_config import MCPServerConfig


@pytest.fixture
def mock_config():
    """Mock MCP server config."""
    return MCPServerConfig(
        command="test-server",
        args=["--test"],
        env={"TEST": "1"},
    )


@patch("nexus_dev.gateway.connection_manager.stdio_client")
@patch("nexus_dev.gateway.connection_manager.ClientSession")
@pytest.mark.asyncio
async def test_mcp_connection_lifecycle(mock_session_cls, mock_stdio, mock_config):
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


@patch("nexus_dev.gateway.connection_manager.MCPConnection")
@pytest.mark.asyncio
async def test_connection_manager_reuse(mock_conn_cls, mock_config):
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
async def test_connection_manager_disconnect_all(mock_config):
    """Test disconnecting all servers."""
    manager = ConnectionManager()

    # We'll use real MCPConnection mocked partially to verify disconnect logic
    # But it's easier to mock internal _connections dict if we want to verify calls
    # Let's just mock the connection objects stored in the manager.

    mock_conn1 = AsyncMock()
    mock_conn2 = AsyncMock()

    # Manually inject mocks into manager to test disconnect logic
    # Note: accessing private member for testing purposes
    manager._connections["c1"] = mock_conn1
    manager._connections["c2"] = mock_conn2

    await manager.disconnect_all()

    assert mock_conn1.disconnect.called
    assert mock_conn2.disconnect.called
    assert len(manager._connections) == 0


@patch("nexus_dev.gateway.connection_manager.MCPConnection")
@pytest.mark.asyncio
async def test_connection_manager_invoke_tool(mock_conn_cls, mock_config):
    """Test invoke_tool uses connection and calls tool."""
    manager = ConnectionManager()

    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value="result")

    mock_conn_instance = AsyncMock()
    mock_conn_instance.connect.return_value = mock_session
    mock_conn_cls.return_value = mock_conn_instance

    result = await manager.invoke_tool("test", mock_config, "my_tool", {"arg": "value"})

    assert result == "result"
    mock_session.call_tool.assert_called_once_with("my_tool", {"arg": "value"})
