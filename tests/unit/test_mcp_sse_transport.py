"""Tests for MCP SSE transport with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.mcp_client import MCPClientManager, MCPServerConnection


@pytest.mark.asyncio
class TestMCPSSETransport:
    """Test suite for MCP SSE transport."""

    async def test_get_tools_sse_success(self):
        """Test get_tools with SSE transport success."""
        server = MCPServerConnection(
            name="test-sse",
            command="",
            args=[],
            transport="sse",
            url="http://test.com/sse",
            headers={"Authorization": "Bearer token"},
        )

        # Mock tool objects
        mock_tool = MagicMock()
        mock_tool.name = "sse-tool"
        mock_tool.description = "SSE Tool"
        mock_tool.inputSchema = {"type": "object"}

        # Mock list_tools result
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        # Mock session
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock sse_client
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        mock_sse_cm = MagicMock()
        mock_sse_cm.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_sse_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("nexus_dev.mcp_client.sse_client", return_value=mock_sse_cm) as mock_sse_func,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
        ):
            manager = MCPClientManager()
            tools = await manager.get_tools(server)

            assert len(tools) == 1
            assert tools[0].name == "sse-tool"
            assert tools[0].description == "SSE Tool"

            mock_sse_func.assert_called_once_with(
                url="http://test.com/sse", headers={"Authorization": "Bearer token"}
            )
            mock_session.initialize.assert_awaited_once()
            mock_session.list_tools.assert_awaited_once()

    async def test_get_tools_sse_invalid_url(self):
        """Test get_tools with SSE transport missing URL."""
        server = MCPServerConnection(
            name="test-sse-no-url", command="", args=[], transport="sse", url=None
        )

        manager = MCPClientManager()
        with pytest.raises(ValueError, match="URL required for SSE transport"):
            await manager.get_tools(server)

    async def test_call_tool_sse_success(self):
        """Test call_tool with SSE transport success."""
        server = MCPServerConnection(
            name="test-sse",
            command="",
            args=[],
            transport="sse",
            url="http://test.com/sse",
        )

        # Mock call_tool result
        mock_result = {"status": "success", "data": "value"}

        # Mock session
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock sse_client
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        mock_sse_cm = MagicMock()
        mock_sse_cm.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_sse_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("nexus_dev.mcp_client.sse_client", return_value=mock_sse_cm) as mock_sse_func,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
        ):
            manager = MCPClientManager()
            result = await manager.call_tool(server, "test-tool", {"arg": "value"})

            assert result == mock_result

            mock_sse_func.assert_called_once_with(url="http://test.com/sse", headers={})
            mock_session.initialize.assert_awaited_once()
            mock_session.call_tool.assert_awaited_once_with("test-tool", {"arg": "value"})

    async def test_call_tool_sse_invalid_url(self):
        """Test call_tool with SSE transport missing URL."""
        server = MCPServerConnection(
            name="test-sse-no-url", command="", args=[], transport="sse", url=None
        )

        manager = MCPClientManager()
        with pytest.raises(ValueError, match="URL required for SSE transport"):
            await manager.call_tool(server, "test-tool", {})

    async def test_call_tool_sse_connection_error(self):
        """Test call_tool with SSE connection error."""
        server = MCPServerConnection(
            name="test-sse-error", command="", args=[], transport="sse", url="http://test.com/sse"
        )

        # Mock sse_client raising exception
        mock_sse_cm = MagicMock()
        mock_sse_cm.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
        mock_sse_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("nexus_dev.mcp_client.sse_client", return_value=mock_sse_cm):
            manager = MCPClientManager()
            with pytest.raises(Exception, match="Connection failed"):
                await manager.call_tool(server, "test-tool", {})
