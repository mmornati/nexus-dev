from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.mcp_client import MCPClientManager, MCPServerConnection


@pytest.mark.asyncio
async def test_get_tools_http_success():
    """Test get_tools with HTTP transport success."""
    server = MCPServerConnection(
        name="test-http",
        command="",
        args=[],
        transport="http",
        url="http://test.com/mcp",
        headers={"Authorization": "Bearer token"},
        timeout=10.0,
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {
            "tools": [
                {
                    "name": "test-tool",
                    "description": "A test tool",
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        manager = MCPClientManager()
        tools = await manager.get_tools(server)

        assert len(tools) == 1
        assert tools[0].name == "test-tool"
        assert tools[0].description == "A test tool"

        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        assert args[0] == "http://test.com/mcp"
        assert kwargs["headers"] == {"Authorization": "Bearer token"}
        assert kwargs["json"] == {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        assert kwargs["timeout"] == 10.0


@pytest.mark.asyncio
async def test_get_tools_http_error():
    """Test get_tools with HTTP transport error."""
    server = MCPServerConnection(
        name="test-http-error", command="", args=[], transport="http", url="http://test.com/mcp"
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"error": {"code": -32600, "message": "Invalid Request"}}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        manager = MCPClientManager()
        with pytest.raises(RuntimeError, match="JSON-RPC error"):
            await manager.get_tools(server)


@pytest.mark.asyncio
async def test_get_tools_http_missing_url():
    """Test get_tools with HTTP transport missing URL."""
    server = MCPServerConnection(
        name="test-http-no-url", command="", args=[], transport="http", url=None
    )

    manager = MCPClientManager()
    with pytest.raises(ValueError, match="URL required for HTTP transport"):
        await manager.get_tools(server)


@pytest.mark.asyncio
async def test_get_tools_http_sse_wrapped_json():
    """Test get_tools with HTTP transport and SSE-wrapped JSON response (GitHub style)."""
    server = MCPServerConnection(
        name="test-http-sse",
        command="",
        args=[],
        transport="http",
        url="http://test.com/mcp",
        timeout=10.0,
    )

    # Mock response that fails json() but has text with SSE format
    mock_response = MagicMock()
    mock_response.json.side_effect = Exception("JSON Decode Error")
    mock_response.text = (
        "event: message\n"
        'data: {"result": {"tools": [{"name": "sse-tool", "description": "SSE Tool"}]}}\n\n'
    )
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        manager = MCPClientManager()
        tools = await manager.get_tools(server)

        assert len(tools) == 1
        assert tools[0].name == "sse-tool"
        assert tools[0].description == "SSE Tool"
