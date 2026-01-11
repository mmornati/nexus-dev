"""Tests for MCP client with mocked dependencies."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.mcp_client import (
    MCPClientManager,
    MCPServerConnection,
    MCPToolSchema,
    expand_env_vars,
)


class TestExpandEnvVars:
    """Test suite for expand_env_vars function."""

    def test_expand_single_var(self):
        """Test expanding a single environment variable."""
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            result = expand_env_vars({"key": "${API_KEY}"})
            assert result == {"key": "secret123"}

    def test_expand_multiple_vars(self):
        """Test expanding multiple environment variables."""
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
            result = expand_env_vars({"url": "${HOST}:${PORT}", "host": "${HOST}"})
            assert result == {"url": "localhost:8080", "host": "localhost"}

    def test_expand_missing_var(self):
        """Test that missing environment variables are replaced with empty string."""
        with patch.dict(os.environ, {}, clear=True):
            result = expand_env_vars({"key": "${MISSING_VAR}"})
            assert result == {"key": ""}

    def test_expand_with_literal_text(self):
        """Test expanding variables mixed with literal text."""
        with patch.dict(os.environ, {"USER": "john"}):
            result = expand_env_vars({"greeting": "Hello ${USER}!"})
            assert result == {"greeting": "Hello john!"}

    def test_expand_no_vars(self):
        """Test that strings without variables are unchanged."""
        result = expand_env_vars({"key": "plain_value", "number": "123"})
        assert result == {"key": "plain_value", "number": "123"}

    def test_expand_empty_dict(self):
        """Test with empty dictionary."""
        result = expand_env_vars({})
        assert result == {}


class TestMCPServerConnection:
    """Test suite for MCPServerConnection dataclass."""

    def test_create_connection_minimal(self):
        """Test creating a server connection with minimal fields."""
        conn = MCPServerConnection(name="test", command="python", args=["-m", "test"])
        assert conn.name == "test"
        assert conn.command == "python"
        assert conn.args == ["-m", "test"]
        assert conn.env is None

    def test_create_connection_with_env(self):
        """Test creating a server connection with environment variables."""
        conn = MCPServerConnection(
            name="test", command="python", args=[], env={"API_KEY": "secret"}
        )
        assert conn.env == {"API_KEY": "secret"}


class TestMCPToolSchema:
    """Test suite for MCPToolSchema dataclass."""

    def test_create_tool_schema(self):
        """Test creating a tool schema."""
        schema = MCPToolSchema(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert "properties" in schema.input_schema


class TestMCPClientManager:
    """Test suite for MCPClientManager."""

    @pytest.mark.asyncio
    async def test_get_tools_success(self):
        """Test getting tools from an MCP server successfully."""
        # Create mock tool objects
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "First tool"
        mock_tool1.inputSchema = {"type": "object"}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Second tool"
        mock_tool2.inputSchema = {"type": "string"}

        # Mock list_tools result
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool1, mock_tool2]

        # Mock session
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock stdio_client
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
            patch("nexus_dev.mcp_client.StdioServerParameters") as mock_params,
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

            manager = MCPClientManager()
            server = MCPServerConnection(name="test", command="python", args=["-m", "test"])

            tools = await manager.get_tools(server)

            assert len(tools) == 2
            assert tools[0].name == "tool1"
            assert tools[0].description == "First tool"
            assert tools[0].input_schema == {"type": "object"}
            assert tools[1].name == "tool2"
            assert tools[1].description == "Second tool"
            # Verify StdioServerParameters was created with correct args
            mock_params.assert_called_once_with(command="python", args=["-m", "test"], env=None)

    @pytest.mark.asyncio
    async def test_get_tools_with_env_vars(self):
        """Test getting tools with environment variable expansion."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test"
        mock_tool.inputSchema = {}

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
            patch("nexus_dev.mcp_client.StdioServerParameters") as mock_params,
            patch.dict(os.environ, {"API_KEY": "test_key_123"}),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

            manager = MCPClientManager()
            server = MCPServerConnection(
                name="test", command="python", args=[], env={"KEY": "${API_KEY}"}
            )

            tools = await manager.get_tools(server)

            # Verify StdioServerParameters was called with expanded env vars
            mock_params.assert_called_once_with(
                command="python", args=[], env={"KEY": "test_key_123"}
            )
            assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_get_tools_empty_result(self):
        """Test getting tools when server returns no tools."""
        mock_tools_result = MagicMock()
        mock_tools_result.tools = []

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
            patch("nexus_dev.mcp_client.StdioServerParameters"),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

            manager = MCPClientManager()
            server = MCPServerConnection(name="test", command="python", args=[])

            tools = await manager.get_tools(server)

            assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_get_tools_with_null_description(self):
        """Test getting tools when description is None."""
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = None  # Null description
        mock_tool.inputSchema = {"type": "object"}

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
            patch("nexus_dev.mcp_client.StdioServerParameters"),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

            manager = MCPClientManager()
            server = MCPServerConnection(name="test", command="python", args=[])

            tools = await manager.get_tools(server)

            assert len(tools) == 1
            assert tools[0].description == ""  # Should default to empty string

    @pytest.mark.asyncio
    async def test_get_tools_with_null_input_schema(self):
        """Test getting tools when inputSchema is None."""
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = None  # Null schema

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
            patch("nexus_dev.mcp_client.StdioServerParameters"),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

            manager = MCPClientManager()
            server = MCPServerConnection(name="test", command="python", args=[])

            tools = await manager.get_tools(server)

            assert len(tools) == 1
            assert tools[0].input_schema == {}  # Should default to empty dict

    @pytest.mark.asyncio
    async def test_get_tool_schema_found(self):
        """Test getting a specific tool schema when it exists."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "First tool"
        mock_tool1.inputSchema = {"type": "object"}

        mock_tool2 = MagicMock()
        mock_tool2.name = "target_tool"
        mock_tool2.description = "Target tool"
        mock_tool2.inputSchema = {"type": "string"}

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool1, mock_tool2]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
            patch("nexus_dev.mcp_client.StdioServerParameters"),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

            manager = MCPClientManager()
            server = MCPServerConnection(name="test", command="python", args=[])

            tool = await manager.get_tool_schema(server, "target_tool")

            assert tool is not None
            assert tool.name == "target_tool"
            assert tool.description == "Target tool"
            assert tool.input_schema == {"type": "string"}

    @pytest.mark.asyncio
    async def test_get_tool_schema_not_found(self):
        """Test getting a specific tool schema when it doesn't exist."""
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "First tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_streams = (mock_read, mock_write)

        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.ClientSession", return_value=mock_session),
            patch("nexus_dev.mcp_client.StdioServerParameters"),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

            manager = MCPClientManager()
            server = MCPServerConnection(name="test", command="python", args=[])

            tool = await manager.get_tool_schema(server, "nonexistent_tool")

            assert tool is None

    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test handling connection failures."""
        with (
            patch("nexus_dev.mcp_client.stdio_client") as mock_stdio,
            patch("nexus_dev.mcp_client.StdioServerParameters"),
        ):
            mock_stdio.side_effect = Exception("Connection failed")

            manager = MCPClientManager()
            server = MCPServerConnection(name="test", command="invalid", args=[])

            with pytest.raises(Exception, match="Connection failed"):
                await manager.get_tools(server)
