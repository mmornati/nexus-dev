"""Unit tests for dynamic agent tool registration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from typing import Any

from nexus_dev.server import refresh_agents, _register_agent_tools
from nexus_dev.agents import AgentManager, AgentConfig

@pytest.fixture
def mock_ctx():
    """Mock FastMCP Context with session."""
    ctx = MagicMock()
    ctx.session = AsyncMock()
    return ctx

@pytest.fixture
def mock_db():
    """Mock NexusDatabase."""
    return MagicMock()

class TestDynamicAgents:
    """Test suite for dynamic agent registration."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_project_root_from_session")
    @patch("nexus_dev.server.AgentManager")
    @patch("nexus_dev.server.mcp")
    async def test_refresh_agents_success(self, mock_mcp, mock_agent_manager_class, mock_get_root, mock_get_db, mock_ctx):
        """Test successful agent refresh and registration."""
        # Setup
        mock_get_db.return_value = MagicMock()
        mock_get_root.return_value = Path("/test/project")
        
        # Mock agents directory
        with patch.object(Path, "exists", return_value=True):
            # Mock AgentManager
            agent1 = MagicMock(spec=AgentConfig)
            agent1.name = "agent1"
            agent1.description = "Agent 1"
            
            mock_manager = MagicMock()
            mock_manager.__iter__.return_value = [agent1]
            mock_manager.__len__.return_value = 1
            mock_agent_manager_class.return_value = mock_manager
            
            # Execute
            result = await refresh_agents(mock_ctx)
            
            # Verify root was queried
            mock_get_root.assert_called_once_with(mock_ctx)
            
            # Verify AgentManager was initialized with correct path
            mock_agent_manager_class.assert_called_once_with(agents_dir=Path("/test/project/agents"))
            
            # Verify tool was added to MCP
            # mcp.add_tool(fn=..., name="ask_agent1", description="Agent 1")
            mock_mcp.add_tool.assert_called()
            call_kwargs = mock_mcp.add_tool.call_args.kwargs
            assert call_kwargs["name"] == "ask_agent1"
            assert call_kwargs["description"] == "Agent 1"
            
            # Verify notification was sent
            mock_ctx.session.send_tool_list_changed.assert_called_once()
            
            assert "Successfully registered 1 agent tools" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_project_root_from_session")
    async def test_refresh_agents_no_root(self, mock_get_root, mock_get_db, mock_ctx):
        """Test refresh fails if no root found."""
        mock_get_db.return_value = MagicMock()
        mock_get_root.return_value = None
        
        result = await refresh_agents(mock_ctx)
        
        assert "No nexus project root found" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_project_root_from_session")
    async def test_refresh_agents_no_agents_dir(self, mock_get_root, mock_get_db, mock_ctx):
        """Test refresh fails if agents directory missing."""
        mock_get_db.return_value = MagicMock()
        mock_get_root.return_value = Path("/test/project")
        
        # Force Path.exists to return False for agents dir
        with patch.object(Path, "exists", return_value=False):
            result = await refresh_agents(mock_ctx)
            assert "No agents directory found" in result
