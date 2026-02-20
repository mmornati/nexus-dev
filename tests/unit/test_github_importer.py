from unittest.mock import MagicMock

import pytest

from nexus_dev.github_importer import GitHubImporter
from nexus_dev.mcp_client import MCPClientManager, MCPServerConnection


@pytest.fixture
def mock_database():
    return MagicMock()


@pytest.fixture
def mock_mcp_client_manager():
    return MagicMock(spec=MCPClientManager)


@pytest.fixture
def mock_mcp_config():
    config = MagicMock()
    config.servers = {"github": MagicMock()}
    return config


@pytest.mark.asyncio
async def test_fetch_tool_items_concatenation(
    mock_database, mock_mcp_client_manager, mock_mcp_config
):
    """Test that _fetch_tool_items correctly concatenates text content from multiple items."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test-project",
        client_manager=mock_mcp_client_manager,
        mcp_config=mock_mcp_config,
    )

    # Mock connection
    connection = MagicMock(spec=MCPServerConnection)

    # Mock tool result with multiple content items that form a valid JSON string
    # simulating a large JSON response broken into chunks
    content1 = MagicMock()
    content1.text = '[{"id": 1, "title": "Issue 1"}, '

    content2 = {"text": '{"id": 2, "title": "Issue 2"}]'}

    mock_result = MagicMock()
    mock_result.content = [content1, content2]

    mock_mcp_client_manager.call_tool.return_value = mock_result

    # Call the private method (or public method that calls it)
    # Since _fetch_tool_items is private, we'll access it directly for testing
    items = await importer._fetch_tool_items(
        connection=connection,
        tool_name="list_issues",
        owner="owner",
        repo="repo",
        limit=10,
        state="open",
    )

    assert len(items) == 2
    assert items[0]["id"] == 1
    assert items[0]["title"] == "Issue 1"
    assert items[1]["id"] == 2
    assert items[1]["title"] == "Issue 2"
