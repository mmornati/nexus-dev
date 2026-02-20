"""Tests for GitHubImporter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus_dev.database import DocumentType, NexusDatabase
from nexus_dev.github_importer import GitHubImporter
from nexus_dev.mcp_client import MCPClientManager, MCPServerConnection
from nexus_dev.mcp_config import MCPConfig, MCPServerConfig


@pytest.fixture
def mock_database():
    """Mock NexusDatabase."""
    db = MagicMock(spec=NexusDatabase)
    db.embedder = AsyncMock()
    db.embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]  # Dummy embedding
    db.upsert_documents = AsyncMock()
    return db


@pytest.fixture
def mock_client_manager():
    """Mock MCPClientManager."""
    manager = MagicMock(spec=MCPClientManager)
    manager.call_tool = AsyncMock()
    return manager


@pytest.fixture
def mock_mcp_config():
    """Mock MCPConfig with GitHub server."""
    config = MagicMock(spec=MCPConfig)
    config.servers = {
        "github": MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": "test_token"},
            transport="stdio",
        )
    }
    return config


@pytest.mark.asyncio
async def test_fetch_tool_items_concatenation(
    mock_database, mock_client_manager, mock_mcp_config
):
    """Test that _fetch_tool_items correctly concatenates text content from multiple items."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test-project",
        client_manager=mock_client_manager,
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

    mock_client_manager.call_tool.return_value = mock_result

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


def test_github_importer_init(mock_database, mock_client_manager, mock_mcp_config):
    """Test initialization of GitHubImporter."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=mock_mcp_config,
    )

    assert importer.database == mock_database
    assert importer.project_id == "test_project"
    assert importer.client_manager == mock_client_manager
    assert importer.mcp_config == mock_mcp_config


@pytest.mark.asyncio
async def test_import_issues_success(mock_database, mock_client_manager, mock_mcp_config):
    """Test successful import of issues and PRs."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=mock_mcp_config,
    )

    # Mock tool responses
    issues_response = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    [
                        {
                            "number": 1,
                            "title": "Test Issue 1",
                            "body": "This is a test issue.",
                            "html_url": "https://github.com/owner/repo/issues/1",
                            "state": "open",
                            "user": {"login": "tester"},
                        }
                    ]
                ),
            }
        ]
    }

    prs_response = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    [
                        {
                            "number": 2,
                            "title": "Test PR 1",
                            "body": "This is a test PR.",
                            "html_url": "https://github.com/owner/repo/pull/2",
                            "state": "open",
                            "user": {"login": "tester"},
                            "pull_request": {},  # Mark as PR
                        }
                    ]
                ),
            }
        ]
    }

    # Set up call_tool to return different responses based on tool_name
    async def side_effect(connection, tool_name, args):
        if tool_name == "list_issues":
            return issues_response
        elif tool_name == "list_pull_requests":
            return prs_response
        return {}

    mock_client_manager.call_tool.side_effect = side_effect

    # Mock embedder to return correct number of embeddings
    mock_database.embedder.embed_batch.return_value = [
        [0.1] * 1536,
        [0.2] * 1536,
    ]  # Two documents

    count = await importer.import_issues("owner", "repo")

    assert count == 2
    assert mock_database.upsert_documents.called
    args, _ = mock_database.upsert_documents.call_args
    docs = args[0]
    assert len(docs) == 2

    # Check first document (Issue)
    doc1 = docs[0]
    assert doc1.project_id == "test_project"
    assert doc1.file_path == "https://github.com/owner/repo/issues/1"
    assert doc1.doc_type == DocumentType.GITHUB_ISSUE
    assert "Test Issue 1" in doc1.name

    # Check second document (PR)
    doc2 = docs[1]
    assert doc2.project_id == "test_project"
    assert doc2.file_path == "https://github.com/owner/repo/pull/2"
    assert doc2.doc_type == DocumentType.GITHUB_PR
    assert "Test PR 1" in doc2.name


@pytest.mark.asyncio
async def test_import_issues_missing_config(mock_database, mock_client_manager):
    """Test error when MCP config is missing."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=None,  # Missing config
    )

    with pytest.raises(ValueError, match="MCP Config is required"):
        await importer.import_issues("owner", "repo")


@pytest.mark.asyncio
async def test_import_issues_missing_server(mock_database, mock_client_manager):
    """Test error when 'github' server is missing from config."""
    config = MagicMock(spec=MCPConfig)
    config.servers = {}  # Empty servers

    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=config,
    )

    with pytest.raises(ValueError, match="Server 'github' not found"):
        await importer.import_issues("owner", "repo")


@pytest.mark.asyncio
async def test_import_issues_empty_results(mock_database, mock_client_manager, mock_mcp_config):
    """Test behavior when no issues or PRs are found."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=mock_mcp_config,
    )

    # Mock empty responses
    mock_client_manager.call_tool.return_value = {"content": []}

    count = await importer.import_issues("owner", "repo")

    assert count == 0
    assert not mock_database.upsert_documents.called


@pytest.mark.asyncio
async def test_import_issues_tool_failure(mock_database, mock_client_manager, mock_mcp_config):
    """Test behavior when tool call fails (raises exception)."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=mock_mcp_config,
    )

    # Mock tool call failure
    mock_client_manager.call_tool.side_effect = Exception("Tool failure")

    # Should log warning but not raise exception for tool call itself, returns empty list
    count = await importer.import_issues("owner", "repo")

    assert count == 0
    assert not mock_database.upsert_documents.called


@pytest.mark.asyncio
async def test_import_issues_json_error(mock_database, mock_client_manager, mock_mcp_config):
    """Test behavior when tool returns invalid JSON."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=mock_mcp_config,
    )

    # Mock invalid JSON response
    mock_client_manager.call_tool.return_value = {
        "content": [{"type": "text", "text": "Invalid JSON"}]
    }

    count = await importer.import_issues("owner", "repo")

    assert count == 0
    assert not mock_database.upsert_documents.called


@pytest.mark.asyncio
async def test_import_issues_non_list_response(
    mock_database, mock_client_manager, mock_mcp_config
):
    """Test when tool returns a dict or single item instead of a list."""
    importer = GitHubImporter(
        database=mock_database,
        project_id="test_project",
        client_manager=mock_client_manager,
        mcp_config=mock_mcp_config,
    )

    # Mock response wrapped in a dict
    mock_client_manager.call_tool.return_value = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "items": [
                            {
                                "number": 1,
                                "title": "Issue",
                                "state": "open",
                                "html_url": "url",
                                "user": {"login": "user"},
                            }
                        ]
                    }
                ),
            }
        ]
    }

    # Mock embedder
    mock_database.embedder.embed_batch.return_value = [[0.1] * 1536] * 2

    # Since we fetch both issues and PRs, let's make sure both calls return something
    # valid or handled. If we return the same response for both calls, we'll get
    # 2 items total (1 issue, 1 PR/Issue treated as PR)

    count = await importer.import_issues("owner", "repo")

    # Should find 2 items (one from list_issues, one from list_pull_requests)
    assert count == 2
    assert mock_database.upsert_documents.called
