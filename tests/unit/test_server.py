"""Tests for MCP server tool handlers with mocked dependencies."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.database import DocumentType, SearchResult


# Mock SearchResult factory
def make_search_result(
    id="result-1",
    text="def foo(): pass",
    score=0.5,
    project_id="test",
    file_path="/foo.py",
    doc_type="code",
    chunk_type="function",
    language="python",
    name="foo",
    start_line=1,
    end_line=10,
    server_name="",
    parameters_schema="",
):
    """Create a mock SearchResult."""
    return SearchResult(
        id=id,
        text=text,
        score=score,
        project_id=project_id,
        file_path=file_path,
        doc_type=doc_type,
        chunk_type=chunk_type,
        language=language,
        name=name,
        start_line=start_line,
        end_line=end_line,
        server_name=server_name,
        parameters_schema=parameters_schema,
    )


@pytest.fixture
def mock_ctx():
    """Mock Context with session."""
    ctx = MagicMock()
    ctx.session = AsyncMock()
    return ctx


class TestSearchKnowledge:
    """Test suite for search_knowledge tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_returns_results(self, mock_get_config, mock_get_db):
        """Test search_knowledge returns formatted results."""
        from nexus_dev.server import search_knowledge

        # Setup mocks
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[make_search_result(name="my_function", text="def my_function(): pass")]
        )
        mock_get_db.return_value = mock_db

        result = await search_knowledge("find function")

        assert "my_function" in result
        assert "Search Results" in result
        mock_db.search.assert_called_once()

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_no_results(self, mock_get_config, mock_get_db):
        """Test search_knowledge with no results."""
        from nexus_dev.server import search_knowledge

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        result = await search_knowledge("nonexistent query")

        assert "No results found" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_with_content_type_filter(self, mock_get_config, mock_get_db):
        """Test search with content_type filter."""
        from nexus_dev.server import search_knowledge

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[make_search_result(doc_type="code")])
        mock_get_db.return_value = mock_db

        result = await search_knowledge("test", content_type="code")

        # Verify filter was applied
        call_args = mock_db.search.call_args
        assert call_args.kwargs["doc_type"] == DocumentType.CODE
        assert "[CODE]" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_clamps_limit(self, mock_get_config, mock_get_db):
        """Test that limit is clamped to 1-20."""
        from nexus_dev.server import search_knowledge

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        # Test too high
        await search_knowledge("test", limit=100)
        assert mock_db.search.call_args.kwargs["limit"] == 20

        # Test too low
        await search_knowledge("test", limit=0)
        assert mock_db.search.call_args.kwargs["limit"] == 1

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_handles_exception(self, mock_get_config, mock_get_db):
        """Test search handles database exceptions."""
        from nexus_dev.server import search_knowledge

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(side_effect=Exception("DB error"))
        mock_get_db.return_value = mock_db

        result = await search_knowledge("test")

        assert "Search failed" in result
        assert "DB error" in result


class TestSearchCode:
    """Test suite for search_code tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_code_returns_results(self, mock_get_config, mock_get_db):
        """Test search_code returns formatted code results."""
        from nexus_dev.server import search_code

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[make_search_result(chunk_type="function", name="validate")]
        )
        mock_get_db.return_value = mock_db

        result = await search_code("validate function")

        assert "validate" in result
        assert "Code Search" in result
        # Verify it searched only code
        call_args = mock_db.search.call_args
        assert call_args.kwargs["doc_type"] == DocumentType.CODE

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_code_no_results(self, mock_get_config, mock_get_db):
        """Test search_code with no results."""
        from nexus_dev.server import search_code

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        result = await search_code("nonexistent")

        assert "No code found" in result


class TestSearchDocs:
    """Test suite for search_docs tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_docs_returns_results(self, mock_get_config, mock_get_db):
        """Test search_docs returns formatted documentation results."""
        from nexus_dev.server import search_docs

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="documentation",
                    name="Installation",
                    text="# Installation\n\nRun pip install...",
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await search_docs("how to install")

        assert "Installation" in result
        assert "Documentation Search" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_docs_no_results(self, mock_get_config, mock_get_db):
        """Test search_docs with no results."""
        from nexus_dev.server import search_docs

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        result = await search_docs("nonexistent")

        assert "No documentation found" in result


class TestSearchLessons:
    """Test suite for search_lessons tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_lessons_returns_results(self, mock_get_config, mock_get_db):
        """Test search_lessons returns formatted lesson results."""
        from nexus_dev.server import search_lessons

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="lesson",
                    name="lesson_001",
                    text="## Problem\nTypeError\n\n## Solution\nAdd null check",
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await search_lessons("TypeError")

        assert "Lessons Found" in result
        assert "lesson_001" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_search_lessons_no_results(self, mock_get_config, mock_get_db):
        """Test search_lessons with no results shows tip."""
        from nexus_dev.server import search_lessons

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        result = await search_lessons("nonexistent")

        assert "No lessons found" in result
        assert "record_lesson" in result  # Tip to use record_lesson


class TestRecordLesson:
    """Test suite for record_lesson tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_embedder")
    @patch("nexus_dev.server._get_config")
    async def test_record_lesson_success(self, mock_get_config, mock_get_embedder, mock_get_db):
        """Test recording a lesson successfully."""
        from nexus_dev.server import record_lesson

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.upsert_document = AsyncMock(return_value="lesson-id")
        mock_get_db.return_value = mock_db

        result = await record_lesson("TypeError issue", "Added null check")

        assert "Lesson recorded" in result
        mock_db.upsert_document.assert_called_once()

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_embedder")
    @patch("nexus_dev.server._get_config")
    async def test_record_lesson_with_context(
        self, mock_get_config, mock_get_embedder, mock_get_db
    ):
        """Test recording a lesson with context."""
        from nexus_dev.server import record_lesson

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.upsert_document = AsyncMock(return_value="lesson-id")
        mock_get_db.return_value = mock_db

        result = await record_lesson(
            "TypeError issue", "Added null check", context="In user_service.py"
        )

        assert "Lesson recorded" in result
        # Verify context was included in the document
        call_args = mock_db.upsert_document.call_args
        doc = call_args[0][0]
        assert "user_service" in doc.text


class TestIndexFile:
    """Test suite for index_file tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_embedder")
    @patch("nexus_dev.server._get_config")
    @patch("nexus_dev.server.ChunkerRegistry")
    async def test_index_file_with_content(
        self, mock_registry, mock_get_config, mock_get_embedder, mock_get_db
    ):
        """Test indexing a file with provided content."""
        from nexus_dev.server import index_file

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.delete_by_file = AsyncMock(return_value=0)
        mock_db.upsert_documents = AsyncMock(return_value=["doc-1"])
        mock_get_db.return_value = mock_db

        # Mock chunker
        mock_chunk = MagicMock()
        mock_chunk.chunk_type.value = "function"
        mock_chunk.get_searchable_text.return_value = "def test(): pass"
        mock_chunk.file_path = "/test.py"
        mock_chunk.name = "test"
        mock_chunk.start_line = 1
        mock_chunk.end_line = 2
        mock_chunk.language = "python"
        mock_registry.chunk_file.return_value = [mock_chunk]
        mock_registry.get_language.return_value = "python"

        result = await index_file("/test.py", content="def test(): pass")

        assert "Indexed" in result
        assert "test.py" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_config")
    async def test_index_file_not_found(self, mock_get_config):
        """Test indexing a file that doesn't exist."""
        from nexus_dev.server import index_file

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_get_config.return_value = mock_config

        result = await index_file("/nonexistent/file.py")

        assert "Error" in result
        assert "not found" in result


class TestGetProjectContext:
    """Test suite for get_project_context tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    async def test_get_project_context(self, mock_get_config, mock_get_db):
        """Test getting project context with stats and lessons."""
        from nexus_dev.server import get_project_context

        mock_config = MagicMock()
        mock_config.project_id = "test"
        mock_config.project_name = "Test Project"
        mock_get_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.get_project_stats = AsyncMock(
            return_value={"total": 100, "code": 80, "documentation": 15, "lesson": 5}
        )
        mock_db.get_recent_lessons = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="lesson",
                    name="lesson_001",
                    text="## Problem\nBug\n\n## Solution\nFix",
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await get_project_context()

        assert "Test Project" in result
        assert "100" in result  # Total chunks
        assert "lesson_001" in result


class TestHelperFunctions:
    """Test suite for helper functions."""

    @patch("nexus_dev.server._config", None)
    @patch("nexus_dev.server.NexusConfig")
    @patch("nexus_dev.server.Path")
    @patch("nexus_dev.server._find_project_root")
    def test_get_config_loads_from_file(self, mock_find_root, mock_path, mock_nexus_config):
        """Test _get_config loads config when file exists."""
        # Reset global state
        import nexus_dev.server as server
        from nexus_dev.server import _get_config

        server._config = None

        # Setup _find_project_root to return a mock path
        mock_root = MagicMock()
        mock_find_root.return_value = mock_root

        # When _get_config calls root / "nexus_config.json" -> config_path
        mock_config_path = MagicMock()
        mock_root.__truediv__.return_value = mock_config_path
        mock_config_path.exists.return_value = True

        # Mock the config object returned by load
        mock_config_instance = MagicMock()
        mock_config_instance.project_id = "test-project"
        mock_nexus_config.load.return_value = mock_config_instance

        config = _get_config()

        assert config is not None
        assert config.project_id is not None
        mock_nexus_config.load.assert_called_once()

    @patch("nexus_dev.server._embedder", None)
    @patch("nexus_dev.server._get_config")
    @patch("nexus_dev.server.create_embedder")
    def test_get_embedder_creates_once(self, mock_create, mock_get_config):
        """Test _get_embedder creates embedder only once."""
        import nexus_dev.server as server
        from nexus_dev.server import _get_embedder

        server._embedder = None

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_embedder = MagicMock()
        mock_create.return_value = mock_embedder

        embedder1 = _get_embedder()
        embedder2 = _get_embedder()

        assert embedder1 is embedder2
        mock_create.assert_called_once()


class TestSearchTools:
    """Test suite for search_tools tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    async def test_search_tools_returns_results(self, mock_get_db):
        """Test search_tools returns formatted tool results."""
        from nexus_dev.server import search_tools

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="create_pull_request",
                    server_name="github",
                    text="Create a new pull request in a GitHub repository.",
                    parameters_schema=(
                        '{"type": "object", "properties": {"owner": {"type": "string"}}}'
                    ),
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await search_tools("create a pull request")

        assert "MCP Tools matching" in result
        assert "github.create_pull_request" in result
        assert "Create a new pull request" in result
        assert "Parameters:" in result
        assert '"owner"' in result
        # Verify it searched only tools
        call_args = mock_db.search.call_args
        assert call_args.kwargs["doc_type"] == DocumentType.TOOL

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    async def test_search_tools_no_results(self, mock_get_db):
        """Test search_tools with no results."""
        from nexus_dev.server import search_tools

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        result = await search_tools("nonexistent tool")

        assert "No tools found matching" in result
        assert "nonexistent tool" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    async def test_search_tools_filters_by_server(self, mock_get_db):
        """Test search_tools filters results by server name."""
        from nexus_dev.server import search_tools

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="create_issue",
                    server_name="github",
                    text="Create a GitHub issue",
                    parameters_schema='{"type": "object"}',
                ),
                make_search_result(
                    doc_type="tool",
                    name="send_notification",
                    server_name="homeassistant",
                    text="Send a notification",
                    parameters_schema='{"type": "object"}',
                ),
            ]
        )
        mock_get_db.return_value = mock_db

        result = await search_tools("create something", server="github")

        # Only github tool should be in results
        assert "github.create_issue" in result
        assert "homeassistant.send_notification" not in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    async def test_search_tools_no_results_with_server_filter(self, mock_get_db):
        """Test search_tools with no results after server filtering."""
        from nexus_dev.server import search_tools

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="send_notification",
                    server_name="homeassistant",
                    text="Send a notification",
                    parameters_schema='{"type": "object"}',
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await search_tools("send notification", server="github")

        assert "No tools found matching" in result
        assert "github" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    async def test_search_tools_clamps_limit(self, mock_get_db):
        """Test that limit is clamped to 1-10."""
        from nexus_dev.server import search_tools

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        # Test too high
        await search_tools("test", limit=100)
        assert mock_db.search.call_args.kwargs["limit"] == 10

        # Test too low
        await search_tools("test", limit=0)
        assert mock_db.search.call_args.kwargs["limit"] == 1

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    async def test_search_tools_without_parameters(self, mock_get_db):
        """Test search_tools formats results without parameters schema."""
        from nexus_dev.server import search_tools

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="simple_tool",
                    server_name="test",
                    text="A simple tool without parameters",
                    parameters_schema="",
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await search_tools("simple tool")

        assert "test.simple_tool" in result
        assert "A simple tool without parameters" in result
        # Should not have Parameters section if schema is empty
        lines = result.split("\n")
        params_count = sum(1 for line in lines if "**Parameters:**" in line)
        assert params_count == 0


class TestMCPConfigLoading:
    """Test suite for MCP configuration loading helpers."""

    @patch("nexus_dev.server._mcp_config", None)
    @patch("nexus_dev.server.MCPConfig")
    @patch("nexus_dev.server.Path")
    @patch("nexus_dev.server._find_project_root")
    def test_get_mcp_config_loads_from_file(self, mock_find_root, mock_path, mock_mcp_config):
        """Test _get_mcp_config loads config when file exists."""
        import nexus_dev.server as server
        from nexus_dev.server import _get_mcp_config

        server._mcp_config = None

        # Setup _find_project_root
        mock_root = MagicMock()
        mock_find_root.return_value = mock_root

        # mcp_config path
        mock_config_path = MagicMock()
        # root / ".nexus" / "mcp_config.json"
        mock_root.__truediv__.return_value.__truediv__.return_value = mock_config_path
        mock_config_path.exists.return_value = True

        mock_config_instance = MagicMock()
        mock_mcp_config.load.return_value = mock_config_instance

        config = _get_mcp_config()

        assert config is mock_config_instance
        mock_mcp_config.load.assert_called_once()

    @patch("nexus_dev.server._mcp_config", None)
    @patch("nexus_dev.server.Path")
    def test_get_mcp_config_returns_none_if_not_exists(self, mock_path):
        """Test _get_mcp_config returns None if file doesn't exist."""
        import nexus_dev.server as server
        from nexus_dev.server import _get_mcp_config

        server._mcp_config = None

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path.cwd.return_value.__truediv__.return_value.__truediv__.return_value = mock_path_obj

        config = _get_mcp_config()

        assert config is None

    @patch("nexus_dev.server._get_mcp_config")
    def test_get_active_server_names(self, mock_get_mcp_config):
        """Test _get_active_server_names returns enabled server names."""
        from nexus_dev.server import _get_active_server_names

        mock_config = MagicMock()
        mock_server1 = MagicMock()
        mock_server2 = MagicMock()

        mock_config.servers = {"server1": mock_server1, "server2": mock_server2}
        mock_config.get_active_servers.return_value = [mock_server1]
        mock_get_mcp_config.return_value = mock_config

        active_names = _get_active_server_names()

        assert active_names == ["server1"]

    @patch("nexus_dev.server._get_mcp_config")
    def test_get_active_server_names_no_config(self, mock_get_mcp_config):
        """Test _get_active_server_names returns empty list if no config."""
        from nexus_dev.server import _get_active_server_names

        mock_get_mcp_config.return_value = None

        active_names = _get_active_server_names()

        assert active_names == []

    @patch("nexus_dev.server._mcp_config", None)
    @patch("nexus_dev.server.MCPConfig")
    @patch("nexus_dev.server.Path")
    def test_get_mcp_config_handles_exception(self, mock_path, mock_mcp_config):
        """Test _get_mcp_config handles loading exceptions."""
        import nexus_dev.server as server
        from nexus_dev.server import _get_mcp_config

        server._mcp_config = None

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path.cwd.return_value.__truediv__.return_value.__truediv__.return_value = mock_path_obj

        mock_mcp_config.load.side_effect = Exception("Config error")

        config = _get_mcp_config()

        assert config is None


class TestListServers:
    """Test suite for list_servers tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_list_servers_no_config(self, mock_get_mcp_config):
        """Test list_servers returns message when no config exists."""
        from nexus_dev.server import list_servers

        mock_get_mcp_config.return_value = None

        result = await list_servers()

        assert "No MCP config" in result
        assert "nexus-mcp init" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_list_servers_with_active_servers(self, mock_get_mcp_config):
        """Test list_servers shows active servers."""
        from nexus_dev.server import list_servers

        mock_server1 = MagicMock()
        mock_server1.command = "python"
        mock_server1.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"test-server": mock_server1}
        mock_config.get_active_servers.return_value = [mock_server1]
        mock_get_mcp_config.return_value = mock_config

        result = await list_servers()

        assert "MCP Servers" in result
        assert "### Active" in result
        assert "test-server" in result
        assert "python" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_list_servers_with_disabled_servers(self, mock_get_mcp_config):
        """Test list_servers shows disabled servers."""
        from nexus_dev.server import list_servers

        mock_server1 = MagicMock()
        mock_server1.command = "python"
        mock_server1.enabled = True

        mock_server2 = MagicMock()
        mock_server2.command = "node"
        mock_server2.enabled = False

        mock_config = MagicMock()
        mock_config.servers = {
            "active-server": mock_server1,
            "disabled-server": mock_server2,
        }
        mock_config.get_active_servers.return_value = [mock_server1]
        mock_get_mcp_config.return_value = mock_config

        result = await list_servers()

        assert "### Active" in result
        assert "active-server" in result
        assert "### Disabled" in result
        assert "disabled-server" in result
        assert "disabled" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_list_servers_empty_config(self, mock_get_mcp_config):
        """Test list_servers with empty servers config."""
        from nexus_dev.server import list_servers

        mock_config = MagicMock()
        mock_config.servers = {}
        mock_config.get_active_servers.return_value = []
        mock_get_mcp_config.return_value = mock_config

        result = await list_servers()

        assert "MCP Servers" in result
        assert "No active servers" in result
        assert "No disabled servers" in result


class TestMain:
    """Test suite for main entry point."""

    @patch("sys.argv", ["nexus-dev"])
    @patch("nexus_dev.server.mcp")
    @patch("nexus_dev.server._get_mcp_config")
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    def test_main_initializes_components(
        self, mock_get_config, mock_get_db, mock_get_mcp_config, mock_mcp
    ):
        """Test main initializes components and runs server."""
        from nexus_dev.server import main

        main()

        mock_get_config.assert_called_once()
        mock_get_db.assert_called_once()
        mock_get_mcp_config.assert_called_once()
        mock_mcp.run.assert_called_once_with(transport="stdio")

    @patch("sys.argv", ["nexus-dev"])
    @patch("nexus_dev.server.mcp")
    @patch("nexus_dev.server._get_mcp_config")
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_config")
    def test_main_handles_init_exception(
        self, mock_get_config, mock_get_db, mock_get_mcp_config, mock_mcp
    ):
        """Test main handles initialization exceptions."""
        from nexus_dev.server import main

        mock_get_config.side_effect = Exception("Init error")

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1

        mock_get_config.assert_called_once()
        mock_mcp.run.assert_not_called()


class TestGetToolSchema:
    """Test suite for get_tool_schema tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_tool_schema_no_config(self, mock_get_mcp_config):
        """Test get_tool_schema returns message when no config exists."""
        from nexus_dev.server import get_tool_schema

        mock_get_mcp_config.return_value = None

        result = await get_tool_schema("github", "create_issue")

        assert "No MCP config" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_tool_schema_server_not_found(self, mock_get_mcp_config):
        """Test get_tool_schema returns error for missing server."""
        from nexus_dev.server import get_tool_schema

        mock_config = MagicMock()
        mock_config.servers = {"github": MagicMock()}
        mock_get_mcp_config.return_value = mock_config

        result = await get_tool_schema("unknown", "some_tool")

        assert "Server not found" in result
        assert "unknown" in result
        assert "Available" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_tool_schema_server_disabled(self, mock_get_mcp_config):
        """Test get_tool_schema returns error for disabled server."""
        from nexus_dev.server import get_tool_schema

        mock_server = MagicMock()
        mock_server.enabled = False

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        result = await get_tool_schema("github", "create_issue")

        assert "Server is disabled" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_tool_schema_success(self, mock_get_mcp_config, mock_get_conn_manager):
        """Test get_tool_schema returns valid schema."""
        import json

        from nexus_dev.server import get_tool_schema

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        # Mock tool result
        mock_tool = MagicMock()
        mock_tool.name = "create_issue"
        mock_tool.description = "Create a GitHub issue"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        }

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_result

        mock_conn_manager = MagicMock()
        mock_conn_manager.get_connection = AsyncMock(return_value=mock_session)
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await get_tool_schema("github", "create_issue")

        parsed = json.loads(result)
        assert parsed["server"] == "github"
        assert parsed["tool"] == "create_issue"
        assert parsed["description"] == "Create a GitHub issue"
        assert "properties" in parsed["parameters"]

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_tool_schema_tool_not_found(self, mock_get_mcp_config, mock_get_conn_manager):
        """Test get_tool_schema returns error for missing tool."""
        from nexus_dev.server import get_tool_schema

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        # Mock tool result with different tools
        mock_tool = MagicMock()
        mock_tool.name = "other_tool"

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_result

        mock_conn_manager = MagicMock()
        mock_conn_manager.get_connection = AsyncMock(return_value=mock_session)
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await get_tool_schema("github", "nonexistent")

        assert "Tool not found" in result
        assert "github.nonexistent" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_tool_schema_connection_error(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test get_tool_schema handles connection errors."""
        from nexus_dev.server import get_tool_schema

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        mock_conn_manager = MagicMock()
        mock_conn_manager.get_connection = AsyncMock(side_effect=Exception("Connection refused"))
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await get_tool_schema("github", "create_issue")

        assert "Error connecting" in result
        assert "github" in result
        assert "Connection refused" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_tool_schema_handles_timeout_error(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test get_tool_schema handles MCPTimeoutError."""
        from nexus_dev.gateway.connection_manager import MCPTimeoutError
        from nexus_dev.server import get_tool_schema

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        mock_conn_manager = MagicMock()
        mock_conn_manager.get_connection = AsyncMock(
            side_effect=MCPTimeoutError("Connection timed out")
        )
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await get_tool_schema("github", "create_issue")

        assert "Error connecting" in result
        assert "timed out" in result


class TestInvokeTool:
    """Test suite for invoke_tool tool."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_no_config(self, mock_get_mcp_config):
        """Test invoke_tool returns message when no config exists."""
        from nexus_dev.server import invoke_tool

        mock_get_mcp_config.return_value = None

        result = await invoke_tool("github", "create_issue", {"title": "Test"})

        assert "No MCP config" in result
        assert "nexus-mcp init" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_server_not_found(self, mock_get_mcp_config):
        """Test invoke_tool returns error for missing server."""
        from nexus_dev.server import invoke_tool

        mock_config = MagicMock()
        mock_config.servers = {"github": MagicMock()}
        mock_get_mcp_config.return_value = mock_config

        result = await invoke_tool("unknown", "some_tool", {})

        assert "Server not found" in result
        assert "unknown" in result
        assert "Available" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_server_disabled(self, mock_get_mcp_config):
        """Test invoke_tool returns error for disabled server."""
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = False

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        result = await invoke_tool("github", "create_issue", {})

        assert "Server is disabled" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_success_with_text_content(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test invoke_tool returns formatted result with text content."""
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        # Mock result with text content
        mock_content_item = MagicMock()
        mock_content_item.text = "Issue created: #123"

        mock_result = MagicMock()
        mock_result.content = [mock_content_item]

        mock_conn_manager = MagicMock()
        mock_conn_manager.invoke_tool = AsyncMock(return_value=mock_result)
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await invoke_tool("github", "create_issue", {"title": "Test Issue"})

        assert "Issue created: #123" in result
        mock_conn_manager.invoke_tool.assert_called_once_with(
            "github", mock_server, "create_issue", {"title": "Test Issue"}
        )

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_success_no_arguments(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test invoke_tool with no arguments passes empty dict."""
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"homeassistant": mock_server}
        mock_get_mcp_config.return_value = mock_config

        mock_result = MagicMock()
        mock_result.content = []

        mock_conn_manager = MagicMock()
        mock_conn_manager.invoke_tool = AsyncMock(return_value=mock_result)
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await invoke_tool("homeassistant", "turn_on_lights")

        assert "Tool executed successfully" in result
        mock_conn_manager.invoke_tool.assert_called_once_with(
            "homeassistant", mock_server, "turn_on_lights", {}
        )

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_handles_connection_error(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test invoke_tool handles connection errors."""
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        mock_conn_manager = MagicMock()
        mock_conn_manager.invoke_tool = AsyncMock(side_effect=Exception("Connection refused"))
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await invoke_tool("github", "create_issue", {})

        assert "Tool invocation failed" in result
        assert "Connection refused" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_handles_timeout_error(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test invoke_tool handles MCPTimeoutError specifically."""
        from nexus_dev.gateway.connection_manager import MCPTimeoutError
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        mock_conn_manager = MagicMock()
        mock_conn_manager.invoke_tool = AsyncMock(
            side_effect=MCPTimeoutError("Tool 'slow_tool' timed out after 30.0s")
        )
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await invoke_tool("github", "slow_tool", {})

        assert "Tool invocation failed" in result
        assert "timed out" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_handles_mcp_connection_error(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test invoke_tool handles MCPConnectionError specifically."""
        from nexus_dev.gateway.connection_manager import MCPConnectionError
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        mock_conn_manager = MagicMock()
        mock_conn_manager.invoke_tool = AsyncMock(
            side_effect=MCPConnectionError("Failed to connect to github after 3 attempts")
        )
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await invoke_tool("github", "create_issue", {})

        assert "Tool invocation failed" in result
        assert "Failed to connect" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_multiple_content_items(
        self, mock_get_mcp_config, mock_get_conn_manager
    ):
        """Test invoke_tool formats multiple content items."""
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"github": mock_server}
        mock_get_mcp_config.return_value = mock_config

        # Mock result with multiple content items
        mock_item1 = MagicMock()
        mock_item1.text = "Line 1"
        mock_item2 = MagicMock()
        mock_item2.text = "Line 2"

        mock_result = MagicMock()
        mock_result.content = [mock_item1, mock_item2]

        mock_conn_manager = MagicMock()
        mock_conn_manager.invoke_tool = AsyncMock(return_value=mock_result)
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await invoke_tool("github", "some_tool", {})

        assert "Line 1" in result
        assert "Line 2" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_connection_manager")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_invoke_tool_non_text_content(self, mock_get_mcp_config, mock_get_conn_manager):
        """Test invoke_tool handles non-text content items."""
        from nexus_dev.server import invoke_tool

        mock_server = MagicMock()
        mock_server.enabled = True

        mock_config = MagicMock()
        mock_config.servers = {"test": mock_server}
        mock_get_mcp_config.return_value = mock_config

        # Create a simple object without 'text' attribute
        class NonTextContent:
            def __str__(self):
                return "Binary content"

        mock_item = NonTextContent()

        mock_result = MagicMock()
        mock_result.content = [mock_item]

        mock_conn_manager = MagicMock()
        mock_conn_manager.invoke_tool = AsyncMock(return_value=mock_result)
        mock_get_conn_manager.return_value = mock_conn_manager

        result = await invoke_tool("test", "binary_tool", {})

        assert "Binary content" in result


class TestGetActiveToolsResource:
    """Test suite for get_active_tools_resource resource."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_active_server_names")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_active_tools_no_config(
        self, mock_get_mcp_config, mock_get_active_server_names, mock_get_db
    ):
        """Test get_active_tools_resource returns message when no config exists."""
        from nexus_dev.server import get_active_tools_resource

        mock_get_mcp_config.return_value = None

        result = await get_active_tools_resource()

        assert "No MCP config" in result
        assert "nexus-mcp init" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_active_server_names")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_active_tools_no_active_servers(
        self, mock_get_mcp_config, mock_get_active_server_names, mock_get_db
    ):
        """Test get_active_tools_resource with no active servers."""
        from nexus_dev.server import get_active_tools_resource

        mock_config = MagicMock()
        mock_config.active_profile = "default"
        mock_get_mcp_config.return_value = mock_config
        mock_get_active_server_names.return_value = []

        result = await get_active_tools_resource()

        assert "No active servers" in result
        assert "default" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_active_server_names")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_active_tools_with_tools(
        self, mock_get_mcp_config, mock_get_active_server_names, mock_get_db
    ):
        """Test get_active_tools_resource returns tools grouped by server."""
        from nexus_dev.server import get_active_tools_resource

        mock_config = MagicMock()
        mock_config.active_profile = "development"
        mock_get_mcp_config.return_value = mock_config

        mock_get_active_server_names.return_value = ["github", "homeassistant"]

        # Mock database search results
        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="create_issue",
                    server_name="github",
                    text="Create a new GitHub issue in a repository",
                ),
                make_search_result(
                    doc_type="tool",
                    name="create_pull_request",
                    server_name="github",
                    text="Create a new pull request",
                ),
                make_search_result(
                    doc_type="tool",
                    name="turn_on_light",
                    server_name="homeassistant",
                    text="Turn on a light in Home Assistant",
                ),
            ]
        )
        mock_get_db.return_value = mock_db

        result = await get_active_tools_resource()

        # Check header
        assert "Active Tools (profile: development)" in result

        # Check server sections
        assert "## github" in result
        assert "## homeassistant" in result

        # Check tools are listed
        assert "create_issue" in result
        assert "create_pull_request" in result
        assert "turn_on_light" in result

        # Check descriptions are included
        assert "Create a new GitHub issue" in result
        assert "Turn on a light" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_active_server_names")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_active_tools_server_with_no_tools(
        self, mock_get_mcp_config, mock_get_active_server_names, mock_get_db
    ):
        """Test get_active_tools_resource with server that has no tools."""
        from nexus_dev.server import get_active_tools_resource

        mock_config = MagicMock()
        mock_config.active_profile = "default"
        mock_get_mcp_config.return_value = mock_config

        mock_get_active_server_names.return_value = ["empty-server", "github"]

        # Mock database returns tools only for github
        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="create_issue",
                    server_name="github",
                    text="Create a GitHub issue",
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await get_active_tools_resource()

        # Both servers should be in output
        assert "## empty-server" in result
        assert "## github" in result

        # Empty server should show "No tools found"
        assert "No tools found" in result
        assert "create_issue" in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_active_server_names")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_active_tools_truncates_long_descriptions(
        self, mock_get_mcp_config, mock_get_active_server_names, mock_get_db
    ):
        """Test get_active_tools_resource truncates descriptions longer than 100 chars."""
        from nexus_dev.server import get_active_tools_resource

        mock_config = MagicMock()
        mock_config.active_profile = "default"
        mock_get_mcp_config.return_value = mock_config

        mock_get_active_server_names.return_value = ["test-server"]

        long_description = "A" * 150  # 150 character description

        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="long_tool",
                    server_name="test-server",
                    text=long_description,
                )
            ]
        )
        mock_get_db.return_value = mock_db

        result = await get_active_tools_resource()

        # Check that description is truncated to 100 chars + "..."
        assert "long_tool" in result
        assert "..." in result
        # The full 150-char description should not be in the output
        assert long_description not in result

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_active_server_names")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_active_tools_with_high_limit(
        self, mock_get_mcp_config, mock_get_active_server_names, mock_get_db
    ):
        """Test get_active_tools_resource queries with high limit for all tools."""
        from nexus_dev.server import get_active_tools_resource

        mock_config = MagicMock()
        mock_config.active_profile = "default"
        mock_get_mcp_config.return_value = mock_config

        mock_get_active_server_names.return_value = ["test-server"]

        mock_db = MagicMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_get_db.return_value = mock_db

        await get_active_tools_resource()

        # Verify search was called with limit=1000 to get all tools
        call_args = mock_db.search.call_args
        assert call_args.kwargs["limit"] == 1000

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_database")
    @patch("nexus_dev.server._get_active_server_names")
    @patch("nexus_dev.server._get_mcp_config")
    async def test_get_active_tools_filters_by_active_servers(
        self, mock_get_mcp_config, mock_get_active_server_names, mock_get_db
    ):
        """Test get_active_tools_resource only shows tools from active servers.

        This test verifies that when multiple servers' tools exist in the database,
        only tools from servers in the active profile are returned.
        """
        from nexus_dev.server import get_active_tools_resource

        mock_config = MagicMock()
        mock_config.active_profile = "dev"
        mock_get_mcp_config.return_value = mock_config

        # Only github is in the active profile
        mock_get_active_server_names.return_value = ["github"]

        # Mock database returns tools from both active and inactive servers
        mock_db = MagicMock()
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(
                    doc_type="tool",
                    name="create_issue",
                    server_name="github",  # Active server
                    text="GitHub tool",
                ),
                make_search_result(
                    doc_type="tool",
                    name="turn_on_light",
                    server_name="homeassistant",  # Inactive server
                    text="Home Assistant tool",
                ),
            ]
        )
        mock_get_db.return_value = mock_db

        result = await get_active_tools_resource()

        # Only github tool should be in output (active server)
        assert "github" in result
        assert "create_issue" in result
        # homeassistant tools should NOT be shown (inactive server)
        assert "homeassistant" not in result
        assert "turn_on_light" not in result


class TestProjectRootDiscovery:
    """Test suite for project root discovery functions."""

    @pytest.mark.asyncio
    @patch("nexus_dev.server._get_project_root_from_session")
    @patch("os.environ", {"NEXUS_PROJECT_ROOT": "/env/project"})
    async def test_find_project_root_from_env(self, mock_get_root_session):
        """Test _find_project_root using environment variable."""
        from nexus_dev.server import _find_project_root

        # We need to mock existence check for /env/project/nexus_config.json
        def mock_exists(self):
            s = str(self).rstrip("/")
            return s in ["/env/project", "/env/project/nexus_config.json"]

        with (
            patch.object(Path, "exists", autospec=True, side_effect=mock_exists),
            patch("nexus_dev.server._project_root", None),
        ):
            root = _find_project_root()
            assert root == Path("/env/project")

    @pytest.mark.asyncio
    @patch("nexus_dev.server.Path.cwd")
    async def test_find_project_root_by_walking_up(self, mock_cwd):
        """Test _find_project_root by walking up from current directory."""
        from nexus_dev.server import _find_project_root

        # Mock Path.cwd().resolve() to return /a/b/c
        mock_cwd.return_value.resolve.return_value = Path("/a/b/c")

        # Mock existence: only /a has nexus_config.json
        def mock_exists(self):
            s = str(self).rstrip("/")
            return s in ["/a", "/a/nexus_config.json"]

        with (
            patch.object(Path, "exists", autospec=True, side_effect=mock_exists),
            patch("nexus_dev.server._project_root", None),
        ):
            root = _find_project_root()
            assert root == Path("/a")

    @pytest.mark.asyncio
    async def test_get_project_root_from_session_success(self, mock_ctx):
        """Test _get_project_root_from_session with valid root."""
        from nexus_dev.server import _get_project_root_from_session

        mock_root = MagicMock()
        mock_root.uri = "file:///test/project"
        mock_ctx.session.list_roots.return_value = MagicMock(roots=[mock_root])

        # Mock existence of both city and nexus_config.json
        def mock_exists(self):
            s = str(self).rstrip("/")
            return s in ["/test/project", "/test/project/nexus_config.json"]

        with patch.object(Path, "exists", autospec=True, side_effect=mock_exists):
            root = await _get_project_root_from_session(mock_ctx)
            assert root == Path("/test/project")

    @pytest.mark.asyncio
    async def test_get_project_root_from_session_fallback(self, mock_ctx):
        """Test fallback to first root if no nexus_config.json found."""
        from nexus_dev.server import _get_project_root_from_session

        mock_root = MagicMock()
        mock_root.uri = "file:///test/root"
        mock_ctx.session.list_roots.return_value = MagicMock(roots=[mock_root])

        # Mock existence: /test/root/nexus_config.json does NOT exist,
        # but /test/root (the directory itself) DOES exist.
        def mock_exists(self):
            return str(self).rstrip("/") == "/test/root"

        with patch.object(Path, "exists", autospec=True, side_effect=mock_exists):
            root = await _get_project_root_from_session(mock_ctx)
            assert root == Path("/test/root")
