"""Tests for MCP server tool handlers with mocked dependencies."""

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
    def test_get_config_loads_from_file(self, mock_path, mock_nexus_config):
        """Test _get_config loads config when file exists."""
        # Reset global state
        import nexus_dev.server as server
        from nexus_dev.server import _get_config

        server._config = None

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path.cwd.return_value.__truediv__.return_value = mock_path_obj

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
    def test_get_mcp_config_loads_from_file(self, mock_path, mock_mcp_config):
        """Test _get_mcp_config loads config when file exists."""
        import nexus_dev.server as server
        from nexus_dev.server import _get_mcp_config

        server._mcp_config = None

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path.cwd.return_value.__truediv__.return_value.__truediv__.return_value = mock_path_obj

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


class TestMain:
    """Test suite for main entry point."""

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

        main()

        mock_get_config.assert_called_once()
        mock_mcp.run.assert_called_once_with(transport="stdio")
