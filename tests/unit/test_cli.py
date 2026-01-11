"""Tests for CLI commands with Click's CliRunner."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from nexus_dev.cli import cli


@pytest.fixture
def runner():
    """Create a Click CliRunner."""
    return CliRunner()


class TestCliInit:
    """Test suite for nexus-init command."""

    def test_init_creates_config(self, runner, tmp_path):
        """Test init command creates configuration."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["init", "--project-name", "test-project", "--no-hook"],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            assert "Created nexus_config.json" in result.output
            assert (Path.cwd() / "nexus_config.json").exists()

    def test_init_creates_lessons_directory(self, runner, tmp_path):
        """Test init command creates .nexus/lessons directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--project-name", "test", "--no-hook"])

            assert result.exit_code == 0
            lessons_dir = Path.cwd() / ".nexus" / "lessons"
            assert lessons_dir.exists()

    def test_init_with_ollama_provider(self, runner, tmp_path):
        """Test init with ollama embedding provider."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                [
                    "init",
                    "--project-name",
                    "test",
                    "--embedding-provider",
                    "ollama",
                    "--no-hook",
                ],
            )

            assert result.exit_code == 0
            # Should not show OpenAI warning
            assert "OPENAI_API_KEY" not in result.output

    def test_init_warns_openai_api_key(self, runner, tmp_path):
        """Test init warns about OPENAI_API_KEY when using openai provider."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                [
                    "init",
                    "--project-name",
                    "test",
                    "--embedding-provider",
                    "openai",
                    "--no-hook",
                ],
            )

            assert result.exit_code == 0
            assert "OPENAI_API_KEY" in result.output

    def test_init_existing_config_abort(self, runner, tmp_path):
        """Test init aborts if config exists and user declines overwrite."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            result = runner.invoke(
                cli,
                ["init", "--project-name", "test", "--no-hook"],
                input="n\n",  # Decline overwrite
            )

            assert "Aborted" in result.output


class TestCliStatus:
    """Test suite for nexus-status command."""

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    def test_status_shows_project_info(
        self, mock_config_cls, mock_db_cls, mock_embedder_fn, runner, tmp_path
    ):
        """Test status command shows project information."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config file
            (Path.cwd() / "nexus_config.json").write_text(
                '{"project_id": "test", "project_name": "Test", '
                '"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"}'
            )

            # Mock config
            mock_config = MagicMock()
            mock_config.project_name = "Test Project"
            mock_config.project_id = "test-123"
            mock_config.embedding_provider = "openai"
            mock_config.embedding_model = "text-embedding-3-small"
            mock_config.get_db_path.return_value = Path("/tmp/db")
            mock_config_cls.load.return_value = mock_config

            # Mock database
            mock_db = MagicMock()
            mock_stats = {"total": 50, "code": 40, "documentation": 8, "lesson": 2}

            async def mock_get_stats(project_id):
                return mock_stats

            mock_db.get_project_stats = mock_get_stats
            mock_db_cls.return_value = mock_db

            result = runner.invoke(cli, ["status"])

            assert "Test Project" in result.output
            assert "test-123" in result.output

    def test_status_not_initialized(self, runner, tmp_path):
        """Test status command when not initialized."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["status"])

            assert "not initialized" in result.output


class TestCliIndex:
    """Test suite for nexus-index command."""

    def test_index_no_config(self, runner, tmp_path):
        """Test index fails gracefully without config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["index", "file.py"])

            assert "nexus_config.json not found" in result.output

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.ChunkerRegistry")
    def test_index_file_success(
        self,
        mock_registry,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test indexing a file successfully."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create test file
            test_file = Path.cwd() / "test.py"
            test_file.write_text("def hello(): pass")

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config.exclude_patterns = []
            mock_config.include_patterns = ["*.py"]
            mock_config_cls.load.return_value = mock_config

            # Mock embedder
            mock_embedder = MagicMock()
            mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
            mock_embedder_fn.return_value = mock_embedder

            # Mock database
            mock_db = MagicMock()
            mock_db.delete_by_file = AsyncMock(return_value=0)
            mock_db.upsert_documents = AsyncMock(return_value=["doc-1"])
            mock_db_cls.return_value = mock_db

            # Mock chunker
            mock_chunk = MagicMock()
            mock_chunk.chunk_type.value = "function"
            mock_chunk.get_searchable_text.return_value = "def hello(): pass"
            mock_chunk.file_path = str(test_file)
            mock_chunk.name = "hello"
            mock_chunk.start_line = 1
            mock_chunk.end_line = 1
            mock_chunk.language = "python"
            mock_registry.chunk_file.return_value = [mock_chunk]

            result = runner.invoke(cli, ["index", "test.py"])

            assert result.exit_code == 0 or "Indexed" in result.output

    def test_index_ignores_recursive_excludes(self, runner, tmp_path):
        """Test that directory exclusion patterns starting with **/ work correctly."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create structure
            # .venv/lib/foo.py (should be excluded by **/.venv/**)
            venv_dir = Path.cwd() / ".venv" / "lib"
            venv_dir.mkdir(parents=True)
            (venv_dir / "foo.py").write_text("pass")

            # src/main.py (should be included)
            src_dir = Path.cwd() / "src"
            src_dir.mkdir()
            (src_dir / "main.py").write_text("pass")

            # Mock config with problematic pattern
            with (
                patch("nexus_dev.cli.NexusConfig") as mock_config_cls,
                patch("nexus_dev.cli.create_embedder") as mock_embedder_fn,
                patch("nexus_dev.cli.NexusDatabase") as mock_db_cls,
            ):
                mock_config = MagicMock()
                mock_config.project_id = "test"
                mock_config.include_patterns = ["**/*.py"]
                # The pattern that was failing for root directories
                mock_config.exclude_patterns = ["**/.venv/**"]
                mock_config_cls.load.return_value = mock_config

                # Mock dependencies
                mock_embedder = MagicMock()
                mock_embedder.embed_batch = AsyncMock(return_value=[])  # Empty list is fine
                mock_embedder_fn.return_value = mock_embedder

                mock_db = MagicMock()
                mock_db.delete_by_file = AsyncMock()
                mock_db_cls.return_value = mock_db

                # Run index command recursively on current dir
                result = runner.invoke(cli, ["index", ".", "-r"])

                assert result.exit_code == 0

                # Check output to see what was indexed
                # valid file should be indexed
                assert "src/main.py" in result.output or "main.py" in result.output
                # venv file should NOT be indexed
                assert ".venv/lib/foo.py" not in result.output and "foo.py" not in result.output


class TestCliReindex:
    """Test suite for nexus-reindex command."""

    def test_reindex_no_config(self, runner, tmp_path):
        """Test reindex fails gracefully without config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["reindex", "--yes"])

            assert "nexus_config.json not found" in result.output


class TestCliIndexLesson:
    """Test suite for nexus-index-lesson command."""

    def test_index_lesson_file_not_found(self, runner, tmp_path):
        """Test index-lesson with nonexistent file."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["index-lesson", "nonexistent.md"])

            assert "not found" in result.output


class TestCliVersion:
    """Test CLI version option."""

    def test_version(self, runner):
        """Test --version option."""
        result = runner.invoke(cli, ["--version"])

        assert "0.1.0" in result.output


class TestCliIndexMCP:
    """Test suite for nexus-index-mcp command."""

    def test_index_mcp_no_mcp_config(self, runner, tmp_path):
        """Test index-mcp fails when MCP config is not found."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["index-mcp", "--all"])

            assert "MCP config not found" in result.output

    def test_index_mcp_missing_options(self, runner, tmp_path):
        """Test index-mcp fails when neither --server nor --all is specified."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create MCP config
            mcp_config_dir = Path.home() / ".config" / "mcp"
            mcp_config_dir.mkdir(parents=True, exist_ok=True)
            mcp_config_path = mcp_config_dir / "config.json"
            mcp_config_path.write_text('{"mcpServers": {}}')

            result = runner.invoke(cli, ["index-mcp"])

            assert "Specify --server or --all" in result.output

            # Cleanup
            mcp_config_path.unlink()

    def test_index_mcp_invalid_json(self, runner, tmp_path):
        """Test index-mcp fails gracefully with invalid JSON in config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create invalid MCP config
            mcp_config_dir = Path.home() / ".config" / "mcp"
            mcp_config_dir.mkdir(parents=True, exist_ok=True)
            mcp_config_path = mcp_config_dir / "config.json"
            mcp_config_path.write_text("{invalid json}")

            result = runner.invoke(cli, ["index-mcp", "--all"])

            assert "Invalid JSON in MCP config" in result.output

            # Cleanup
            mcp_config_path.unlink()

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.MCPClientManager")
    def test_index_mcp_single_server_success(
        self,
        mock_client_cls,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test indexing a single server successfully."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create nexus config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create MCP config
            mcp_config_dir = Path.home() / ".config" / "mcp"
            mcp_config_dir.mkdir(parents=True, exist_ok=True)
            mcp_config_path = mcp_config_dir / "config.json"
            mcp_config = {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_TOKEN": "test"},
                    }
                }
            }
            mcp_config_path.write_text(json.dumps(mcp_config))

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config_cls.load.return_value = mock_config

            # Mock embedder
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_embedder_fn.return_value = mock_embedder

            # Mock database
            mock_db = MagicMock()
            mock_db.upsert_document = AsyncMock()
            mock_db_cls.return_value = mock_db

            # Mock MCP client
            mock_client = MagicMock()
            mock_tool1 = MagicMock()
            mock_tool1.name = "create_issue"
            mock_tool1.description = "Create a GitHub issue"
            mock_tool1.input_schema = {"type": "object"}
            mock_client.get_tools = AsyncMock(return_value=[mock_tool1])
            mock_client_cls.return_value = mock_client

            result = runner.invoke(cli, ["index-mcp", "--server", "github"])

            assert result.exit_code == 0
            assert "Indexing tools from: github" in result.output
            assert "Found 1 tools" in result.output
            assert "✅ Indexed 1 tools from github" in result.output
            assert "Done!" in result.output

            # Cleanup
            mcp_config_path.unlink()

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.MCPClientManager")
    def test_index_mcp_all_servers(
        self,
        mock_client_cls,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test indexing all servers."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create nexus config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create MCP config with multiple servers
            mcp_config_dir = Path.home() / ".config" / "mcp"
            mcp_config_dir.mkdir(parents=True, exist_ok=True)
            mcp_config_path = mcp_config_dir / "config.json"
            mcp_config = {
                "mcpServers": {
                    "github": {"command": "npx", "args": ["github-server"]},
                    "gitlab": {"command": "npx", "args": ["gitlab-server"]},
                }
            }
            mcp_config_path.write_text(json.dumps(mcp_config))

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config_cls.load.return_value = mock_config

            # Mock embedder
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_embedder_fn.return_value = mock_embedder

            # Mock database
            mock_db = MagicMock()
            mock_db.upsert_document = AsyncMock()
            mock_db_cls.return_value = mock_db

            # Mock MCP client
            mock_client = MagicMock()
            mock_tool1 = MagicMock()
            mock_tool1.name = "tool1"
            mock_tool1.description = "Test tool"
            mock_tool1.input_schema = {}
            mock_client.get_tools = AsyncMock(return_value=[mock_tool1])
            mock_client_cls.return_value = mock_client

            result = runner.invoke(cli, ["index-mcp", "--all"])

            assert result.exit_code == 0
            assert "Indexing tools from: github" in result.output
            assert "Indexing tools from: gitlab" in result.output
            assert "Done!" in result.output

            # Cleanup
            mcp_config_path.unlink()

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.MCPClientManager")
    def test_index_mcp_custom_config_path(
        self,
        mock_client_cls,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test indexing with custom MCP config path."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create nexus config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create custom MCP config
            custom_mcp_config = Path.cwd() / "custom_mcp.json"
            mcp_config = {"mcpServers": {"test": {"command": "test", "args": []}}}
            custom_mcp_config.write_text(json.dumps(mcp_config))

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config_cls.load.return_value = mock_config

            # Mock embedder
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_embedder_fn.return_value = mock_embedder

            # Mock database
            mock_db = MagicMock()
            mock_db.upsert_document = AsyncMock()
            mock_db_cls.return_value = mock_db

            # Mock MCP client
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            result = runner.invoke(cli, ["index-mcp", "--config", str(custom_mcp_config), "--all"])

            assert result.exit_code == 0
            assert "Indexing tools from: test" in result.output

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.MCPClientManager")
    def test_index_mcp_server_not_found(
        self,
        mock_client_cls,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test indexing with nonexistent server name."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create nexus config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create MCP config
            mcp_config_dir = Path.home() / ".config" / "mcp"
            mcp_config_dir.mkdir(parents=True, exist_ok=True)
            mcp_config_path = mcp_config_dir / "config.json"
            mcp_config = {"mcpServers": {"github": {"command": "npx", "args": []}}}
            mcp_config_path.write_text(json.dumps(mcp_config))

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config_cls.load.return_value = mock_config

            result = runner.invoke(cli, ["index-mcp", "--server", "nonexistent"])

            assert result.exit_code == 0
            assert "Server not found: nonexistent" in result.output

            # Cleanup
            mcp_config_path.unlink()

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.MCPClientManager")
    def test_index_mcp_connection_error(
        self,
        mock_client_cls,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test graceful error handling on connection failure."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create nexus config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create MCP config
            mcp_config_dir = Path.home() / ".config" / "mcp"
            mcp_config_dir.mkdir(parents=True, exist_ok=True)
            mcp_config_path = mcp_config_dir / "config.json"
            mcp_config = {"mcpServers": {"github": {"command": "npx", "args": []}}}
            mcp_config_path.write_text(json.dumps(mcp_config))

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config_cls.load.return_value = mock_config

            # Mock embedder
            mock_embedder = MagicMock()
            mock_embedder_fn.return_value = mock_embedder

            # Mock database
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db

            # Mock MCP client to raise exception
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client_cls.return_value = mock_client

            result = runner.invoke(cli, ["index-mcp", "--server", "github"])

            assert result.exit_code == 0
            assert "❌ Failed to index github: Connection failed" in result.output
            assert "Done!" in result.output

            # Cleanup
            mcp_config_path.unlink()

    def test_index_mcp_no_nexus_config(self, runner, tmp_path):
        """Test index-mcp fails when nexus_config.json is not found."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create MCP config
            mcp_config_dir = Path.home() / ".config" / "mcp"
            mcp_config_dir.mkdir(parents=True, exist_ok=True)
            mcp_config_path = mcp_config_dir / "config.json"
            mcp_config = {"mcpServers": {"test": {"command": "test", "args": []}}}
            mcp_config_path.write_text(json.dumps(mcp_config))

            result = runner.invoke(cli, ["index-mcp", "--server", "test"])

            assert "nexus_config.json not found" in result.output

            # Cleanup
            mcp_config_path.unlink()
