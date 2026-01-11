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
                input="allow-lessons\n",
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            assert "Created nexus_config.json" in result.output
            assert (Path.cwd() / "nexus_config.json").exists()

    def test_init_creates_lessons_directory(self, runner, tmp_path):
        """Test init command creates .nexus/lessons directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli, ["init", "--project-name", "test", "--no-hook"], input="allow-lessons\n"
            )

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
                input="allow-lessons\n",
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
                input="allow-lessons\n",
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
                input="n\n",  # Decline overwrite (would show gitignore choice if not aborted)
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

            # Use -q to skip confirmation
            result = runner.invoke(cli, ["index", "test.py", "-q"])

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
                result = runner.invoke(cli, ["index", ".", "-r"], input="y\n")

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
            result = runner.invoke(cli, ["reindex"])

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

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.MCPClientManager")
    def test_index_mcp_project_config_success(
        self,
        mock_client_cls,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test indexing from project-specific .nexus/mcp_config.json."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create nexus config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create project-specific MCP config
            mcp_dir = Path.cwd() / ".nexus"
            mcp_dir.mkdir()
            mcp_config_path = mcp_dir / "mcp_config.json"
            mcp_config = {
                "version": "1.0",
                "servers": {
                    "project-server": {
                        "command": "test-cmd",
                        "args": ["arg1"],
                        "env": {},
                        "enabled": True,
                    }
                },
                "profiles": {"default": ["project-server"]},
                "active_profile": "default",
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
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            result = runner.invoke(cli, ["index-mcp", "--all"])

            assert result.exit_code == 0
            assert "Indexing tools from: project-server" in result.output

    @patch("nexus_dev.cli.create_embedder")
    @patch("nexus_dev.cli.NexusDatabase")
    @patch("nexus_dev.cli.NexusConfig")
    @patch("nexus_dev.cli.MCPClientManager")
    def test_index_mcp_priority(
        self,
        mock_client_cls,
        mock_config_cls,
        mock_db_cls,
        mock_embedder_fn,
        runner,
        tmp_path,
    ):
        """Test that project config takes priority over global config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create nexus config
            (Path.cwd() / "nexus_config.json").write_text("{}")

            # Create global config
            global_config_dir = Path.home() / ".config" / "mcp"
            global_config_dir.mkdir(parents=True, exist_ok=True)
            global_config_path = global_config_dir / "config.json"
            global_config_path.write_text('{"mcpServers": {"global-server": {"command": "cmd"}}}')

            # Create project config
            mcp_dir = Path.cwd() / ".nexus"
            mcp_dir.mkdir()
            mcp_config_path = mcp_dir / "mcp_config.json"
            mcp_config = {
                "version": "1.0",
                "servers": {"project-server": {"command": "cmd", "enabled": True}},
            }
            mcp_config_path.write_text(json.dumps(mcp_config))

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config_cls.load.return_value = mock_config

            # Mock dependencies
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_fn.return_value = mock_embedder

            result = runner.invoke(cli, ["index-mcp", "--all"])

            assert result.exit_code == 0
            assert "Indexing tools from: project-server" in result.output
            assert "global-server" not in result.output

            # Cleanup global config
            global_config_path.unlink()

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


class TestCliMCPInit:
    """Test suite for nexus-mcp init command."""

    def test_mcp_init_creates_empty_config(self, runner, tmp_path):
        """Test mcp init creates an empty configuration."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["mcp", "init"])

            assert result.exit_code == 0
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            assert config_path.exists()

            # Verify config content
            config = json.loads(config_path.read_text())
            assert config["version"] == "1.0"
            assert config["servers"] == {}
            assert "Created" in result.output
            assert "Configuration initialized successfully!" in result.output

    def test_mcp_init_from_global_success(self, runner, tmp_path):
        """Test mcp init --from-global imports servers from global config."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create global config
            global_config_dir = Path.home() / ".config" / "mcp"
            global_config_dir.mkdir(parents=True, exist_ok=True)
            global_config_path = global_config_dir / "config.json"
            global_config = {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_TOKEN": "test"},
                    },
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                    },
                }
            }
            global_config_path.write_text(json.dumps(global_config))

            result = runner.invoke(cli, ["mcp", "init", "--from-global"])

            assert result.exit_code == 0
            assert "Imported 2 servers from global config" in result.output

            # Verify created config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            assert config_path.exists()

            config = json.loads(config_path.read_text())
            assert config["version"] == "1.0"
            assert "github" in config["servers"]
            assert "filesystem" in config["servers"]
            assert config["servers"]["github"]["command"] == "npx"
            assert config["servers"]["github"]["args"] == [
                "-y",
                "@modelcontextprotocol/server-github",
            ]
            assert config["servers"]["github"]["env"] == {"GITHUB_TOKEN": "test"}
            assert config["servers"]["github"]["enabled"] is True

            # Cleanup
            global_config_path.unlink()

    def test_mcp_init_from_global_not_found(self, runner, tmp_path):
        """Test mcp init --from-global when global config doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Ensure no global config exists
            global_config_path = Path.home() / ".config" / "mcp" / "config.json"
            if global_config_path.exists():
                global_config_path.unlink()

            result = runner.invoke(cli, ["mcp", "init", "--from-global"])

            assert "Global config not found" in result.output
            # Config should not be created
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            assert not config_path.exists()

    def test_mcp_init_existing_config_abort(self, runner, tmp_path):
        """Test mcp init aborts if config exists and user declines overwrite."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"version": "1.0", "servers": {}}')

            result = runner.invoke(
                cli,
                ["mcp", "init"],
                input="n\n",  # Decline overwrite
            )

            assert "MCP config exists" in result.output
            assert "Aborted" in result.output

    def test_mcp_init_existing_config_overwrite(self, runner, tmp_path):
        """Test mcp init overwrites config when user confirms."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"version": "1.0", "servers": {"old": {}}}')

            result = runner.invoke(
                cli,
                ["mcp", "init"],
                input="y\n",  # Confirm overwrite
            )

            assert result.exit_code == 0
            assert "Created" in result.output

            # Verify old server is gone
            config = json.loads(config_path.read_text())
            assert "old" not in config["servers"]
            assert config["servers"] == {}

    def test_mcp_init_creates_nexus_directory(self, runner, tmp_path):
        """Test mcp init creates .nexus directory if it doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            nexus_dir = Path.cwd() / ".nexus"
            assert not nexus_dir.exists()

            result = runner.invoke(cli, ["mcp", "init"])

            assert result.exit_code == 0
            assert nexus_dir.exists()
            assert nexus_dir.is_dir()

    def test_mcp_init_from_global_invalid_json(self, runner, tmp_path):
        """Test mcp init --from-global handles invalid JSON gracefully."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create global config with invalid JSON
            global_config_dir = Path.home() / ".config" / "mcp"
            global_config_dir.mkdir(parents=True, exist_ok=True)
            global_config_path = global_config_dir / "config.json"
            global_config_path.write_text("{invalid json}")

            result = runner.invoke(cli, ["mcp", "init", "--from-global"])

            assert "Invalid JSON in global config" in result.output

            # Cleanup
            global_config_path.unlink()


class TestCliMCPAdd:
    """Test suite for nexus-mcp add command."""

    def test_mcp_add_simple_server(self, runner, tmp_path):
        """Test adding a simple server without args or env."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create initial config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"version": "1.0", "servers": {}, "profiles": {}}')

            result = runner.invoke(
                cli,
                ["mcp", "add", "test-server", "--command", "test-cmd"],
            )

            assert result.exit_code == 0
            assert "Added test-server to profile 'default'" in result.output

            # Verify config
            config = json.loads(config_path.read_text())
            assert "test-server" in config["servers"]
            assert config["servers"]["test-server"]["command"] == "test-cmd"
            assert config["servers"]["test-server"]["args"] == []
            assert config["servers"]["test-server"]["env"] == {}
            assert config["servers"]["test-server"]["enabled"] is True

            # Verify profile
            assert "default" in config["profiles"]
            assert "test-server" in config["profiles"]["default"]

    def test_mcp_add_with_args(self, runner, tmp_path):
        """Test adding a server with arguments."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create initial config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"version": "1.0", "servers": {}, "profiles": {}}')

            result = runner.invoke(
                cli,
                [
                    "mcp",
                    "add",
                    "github",
                    "--command",
                    "npx",
                    "--args",
                    "-y",
                    "--args",
                    "@modelcontextprotocol/server-github",
                ],
            )

            assert result.exit_code == 0
            assert "Added github to profile 'default'" in result.output

            # Verify config
            config = json.loads(config_path.read_text())
            assert config["servers"]["github"]["command"] == "npx"
            assert config["servers"]["github"]["args"] == [
                "-y",
                "@modelcontextprotocol/server-github",
            ]

    def test_mcp_add_with_env(self, runner, tmp_path):
        """Test adding a server with environment variables."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create initial config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"version": "1.0", "servers": {}, "profiles": {}}')

            result = runner.invoke(
                cli,
                [
                    "mcp",
                    "add",
                    "myserver",
                    "--command",
                    "my-mcp",
                    "--env",
                    "API_KEY=${MY_API_KEY}",
                    "--env",
                    "DEBUG=true",
                ],
            )

            assert result.exit_code == 0

            # Verify config
            config = json.loads(config_path.read_text())
            assert config["servers"]["myserver"]["env"] == {
                "API_KEY": "${MY_API_KEY}",
                "DEBUG": "true",
            }

    def test_mcp_add_to_custom_profile(self, runner, tmp_path):
        """Test adding a server to a custom profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create initial config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"version": "1.0", "servers": {}, "profiles": {}}')

            result = runner.invoke(
                cli,
                [
                    "mcp",
                    "add",
                    "test-server",
                    "--command",
                    "test-cmd",
                    "--profile",
                    "dev",
                ],
            )

            assert result.exit_code == 0
            assert "Added test-server to profile 'dev'" in result.output

            # Verify profile
            config = json.loads(config_path.read_text())
            assert "dev" in config["profiles"]
            assert "test-server" in config["profiles"]["dev"]

    def test_mcp_add_to_existing_profile(self, runner, tmp_path):
        """Test adding a server to an existing profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create initial config with existing profile
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            initial_config = {
                "version": "1.0",
                "servers": {"server1": {"command": "cmd1", "args": [], "env": {}, "enabled": True}},
                "profiles": {"default": ["server1"]},
            }
            config_path.write_text(json.dumps(initial_config))

            result = runner.invoke(
                cli,
                ["mcp", "add", "server2", "--command", "cmd2"],
            )

            assert result.exit_code == 0

            # Verify both servers in profile
            config = json.loads(config_path.read_text())
            assert "server1" in config["profiles"]["default"]
            assert "server2" in config["profiles"]["default"]

    def test_mcp_add_duplicate_server_in_profile(self, runner, tmp_path):
        """Test adding a server that's already in the profile doesn't duplicate."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create initial config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            initial_config = {
                "version": "1.0",
                "servers": {"server1": {"command": "cmd1", "args": [], "env": {}, "enabled": True}},
                "profiles": {"default": ["server1"]},
            }
            config_path.write_text(json.dumps(initial_config))

            # Add same server again (should update server config but not duplicate in profile)
            result = runner.invoke(
                cli,
                ["mcp", "add", "server1", "--command", "new-cmd"],
            )

            assert result.exit_code == 0

            # Verify server only appears once in profile
            config = json.loads(config_path.read_text())
            assert config["profiles"]["default"].count("server1") == 1
            # Verify server command was updated
            assert config["servers"]["server1"]["command"] == "new-cmd"

    def test_mcp_add_no_config(self, runner, tmp_path):
        """Test add command fails gracefully when config doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["mcp", "add", "test", "--command", "cmd"],
            )

            assert "Run 'nexus-mcp init' first" in result.output

    def test_mcp_add_complex_example(self, runner, tmp_path):
        """Test adding a complex server with all options."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create initial config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"version": "1.0", "servers": {}, "profiles": {}}')

            result = runner.invoke(
                cli,
                [
                    "mcp",
                    "add",
                    "github",
                    "--command",
                    "npx",
                    "--args",
                    "-y",
                    "--args",
                    "@modelcontextprotocol/server-github",
                    "--env",
                    "GITHUB_TOKEN=${GITHUB_TOKEN}",
                    "--env",
                    "DEBUG=false",
                    "--profile",
                    "production",
                ],
            )

            assert result.exit_code == 0
            assert "Added github to profile 'production'" in result.output

            # Verify full config
            config = json.loads(config_path.read_text())
            assert config["servers"]["github"]["command"] == "npx"
            assert config["servers"]["github"]["args"] == [
                "-y",
                "@modelcontextprotocol/server-github",
            ]
            assert config["servers"]["github"]["env"] == {
                "GITHUB_TOKEN": "${GITHUB_TOKEN}",
                "DEBUG": "false",
            }
            assert config["servers"]["github"]["enabled"] is True
            assert "production" in config["profiles"]
            assert "github" in config["profiles"]["production"]


class TestCliMCPList:
    """Test suite for nexus-mcp list command."""

    def test_mcp_list_no_config(self, runner, tmp_path):
        """Test mcp list shows error when config doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["mcp", "list"])

            assert "No MCP config" in result.output
            assert "Run 'nexus-mcp init' first" in result.output

    def test_mcp_list_active_profile_servers(self, runner, tmp_path):
        """Test mcp list shows servers from active profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with servers and profiles
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_TOKEN": "test"},
                        "enabled": True,
                    },
                    "gitlab": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-gitlab"],
                        "enabled": True,
                    },
                },
                "profiles": {
                    "default": ["github"],
                    "all": ["github", "gitlab"],
                },
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "list"])

            assert result.exit_code == 0
            assert "Active profile: default" in result.output
            assert "Active servers:" in result.output
            assert "✓ github" in result.output
            assert "Command: npx -y @modelcontextprotocol/server-github" in result.output
            assert "Env: GITHUB_TOKEN" in result.output
            # gitlab should NOT be in the output (not in active profile)
            assert "gitlab" not in result.output
            assert (
                "Profiles: default, all" in result.output
                or "Profiles: all, default" in result.output
            )

    def test_mcp_list_all_servers(self, runner, tmp_path):
        """Test mcp list --all shows all servers."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "github-server"],
                        "enabled": True,
                    },
                    "gitlab": {
                        "command": "npx",
                        "args": ["-y", "gitlab-server"],
                        "enabled": False,
                    },
                },
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "list", "--all"])

            assert result.exit_code == 0
            assert "All servers:" in result.output
            assert "✓ github" in result.output
            assert "✗ gitlab" in result.output  # Should show disabled server with ✗
            assert "Command: npx -y github-server" in result.output
            assert "Command: npx -y gitlab-server" in result.output

    def test_mcp_list_shows_disabled_status(self, runner, tmp_path):
        """Test mcp list shows enabled/disabled status correctly."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with disabled server
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "enabled-server": {"command": "cmd1", "enabled": True},
                    "disabled-server": {"command": "cmd2", "enabled": False},
                },
                "profiles": {"default": ["enabled-server", "disabled-server"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "list"])

            assert result.exit_code == 0
            # Only enabled server should show in active list (disabled are filtered)
            assert "✓ enabled-server" in result.output
            # Disabled server should not appear (filtered out by get_active_servers logic)
            assert "disabled-server" not in result.output

    def test_mcp_list_no_env_vars(self, runner, tmp_path):
        """Test mcp list doesn't show Env line when no env vars."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config without env vars
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"simple": {"command": "simple-cmd", "args": [], "enabled": True}},
                "profiles": {"default": ["simple"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "list"])

            assert result.exit_code == 0
            assert "✓ simple" in result.output
            assert "Command: simple-cmd" in result.output
            # Should NOT show "Env:" line
            assert "Env:" not in result.output

    def test_mcp_list_empty_servers(self, runner, tmp_path):
        """Test mcp list with no servers."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create empty config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {"version": "1.0", "servers": {}, "profiles": {}, "active_profile": "default"}
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "list"])

            assert result.exit_code == 0
            assert "Active profile: default" in result.output
            assert "Active servers:" in result.output
            # No servers to show
            assert "✓" not in result.output
            assert "✗" not in result.output


class TestCliMCPProfile:
    """Test suite for nexus-mcp profile command."""

    def test_mcp_profile_no_config(self, runner, tmp_path):
        """Test profile command fails when config doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["mcp", "profile"])

            assert "Run 'nexus-mcp init' first" in result.output

    def test_mcp_profile_show_current(self, runner, tmp_path):
        """Test showing current active profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with profiles
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": ["github"], "dev": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile"])

            assert result.exit_code == 0
            assert "Active: default" in result.output
            assert "Servers: github" in result.output

    def test_mcp_profile_show_current_empty(self, runner, tmp_path):
        """Test showing current profile with no servers."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with empty profile
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile"])

            assert result.exit_code == 0
            assert "Active: default" in result.output
            assert "Servers: (none)" in result.output

    def test_mcp_profile_switch(self, runner, tmp_path):
        """Test switching to a different profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with multiple profiles
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": ["github"], "dev": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "dev"])

            assert result.exit_code == 0
            assert "Switched to profile: dev" in result.output

            # Verify config was updated
            updated_config = json.loads(config_path.read_text())
            assert updated_config["active_profile"] == "dev"

    def test_mcp_profile_switch_nonexistent(self, runner, tmp_path):
        """Test switching to a nonexistent profile fails."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "nonexistent"])

            assert result.exit_code == 0
            assert "Profile 'nonexistent' not found" in result.output

            # Verify active profile wasn't changed
            updated_config = json.loads(config_path.read_text())
            assert updated_config["active_profile"] == "default"

    def test_mcp_profile_create(self, runner, tmp_path):
        """Test creating a new profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "dev", "--create"])

            assert result.exit_code == 0
            assert "Created profile: dev" in result.output

            # Verify profile was created
            updated_config = json.loads(config_path.read_text())
            assert "dev" in updated_config["profiles"]
            assert updated_config["profiles"]["dev"] == []

    def test_mcp_profile_create_existing(self, runner, tmp_path):
        """Test creating a profile that already exists fails."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with existing profile
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {},
                "profiles": {"default": [], "dev": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "dev", "--create"])

            assert result.exit_code == 0
            assert "Profile 'dev' exists" in result.output

    def test_mcp_profile_add_server(self, runner, tmp_path):
        """Test adding a server to a profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "default", "--add", "github"])

            assert result.exit_code == 0
            assert "Added github to default" in result.output

            # Verify server was added
            updated_config = json.loads(config_path.read_text())
            assert "github" in updated_config["profiles"]["default"]

    def test_mcp_profile_add_multiple_servers(self, runner, tmp_path):
        """Test adding multiple servers to a profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "github": {"command": "npx", "enabled": True},
                    "gitlab": {"command": "npx", "enabled": True},
                },
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(
                cli, ["mcp", "profile", "default", "--add", "github", "--add", "gitlab"]
            )

            assert result.exit_code == 0
            assert "Added github to default" in result.output
            assert "Added gitlab to default" in result.output

            # Verify servers were added
            updated_config = json.loads(config_path.read_text())
            assert "github" in updated_config["profiles"]["default"]
            assert "gitlab" in updated_config["profiles"]["default"]

    def test_mcp_profile_add_duplicate_server(self, runner, tmp_path):
        """Test adding a server that's already in the profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with server already in profile
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "default", "--add", "github"])

            assert result.exit_code == 0
            # Should not add duplicate
            assert "Added github to default" not in result.output
            assert "Server github already in default" in result.output

            # Verify no duplicate
            updated_config = json.loads(config_path.read_text())
            assert updated_config["profiles"]["default"].count("github") == 1

    def test_mcp_profile_add_nonexistent_server_warning(self, runner, tmp_path):
        """Test adding a server that doesn't exist yet shows warning."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config without the server defined
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "default", "--add", "nonexistent"])

            assert result.exit_code == 0
            assert "Added nonexistent to default" in result.output
            assert "⚠️  Server 'nonexistent' not defined" in result.output
            assert "Add it with 'nexus-mcp add'" in result.output

            # Verify server was still added to profile
            updated_config = json.loads(config_path.read_text())
            assert "nonexistent" in updated_config["profiles"]["default"]

    def test_mcp_profile_remove_server(self, runner, tmp_path):
        """Test removing a server from a profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "default", "--remove", "github"])

            assert result.exit_code == 0
            assert "Removed github from default" in result.output

            # Verify server was removed
            updated_config = json.loads(config_path.read_text())
            assert "github" not in updated_config["profiles"]["default"]

    def test_mcp_profile_remove_multiple_servers(self, runner, tmp_path):
        """Test removing multiple servers from a profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "github": {"command": "npx", "enabled": True},
                    "gitlab": {"command": "npx", "enabled": True},
                },
                "profiles": {"default": ["github", "gitlab"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(
                cli, ["mcp", "profile", "default", "--remove", "github", "--remove", "gitlab"]
            )

            assert result.exit_code == 0
            assert "Removed github from default" in result.output
            assert "Removed gitlab from default" in result.output

            # Verify servers were removed
            updated_config = json.loads(config_path.read_text())
            assert "github" not in updated_config["profiles"]["default"]
            assert "gitlab" not in updated_config["profiles"]["default"]

    def test_mcp_profile_remove_nonexistent_server(self, runner, tmp_path):
        """Test removing a server that's not in the profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "default", "--remove", "github"])

            assert result.exit_code == 0
            # Should not output removal message
            assert "Removed github from default" not in result.output

    def test_mcp_profile_add_and_remove(self, runner, tmp_path):
        """Test adding and removing servers in the same command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "github": {"command": "npx", "enabled": True},
                    "gitlab": {"command": "npx", "enabled": True},
                },
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(
                cli,
                [
                    "mcp",
                    "profile",
                    "default",
                    "--add",
                    "gitlab",
                    "--remove",
                    "github",
                ],
            )

            assert result.exit_code == 0
            assert "Added gitlab to default" in result.output
            assert "Removed github from default" in result.output

            # Verify changes
            updated_config = json.loads(config_path.read_text())
            assert "gitlab" in updated_config["profiles"]["default"]
            assert "github" not in updated_config["profiles"]["default"]

    def test_mcp_profile_operations_dont_switch(self, runner, tmp_path):
        """Test that add/remove operations don't switch active profile."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with multiple profiles
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": [], "dev": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            # Add server to dev profile
            result = runner.invoke(cli, ["mcp", "profile", "dev", "--add", "github"])

            assert result.exit_code == 0
            assert "Added github to dev" in result.output
            # Should not switch
            assert "Switched to profile: dev" not in result.output

            # Verify active profile didn't change
            updated_config = json.loads(config_path.read_text())
            assert updated_config["active_profile"] == "default"

    def test_mcp_profile_create_and_add(self, runner, tmp_path):
        """Test creating a profile and adding a server in the same command."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "profile", "dev", "--create", "--add", "github"])

            assert result.exit_code == 0
            assert "Created profile: dev" in result.output
            assert "Added github to dev" in result.output

            # Verify profile was created and server was added
            updated_config = json.loads(config_path.read_text())
            assert "dev" in updated_config["profiles"]
            assert "github" in updated_config["profiles"]["dev"]


class TestCliMCPEnable:
    """Test suite for nexus-mcp enable command."""

    def test_mcp_enable_success(self, runner, tmp_path):
        """Test enabling a server successfully."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with disabled server
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "enabled": False,
                    }
                },
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "enable", "github"])

            assert result.exit_code == 0
            assert "github: enabled" in result.output

            # Verify server was enabled
            updated_config = json.loads(config_path.read_text())
            assert updated_config["servers"]["github"]["enabled"] is True

    def test_mcp_enable_already_enabled(self, runner, tmp_path):
        """Test enabling a server that's already enabled."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with enabled server
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": True}},
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "enable", "github"])

            assert result.exit_code == 0
            assert "github: enabled" in result.output

            # Verify server is still enabled
            updated_config = json.loads(config_path.read_text())
            assert updated_config["servers"]["github"]["enabled"] is True

    def test_mcp_enable_server_not_found(self, runner, tmp_path):
        """Test enabling a server that doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config without the server
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "enable", "nonexistent"])

            assert result.exit_code == 0
            assert "Server not found: nonexistent" in result.output

    def test_mcp_enable_no_config(self, runner, tmp_path):
        """Test enabling a server when config doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["mcp", "enable", "github"])

            assert "Run 'nexus-mcp init' first" in result.output


class TestCliMCPDisable:
    """Test suite for nexus-mcp disable command."""

    def test_mcp_disable_success(self, runner, tmp_path):
        """Test disabling a server successfully."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with enabled server
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "enabled": True,
                    }
                },
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "disable", "github"])

            assert result.exit_code == 0
            assert "github: disabled" in result.output

            # Verify server was disabled
            updated_config = json.loads(config_path.read_text())
            assert updated_config["servers"]["github"]["enabled"] is False

    def test_mcp_disable_already_disabled(self, runner, tmp_path):
        """Test disabling a server that's already disabled."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config with disabled server
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {"github": {"command": "npx", "enabled": False}},
                "profiles": {"default": ["github"]},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "disable", "github"])

            assert result.exit_code == 0
            assert "github: disabled" in result.output

            # Verify server is still disabled
            updated_config = json.loads(config_path.read_text())
            assert updated_config["servers"]["github"]["enabled"] is False

    def test_mcp_disable_server_not_found(self, runner, tmp_path):
        """Test disabling a server that doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config without the server
            config_path = Path.cwd() / ".nexus" / "mcp_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "version": "1.0",
                "servers": {},
                "profiles": {"default": []},
                "active_profile": "default",
            }
            config_path.write_text(json.dumps(config))

            result = runner.invoke(cli, ["mcp", "disable", "nonexistent"])

            assert result.exit_code == 0
            assert "Server not found: nonexistent" in result.output

    def test_mcp_disable_no_config(self, runner, tmp_path):
        """Test disabling a server when config doesn't exist."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["mcp", "disable", "github"])

            assert "Run 'nexus-mcp init' first" in result.output
