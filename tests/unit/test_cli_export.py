"""Tests for CLI export command."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from nexus_dev.cli import cli
from nexus_dev.database import Document, DocumentType


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CliRunner."""
    return CliRunner()


class TestCliExport:
    """Test suite for nexus-export command."""

    @patch("nexus_dev.embeddings.create_embedder")
    @patch("nexus_dev.cli.search.NexusDatabase")
    @patch("nexus_dev.config.NexusConfig")
    def test_export_success(self, mock_config_cls, mock_db_cls, mock_embedder_fn, runner, tmp_path):
        """Test export command retrieves documents correctly."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create config file with ollama to avoid API key check
            (Path.cwd() / "nexus_config.json").write_text(
                '{"project_id": "test", "project_name": "Test", '
                '"embedding_provider": "ollama", "embedding_model": "nomic-embed-text"}'
            )

            # Mock config
            mock_config = MagicMock()
            mock_config.project_id = "test"
            mock_config.embedding_provider = "ollama"
            mock_config.embedding_model = "nomic-embed-text"
            mock_config_cls.load.return_value = mock_config

            # Mock database
            mock_db = MagicMock()

            # Mock list_documents to return some documents
            async def mock_list_documents(*args: Any, **kwargs: Any) -> list[Document]:
                print(f"MOCK CALLED args={args} kwargs={kwargs}")
                return [
                    Document(
                        id="doc1",
                        text="Lesson content",
                        vector=[],
                        project_id="test",
                        file_path="lesson.md",
                        doc_type=DocumentType.LESSON,
                        name="Lesson 1",
                    )
                ]

            mock_db.list_documents = AsyncMock(side_effect=mock_list_documents)
            mock_db_cls.return_value = mock_db

            # Run export
            result = runner.invoke(cli, ["export"])

            if result.exit_code != 0:
                print(result.output)

            assert result.exit_code == 0
            assert "Exporting knowledge for project: test" in result.output
            assert "Found 1 lessons" in result.output

            # Verify file created
            export_dir = Path.cwd() / "nexus-export"
            lesson_dir = export_dir / "lessons"
            assert (lesson_dir / "Lesson1.md").exists()
            assert (lesson_dir / "Lesson1.md").read_text(encoding="utf-8") == "Lesson content"

            # Verify list_documents was called
            assert mock_db.list_documents.call_count >= 3  # called for each type
