"""Tests for database security fixes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.database import (
    Document,
    DocumentType,
    NexusDatabase,
)


@pytest.fixture
def mock_config():
    """Create a mock NexusConfig."""
    config = NexusConfig.create_new("test-project")
    return config


@pytest.fixture
def mock_embedder():
    """Create a mock EmbeddingProvider."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    return embedder


class TestDatabaseSecurity:
    """Test suite for database security (SQL injection prevention)."""

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_upsert_document_escapes_id(self, mock_lancedb, mock_config, mock_embedder):
        """Test that upsert_document escapes document ID."""
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        # Malicious ID
        malicious_id = "doc' OR '1'='1"
        doc = Document(
            id=malicious_id,
            text="content",
            vector=[0.1] * 1536,
            project_id="test",
            file_path="/file.py",
            doc_type=DocumentType.CODE,
        )

        await db.upsert_document(doc)

        # Verify delete was called with escaped ID
        # Expected: id = 'doc'' OR ''1''=''1'
        expected_query = "id = 'doc'' OR ''1''=''1'"
        mock_table.delete.assert_called_with(expected_query)

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_upsert_documents_escapes_id(self, mock_lancedb, mock_config, mock_embedder):
        """Test that upsert_documents escapes document IDs."""
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        # Malicious ID
        malicious_id = "doc' OR '1'='1"
        doc = Document(
            id=malicious_id,
            text="content",
            vector=[0.1] * 1536,
            project_id="test",
            file_path="/file.py",
            doc_type=DocumentType.CODE,
        )

        await db.upsert_documents([doc])

        # Verify delete was called with escaped ID
        expected_query = "id = 'doc'' OR ''1''=''1'"
        mock_table.delete.assert_called_with(expected_query)

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_search_escapes_project_id(self, mock_lancedb, mock_config, mock_embedder):
        """Test that search escapes project_id."""
        mock_search = MagicMock()
        mock_search.limit.return_value = mock_search
        mock_search.to_pandas.return_value = MagicMock(iterrows=lambda: iter([]))

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        malicious_project_id = "proj' OR '1'='1"
        await db.search("query", project_id=malicious_project_id)

        # Verify where was called with escaped project_id
        # Expected: project_id = 'proj'' OR ''1''=''1'
        mock_search.where.assert_called_once()
        args = mock_search.where.call_args[0][0]
        assert "project_id = 'proj'' OR ''1''=''1'" in args

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_delete_by_file_escapes_inputs(self, mock_lancedb, mock_config, mock_embedder):
        """Test that delete_by_file escapes file_path and project_id."""
        mock_search = MagicMock()
        mock_search.where.return_value = mock_search
        # Mock len(to_pandas())
        mock_search.to_pandas.return_value = []

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        malicious_file = "file' OR '1'='1"
        malicious_project = "proj' OR '1'='1"

        await db.delete_by_file(malicious_file, malicious_project)

        # Verify delete was called with escaped inputs
        expected_query = "file_path = 'file'' OR ''1''=''1' AND project_id = 'proj'' OR ''1''=''1'"
        mock_table.delete.assert_called_with(expected_query)

        # Verify search (for count) was also escaped
        mock_search.where.assert_called_with(expected_query)

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_delete_by_project_escapes_project_id(
        self, mock_lancedb, mock_config, mock_embedder
    ):
        """Test that delete_by_project escapes project_id."""
        mock_search = MagicMock()
        mock_search.where.return_value = mock_search
        mock_search.to_pandas.return_value = []

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        malicious_project = "proj' OR '1'='1"

        await db.delete_by_project(malicious_project)

        # Verify delete was called with escaped project_id
        expected_query = "project_id = 'proj'' OR ''1''=''1'"
        mock_table.delete.assert_called_with(expected_query)

        # Verify search (for count) was also escaped
        mock_search.where.assert_called_with(expected_query)

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_get_recent_lessons_escapes_project_id(
        self, mock_lancedb, mock_config, mock_embedder
    ):
        """Test that get_recent_lessons escapes project_id."""
        mock_search = MagicMock()
        mock_search.where.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_pandas.return_value = MagicMock(iterrows=lambda: iter([]))

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        malicious_project = "proj' OR '1'='1"

        await db.get_recent_lessons(malicious_project)

        # Verify where was called with escaped project_id
        mock_search.where.assert_called_once()
        args = mock_search.where.call_args[0][0]
        assert "project_id = 'proj'' OR ''1''=''1'" in args
