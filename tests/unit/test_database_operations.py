"""Tests for NexusDatabase with mocked LanceDB."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.database import (
    Document,
    DocumentType,
    NexusDatabase,
)


class MockRow:
    """Mock pandas Series row for testing."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        """Get value with default fallback."""
        return self._data.get(key, default)


def create_mock_dataframe(data):
    """Create a mock object that behaves like a pandas DataFrame."""
    mock_df = MagicMock()
    mock_df.__len__ = MagicMock(return_value=len(data))

    # Make it iterable via iterrows
    def iterrows():
        for i, row in enumerate(data):
            yield i, MockRow(row)

    mock_df.iterrows = iterrows

    #  Support __getitem__ for column access and filtering
    def getitem(self, key):
        if isinstance(key, str):
            # Column access: df["column_name"]
            return MagicMock(__eq__=lambda self, val: [row.get(key) == val for row in data])
        elif isinstance(key, list):
            # Boolean indexing: df[mask]
            filtered_data = [row for row, keep in zip(data, key, strict=False) if keep]
            return create_mock_dataframe(filtered_data)
        return mock_df

    mock_df.__getitem__ = lambda key: getitem(mock_df, key)

    # Support groupby for stats
    if data:
        groups = {}
        for row in data:
            doc_type = row.get("doc_type", "unknown")
            groups[doc_type] = groups.get(doc_type, 0) + 1
        mock_series = MagicMock()
        mock_series.to_dict.return_value = groups
        mock_groupby = MagicMock()
        mock_groupby.size.return_value = mock_series
        mock_df.groupby.return_value = mock_groupby

    # Support sort_values and head for lessons - need to return new mock with sorted data
    sorted_data = sorted(data, key=lambda x: x.get("timestamp", ""), reverse=True) if data else []

    def make_sorted_df():
        sorted_mock = MagicMock()
        sorted_mock.__len__ = MagicMock(return_value=len(sorted_data))

        def sorted_iterrows():
            for i, row in enumerate(sorted_data):
                yield i, MockRow(row)

        sorted_mock.iterrows = sorted_iterrows
        sorted_mock.head = MagicMock(return_value=sorted_mock)
        return sorted_mock

    mock_df.sort_values = MagicMock(return_value=make_sorted_df())
    mock_df.head = MagicMock(return_value=mock_df)

    return mock_df


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
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
    return embedder


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="doc-123",
        text="def hello(): pass",
        vector=[0.1] * 1536,
        project_id="test-project",
        file_path="/path/to/file.py",
        doc_type=DocumentType.CODE,
        chunk_type="function",
        language="python",
        name="hello",
        start_line=1,
        end_line=2,
    )


class TestNexusDatabase:
    """Test suite for NexusDatabase class."""

    @patch("nexus_dev.database.lancedb")
    def test_connect_creates_table_if_not_exists(self, mock_lancedb, mock_config, mock_embedder):
        """Test that connect creates table when it doesn't exist."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_table = MagicMock()
        mock_db.create_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        mock_lancedb.connect.assert_called_once()
        mock_db.create_table.assert_called_once()
        assert db._table == mock_table

    @patch("nexus_dev.database.lancedb")
    def test_connect_opens_existing_table(self, mock_lancedb, mock_config, mock_embedder):
        """Test that connect opens existing table."""
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        mock_db.open_table.assert_called_once_with("documents")
        mock_db.create_table.assert_not_called()
        assert db._table == mock_table

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_upsert_document(self, mock_lancedb, mock_config, mock_embedder, sample_document):
        """Test upserting a single document."""
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        result = await db.upsert_document(sample_document)

        assert result == "doc-123"
        mock_table.add.assert_called_once()
        # Check that delete was attempted (for upsert logic)
        mock_table.delete.assert_called()

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_upsert_documents_empty_list(self, mock_lancedb, mock_config, mock_embedder):
        """Test upserting empty list returns empty list."""
        db = NexusDatabase(mock_config, mock_embedder)

        result = await db.upsert_documents([])

        assert result == []

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_upsert_documents_multiple(self, mock_lancedb, mock_config, mock_embedder):
        """Test upserting multiple documents."""
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        docs = [
            Document(
                id=f"doc-{i}",
                text=f"content {i}",
                vector=[0.1] * 1536,
                project_id="test",
                file_path=f"/file{i}.py",
                doc_type=DocumentType.CODE,
            )
            for i in range(3)
        ]

        result = await db.upsert_documents(docs)

        assert result == ["doc-0", "doc-1", "doc-2"]
        mock_table.add.assert_called_once()

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_search_basic(self, mock_lancedb, mock_config, mock_embedder):
        """Test basic semantic search."""
        # Create mock search result DataFrame
        mock_df = create_mock_dataframe(
            [
                {
                    "id": "result-1",
                    "text": "def foo(): pass",
                    "_distance": 0.5,
                    "project_id": "test",
                    "file_path": "/foo.py",
                    "doc_type": "code",
                    "chunk_type": "function",
                    "language": "python",
                    "name": "foo",
                    "start_line": 1,
                    "end_line": 2,
                }
            ]
        )

        mock_search = MagicMock()
        mock_search.limit.return_value = mock_search
        mock_search.to_pandas.return_value = mock_df

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        results = await db.search("find foo function")

        assert len(results) == 1
        assert results[0].id == "result-1"
        assert results[0].name == "foo"
        assert results[0].score == 0.5
        mock_embedder.embed.assert_called_once_with("find foo function")

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_lancedb, mock_config, mock_embedder):
        """Test search with project and doc_type filters."""
        mock_df = create_mock_dataframe([])

        mock_search = MagicMock()
        mock_search.limit.return_value = mock_search
        mock_search.where.return_value = mock_search
        mock_search.to_pandas.return_value = mock_df

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        await db.search(
            "test query",
            project_id="my-project",
            doc_type=DocumentType.CODE,
        )

        # Verify where was called with filters
        mock_search.where.assert_called_once()
        filter_arg = mock_search.where.call_args[0][0]
        assert "project_id = 'my-project'" in filter_arg
        assert "doc_type = 'code'" in filter_arg

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_delete_by_file(self, mock_lancedb, mock_config, mock_embedder):
        """Test deleting documents by file path."""
        mock_df = create_mock_dataframe([{"id": "1"}, {"id": "2"}])

        mock_search = MagicMock()
        mock_search.where.return_value = mock_search
        mock_search.to_pandas.return_value = mock_df

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        count = await db.delete_by_file("/path/to/file.py", "test-project")

        assert count == 2
        mock_table.delete.assert_called_once()

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_delete_by_project(self, mock_lancedb, mock_config, mock_embedder):
        """Test deleting all documents for a project."""
        mock_df = create_mock_dataframe([{"id": "1"}, {"id": "2"}, {"id": "3"}])

        mock_search = MagicMock()
        mock_search.where.return_value = mock_search
        mock_search.to_pandas.return_value = mock_df

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        count = await db.delete_by_project("test-project")

        assert count == 3
        mock_table.delete.assert_called_once()

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_get_project_stats(self, mock_lancedb, mock_config, mock_embedder):
        """Test getting project statistics with filtered data."""
        # Return already-filtered data (as if pandas filtering already happened)
        # We're testing stats calculation, not pandas filtering logic
        all_data = [
            {"doc_type": "code", "project_id": "test-project"},
            {"doc_type": "code", "project_id": "test-project"},
            {"doc_type": "lesson", "project_id": "test-project"},
            {"doc_type": "documentation", "project_id": "test-project"},
            {"doc_type": "code", "project_id": "other-project"},  # Will be filtered out
        ]

        # Mock the dataframe to only return the test-project data after filtering
        filtered_data = [d for d in all_data if d.get("project_id") == "test-project"]
        mock_df = create_mock_dataframe(filtered_data)
        # Make the mock support the full pandas query chain
        mock_df.__getitem__ = MagicMock(return_value=mock_df)  # df["project_id"] returns mock
        mock_df.__eq__ = MagicMock(return_value=[True, True, True, True])  # == comparison

        mock_table = MagicMock()
        mock_table.to_pandas.return_value = mock_df

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        stats = await db.get_project_stats("test-project")

        assert stats["total"] == 4
        assert stats["code"] == 2
        assert stats["lesson"] == 1
        assert stats["documentation"] == 1

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_get_project_stats_empty(self, mock_lancedb, mock_config, mock_embedder):
        """Test getting stats for project with no documents."""
        mock_table = MagicMock()
        mock_table.to_pandas.side_effect = Exception("No data")

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        stats = await db.get_project_stats("empty-project")

        assert stats == {"total": 0}

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_get_recent_lessons(self, mock_lancedb, mock_config, mock_embedder):
        """Test getting recent lessons."""
        mock_df = create_mock_dataframe(
            [
                {
                    "id": "lesson-1",
                    "text": "## Problem\nError 1",
                    "project_id": "test",
                    "file_path": ".nexus/lessons/1.md",
                    "doc_type": "lesson",
                    "chunk_type": "lesson",
                    "language": "markdown",
                    "name": "lesson_1",
                    "start_line": 0,
                    "end_line": 0,
                    "timestamp": "2024-01-01T00:00:00",
                },
                {
                    "id": "lesson-2",
                    "text": "## Problem\nError 2",
                    "project_id": "test",
                    "file_path": ".nexus/lessons/2.md",
                    "doc_type": "lesson",
                    "chunk_type": "lesson",
                    "language": "markdown",
                    "name": "lesson_2",
                    "start_line": 0,
                    "end_line": 0,
                    "timestamp": "2024-01-02T00:00:00",
                },
            ]
        )

        mock_search = MagicMock()
        mock_search.where.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_pandas.return_value = mock_df

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        lessons = await db.get_recent_lessons("test", limit=5)

        assert len(lessons) == 2
        # Should be sorted by timestamp descending
        assert lessons[0].name == "lesson_2"
        assert lessons[1].name == "lesson_1"

    @patch("nexus_dev.database.lancedb")
    @pytest.mark.asyncio
    async def test_get_recent_lessons_empty(self, mock_lancedb, mock_config, mock_embedder):
        """Test getting recent lessons when none exist."""
        mock_search = MagicMock()
        mock_search.where.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_pandas.side_effect = Exception("No lessons")

        mock_table = MagicMock()
        mock_table.search.return_value = mock_search

        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)
        db.connect()

        lessons = await db.get_recent_lessons("test")

        assert lessons == []

    @patch("nexus_dev.database.lancedb")
    def test_ensure_connected_auto_connects(self, mock_lancedb, mock_config, mock_embedder):
        """Test that _ensure_connected auto-connects if not connected."""
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["documents"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        db = NexusDatabase(mock_config, mock_embedder)

        # Not connected yet
        assert db._table is None

        # This should auto-connect
        result = db._ensure_connected()

        assert result == mock_table
        mock_lancedb.connect.assert_called_once()

    def test_get_schema(self, mock_config, mock_embedder):
        """Test that schema is created correctly."""
        db = NexusDatabase(mock_config, mock_embedder)
        schema = db._get_schema()

        field_names = [field.name for field in schema]
        expected_fields = [
            "id",
            "text",
            "vector",
            "project_id",
            "file_path",
            "doc_type",
            "chunk_type",
            "language",
            "name",
            "start_line",
            "end_line",
            "timestamp",
            "server_name",
            "parameters_schema",
        ]

        assert field_names == expected_fields
