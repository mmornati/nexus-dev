"""Tests for database module."""

from nexus_dev.database import (
    Document,
    DocumentType,
    SearchResult,
    generate_document_id,
)


class TestDocument:
    """Test suite for Document dataclass."""

    def test_to_dict(self):
        """Test converting document to dictionary."""
        doc = Document(
            id="test-id",
            text="Sample text",
            vector=[0.1, 0.2, 0.3],
            project_id="proj-123",
            file_path="/path/to/file.py",
            doc_type=DocumentType.CODE,
            chunk_type="function",
            language="python",
            name="my_function",
            start_line=10,
            end_line=20,
        )

        result = doc.to_dict()

        assert result["id"] == "test-id"
        assert result["text"] == "Sample text"
        assert result["vector"] == [0.1, 0.2, 0.3]
        assert result["project_id"] == "proj-123"
        assert result["doc_type"] == "code"
        assert result["chunk_type"] == "function"
        assert result["language"] == "python"
        assert "timestamp" in result

    def test_document_type_enum(self):
        """Test DocumentType enum values."""
        assert DocumentType.CODE.value == "code"
        assert DocumentType.LESSON.value == "lesson"
        assert DocumentType.DOCUMENTATION.value == "documentation"
        assert DocumentType.TOOL.value == "tool"

    def test_document_default_values(self):
        """Test document default values."""
        doc = Document(
            id="test",
            text="text",
            vector=[0.1],
            project_id="proj",
            file_path="/file",
            doc_type=DocumentType.CODE,
        )

        assert doc.chunk_type == "module"
        assert doc.language == "unknown"
        assert doc.name == ""
        assert doc.start_line == 0
        assert doc.end_line == 0
        assert doc.timestamp is not None
        assert doc.server_name == ""
        assert doc.parameters_schema == ""

    def test_tool_document_type(self):
        """Test creating a TOOL document type with tool-specific fields."""
        doc = Document(
            id="tool-1",
            text="List files in directory",
            vector=[0.1, 0.2, 0.3],
            project_id="mcp-project",
            file_path="mcp://filesystem/list_directory",
            doc_type=DocumentType.TOOL,
            server_name="filesystem",
            parameters_schema='{"type": "object", "properties": {"path": {"type": "string"}}}',
        )

        assert doc.doc_type == DocumentType.TOOL
        assert doc.server_name == "filesystem"
        expected_schema = '{"type": "object", "properties": {"path": {"type": "string"}}}'
        assert doc.parameters_schema == expected_schema

    def test_tool_document_to_dict(self):
        """Test converting TOOL document to dictionary."""
        doc = Document(
            id="tool-2",
            text="Search code",
            vector=[0.4, 0.5, 0.6],
            project_id="mcp-project",
            file_path="mcp://code-search/search",
            doc_type=DocumentType.TOOL,
            server_name="code-search",
            parameters_schema='{"type": "object", "required": ["query"]}',
        )

        result = doc.to_dict()

        assert result["doc_type"] == "tool"
        assert result["server_name"] == "code-search"
        assert result["parameters_schema"] == '{"type": "object", "required": ["query"]}'
        assert "timestamp" in result


class TestGenerateDocumentId:
    """Test suite for document ID generation."""

    def test_deterministic(self):
        """Test that same inputs produce same ID."""
        id1 = generate_document_id("proj", "file.py", "func", 10)
        id2 = generate_document_id("proj", "file.py", "func", 10)

        assert id1 == id2

    def test_different_inputs_different_ids(self):
        """Test that different inputs produce different IDs."""
        id1 = generate_document_id("proj", "file.py", "func", 10)
        id2 = generate_document_id("proj", "file.py", "func", 11)
        id3 = generate_document_id("proj", "file.py", "other", 10)
        id4 = generate_document_id("proj", "other.py", "func", 10)
        id5 = generate_document_id("other", "file.py", "func", 10)

        ids = [id1, id2, id3, id4, id5]
        assert len(set(ids)) == 5  # All unique

    def test_valid_uuid_format(self):
        """Test that generated ID is valid UUID format."""
        doc_id = generate_document_id("proj", "file.py", "func", 1)

        # UUID format: 8-4-4-4-12
        parts = doc_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_handles_special_characters(self):
        """Test ID generation with special characters."""
        doc_id = generate_document_id(
            "proj with spaces",
            "/path/to/file<>.py",
            "func:name",
            0,
        )

        assert len(doc_id) == 36


class TestSearchResult:
    """Test suite for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            id="result-id",
            text="Found text",
            score=0.95,
            project_id="proj-123",
            file_path="/path/to/file.py",
            doc_type="code",
            chunk_type="function",
            language="python",
            name="found_function",
            start_line=5,
            end_line=15,
        )

        assert result.id == "result-id"
        assert result.text == "Found text"
        assert result.score == 0.95
        assert result.doc_type == "code"
