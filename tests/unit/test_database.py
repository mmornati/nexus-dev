"""Tests for database module."""

from nexus_dev.database import (
    Document,
    DocumentType,
    SearchResult,
    ToolDocument,
    generate_document_id,
    tool_document_from_schema,
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


class TestToolDocument:
    """Test suite for ToolDocument dataclass."""

    def test_tool_document_creation(self):
        """Test creating a ToolDocument with all fields."""
        doc = ToolDocument(
            id="github:create_pull_request",
            server_name="github",
            tool_name="create_pull_request",
            description="Create a new pull request",
            parameters={"type": "object", "properties": {"title": {"type": "string"}}},
            vector=[0.1, 0.2, 0.3],
            examples=["Create PR for bug fix", "Open new feature PR"],
        )

        assert doc.id == "github:create_pull_request"
        assert doc.server_name == "github"
        assert doc.tool_name == "create_pull_request"
        assert doc.description == "Create a new pull request"
        assert doc.parameters["type"] == "object"
        assert len(doc.vector) == 3
        assert len(doc.examples) == 2
        assert doc.timestamp is not None

    def test_tool_document_default_examples(self):
        """Test ToolDocument with default empty examples."""
        doc = ToolDocument(
            id="filesystem:read_file",
            server_name="filesystem",
            tool_name="read_file",
            description="Read a file",
            parameters={},
            vector=[0.5, 0.6],
        )

        assert doc.examples == []

    def test_tool_document_to_dict(self):
        """Test converting ToolDocument to dictionary."""
        doc = ToolDocument(
            id="slack:send_message",
            server_name="slack",
            tool_name="send_message",
            description="Send a message to Slack",
            parameters={"type": "object", "required": ["channel", "text"]},
            vector=[0.7, 0.8, 0.9],
            examples=["Send notification"],
        )

        result = doc.to_dict()

        assert result["id"] == "slack:send_message"
        assert result["project_id"] == "mcp_tools"
        assert result["file_path"] == "mcp://slack/send_message"
        assert result["doc_type"] == "tool"
        assert result["chunk_type"] == "tool"
        assert result["language"] == "mcp"
        assert result["name"] == "send_message"
        assert result["start_line"] == 0
        assert result["end_line"] == 0
        assert result["server_name"] == "slack"
        assert "timestamp" in result
        assert result["vector"] == [0.7, 0.8, 0.9]

        # Check parameters_schema is JSON string
        import json

        params = json.loads(result["parameters_schema"])
        assert params["type"] == "object"
        assert "channel" in params["required"]

    def test_tool_document_get_searchable_text(self):
        """Test get_searchable_text method."""
        doc = ToolDocument(
            id="database:query",
            server_name="database",
            tool_name="query",
            description="Execute a database query",
            parameters={},
            vector=[0.1],
        )

        text = doc.get_searchable_text()

        assert "MCP Tool: database.query" in text
        assert "Description: Execute a database query" in text

    def test_tool_document_get_searchable_text_with_examples(self):
        """Test get_searchable_text with examples."""
        doc = ToolDocument(
            id="email:send",
            server_name="email",
            tool_name="send",
            description="Send an email",
            parameters={},
            vector=[0.1],
            examples=["Send welcome email", "Send notification"],
        )

        text = doc.get_searchable_text()

        assert "MCP Tool: email.send" in text
        assert "Description: Send an email" in text
        assert "Examples: Send welcome email, Send notification" in text

    def test_tool_document_text_in_to_dict(self):
        """Test that to_dict uses get_searchable_text for text field."""
        doc = ToolDocument(
            id="test:tool",
            server_name="test",
            tool_name="tool",
            description="Test tool",
            parameters={},
            vector=[0.1],
            examples=["Example usage"],
        )

        result = doc.to_dict()
        expected_text = doc.get_searchable_text()

        assert result["text"] == expected_text
        assert "MCP Tool: test.tool" in result["text"]
        assert "Examples: Example usage" in result["text"]


class TestToolDocumentFromSchema:
    """Test suite for tool_document_from_schema helper function."""

    def test_basic_schema_conversion(self):
        """Test converting a basic MCP schema to ToolDocument."""
        schema = {
            "description": "List files in a directory",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }

        doc = tool_document_from_schema(
            server_name="filesystem",
            tool_name="list_directory",
            schema=schema,
            vector=[0.1, 0.2, 0.3],
        )

        assert doc.id == "filesystem:list_directory"
        assert doc.server_name == "filesystem"
        assert doc.tool_name == "list_directory"
        assert doc.description == "List files in a directory"
        assert doc.parameters["type"] == "object"
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.examples == []

    def test_schema_without_description(self):
        """Test schema without description field."""
        schema = {
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }
        }

        doc = tool_document_from_schema(
            server_name="search",
            tool_name="search_code",
            schema=schema,
            vector=[0.5],
        )

        assert doc.description == ""
        assert doc.parameters["type"] == "object"

    def test_schema_without_input_schema(self):
        """Test schema without inputSchema field."""
        schema = {"description": "Simple tool"}

        doc = tool_document_from_schema(
            server_name="simple",
            tool_name="do_something",
            schema=schema,
            vector=[0.1],
        )

        assert doc.description == "Simple tool"
        assert doc.parameters == {}

    def test_complex_schema(self):
        """Test schema with complex parameter structure."""
        schema = {
            "description": "Create a GitHub issue",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title"],
            },
        }

        doc = tool_document_from_schema(
            server_name="github",
            tool_name="create_issue",
            schema=schema,
            vector=[0.8, 0.9],
        )

        assert doc.description == "Create a GitHub issue"
        assert "title" in doc.parameters["properties"]
        assert "labels" in doc.parameters["properties"]
        assert doc.parameters["required"] == ["title"]

    def test_id_format(self):
        """Test that ID is formatted as server_name:tool_name."""
        doc = tool_document_from_schema(
            server_name="my-server",
            tool_name="my-tool",
            schema={},
            vector=[0.1],
        )

        assert doc.id == "my-server:my-tool"
