"""LanceDB database manager for Nexus-Dev."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import lancedb
import pyarrow as pa

from .config import NexusConfig
from .embeddings import EmbeddingProvider


class DocumentType(str, Enum):
    """Type of indexed document."""

    CODE = "code"
    LESSON = "lesson"
    DOCUMENTATION = "documentation"
    TOOL = "tool"
    INSIGHT = "insight"  # LLM reasoning, mistakes, backtracking
    IMPLEMENTATION = "implementation"  # Plan summaries, design decisions
    GITHUB_ISSUE = "github_issue"
    GITHUB_PR = "github_pr"


@dataclass
class Document:
    """A document to be stored in the vector database.

    Attributes:
        id: Unique document identifier (UUID).
        text: Document content.
        vector: Embedding vector.
        project_id: Project this document belongs to.
        file_path: Source file path.
        doc_type: Type of document (code, lesson, documentation, tool).
        chunk_type: Type of code chunk (function, class, method, module).
        language: Programming language or "markdown".
        name: Name of the code element (function/class name).
        start_line: Starting line number in source file.
        end_line: Ending line number in source file.
        timestamp: When the document was indexed.
        server_name: For TOOL type: MCP server name.
        parameters_schema: For TOOL type: JSON schema string.
    """

    id: str
    text: str
    vector: list[float]
    project_id: str
    file_path: str
    doc_type: DocumentType
    chunk_type: str = "module"
    language: str = "unknown"
    name: str = ""
    start_line: int = 0
    end_line: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    server_name: str = ""
    parameters_schema: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LanceDB insertion."""
        return {
            "id": self.id,
            "text": self.text,
            "vector": self.vector,
            "project_id": self.project_id,
            "file_path": self.file_path,
            "doc_type": self.doc_type.value,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "timestamp": self.timestamp.isoformat(),
            "server_name": self.server_name,
            "parameters_schema": self.parameters_schema,
        }


@dataclass
class ToolDocument:
    """An MCP tool document for indexing and search.

    Attributes:
        id: Unique identifier (server_name:tool_name)
        server_name: Name of the MCP server (e.g., "github")
        tool_name: Name of the tool (e.g., "create_pull_request")
        description: Tool description/docstring
        parameters: JSON schema dict for parameters
        examples: Optional usage examples
        vector: Embedding vector for semantic search
        timestamp: When the tool was indexed
    """

    id: str
    server_name: str
    tool_name: str
    description: str
    parameters: dict[str, Any]
    vector: list[float]
    examples: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LanceDB insertion."""
        return {
            "id": self.id,
            "text": self.get_searchable_text(),
            "vector": self.vector,
            "project_id": "mcp_tools",  # Special project for tools
            "file_path": f"mcp://{self.server_name}/{self.tool_name}",
            "doc_type": DocumentType.TOOL.value,
            "chunk_type": "tool",
            "language": "mcp",
            "name": self.tool_name,
            "start_line": 0,
            "end_line": 0,
            "timestamp": self.timestamp.isoformat(),
            "server_name": self.server_name,
            "parameters_schema": json.dumps(self.parameters),
        }

    def get_searchable_text(self) -> str:
        """Get text for embedding generation."""
        parts = [
            f"MCP Tool: {self.server_name}.{self.tool_name}",
            f"Description: {self.description}",
        ]
        if self.examples:
            parts.append(f"Examples: {', '.join(self.examples)}")
        return "\n".join(parts)


@dataclass
class SearchResult:
    """Result from a similarity search.

    Attributes:
        id: Document ID.
        text: Document content.
        score: Similarity score (lower is more similar for L2 distance).
        project_id: Project the document belongs to.
        file_path: Source file path.
        doc_type: Type of document.
        chunk_type: Type of code chunk.
        language: Programming language.
        name: Name of the code element.
        start_line: Starting line number.
        end_line: Ending line number.
        server_name: For TOOL type: MCP server name.
        parameters_schema: For TOOL type: JSON schema string.
    """

    id: str
    text: str
    score: float
    project_id: str
    file_path: str
    doc_type: str
    chunk_type: str
    language: str
    name: str
    start_line: int
    end_line: int
    server_name: str = ""
    parameters_schema: str = ""


class NexusDatabase:
    """LanceDB wrapper for Nexus-Dev vector storage."""

    TABLE_NAME = "documents"

    def __init__(
        self,
        config: NexusConfig,
        embedder: EmbeddingProvider,
    ) -> None:
        """Initialize the database connection.

        Args:
            config: Nexus-Dev configuration.
            embedder: Embedding provider for vector generation.
        """
        self.config = config
        self.embedder = embedder
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    def _get_schema(self) -> pa.Schema:
        """Get the PyArrow schema for the documents table."""
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field(
                    "vector",
                    pa.list_(pa.float32(), self.config.get_embedding_dimensions()),
                ),
                pa.field("project_id", pa.string()),
                pa.field("file_path", pa.string()),
                pa.field("doc_type", pa.string()),
                pa.field("chunk_type", pa.string()),
                pa.field("language", pa.string()),
                pa.field("name", pa.string()),
                pa.field("start_line", pa.int32()),
                pa.field("end_line", pa.int32()),
                pa.field("timestamp", pa.string()),
                pa.field("server_name", pa.string()),
                pa.field("parameters_schema", pa.string()),
            ]
        )

    def reset(self) -> None:
        """Delete the entire table to force schema recreation."""
        if self._db is None:
            self.connect()
        assert self._db is not None

        if self.TABLE_NAME in self._db.table_names():
            self._db.drop_table(self.TABLE_NAME)
            self._table = None

    def connect(self) -> None:
        """Connect to the LanceDB database and ensure table exists."""
        db_path = self.config.get_db_path()
        db_path.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(db_path))

        # Create table if it doesn't exist
        if self.TABLE_NAME not in self._db.table_names():
            self._table = self._db.create_table(
                self.TABLE_NAME,
                schema=self._get_schema(),
            )
        else:
            self._table = self._db.open_table(self.TABLE_NAME)

    def _ensure_connected(self) -> lancedb.table.Table:
        """Ensure database is connected and return table.

        We re-open the table to ensure we see the latest updates from other processes.
        """
        if self._db is None:
            self.connect()
        assert self._db is not None

        # Always re-open table to pick up external updates (e.g. from indexer)
        try:
            self._table = self._db.open_table(self.TABLE_NAME)
        except Exception:
            # Table might not exist yet if created but not committed, or other issue
            # If so, rely on connect()'s creation logic or handle error
            if self._table is None:
                self.connect()

        assert self._table is not None
        return self._table

    async def upsert_document(self, doc: Document) -> str:
        """Insert or update a document.

        Args:
            doc: Document to upsert.

        Returns:
            Document ID.
        """
        table = self._ensure_connected()

        # Delete existing document with same ID if exists
        try:
            table.delete(f"id = '{doc.id}'")
        except Exception:
            pass  # Ignore if document doesn't exist

        # Insert new document
        table.add([doc.to_dict()])

        return doc.id

    async def upsert_documents(self, docs: list[Document]) -> list[str]:
        """Insert or update multiple documents.

        Args:
            docs: Documents to upsert.

        Returns:
            List of document IDs.
        """
        if not docs:
            return []

        table = self._ensure_connected()

        # Delete existing documents
        ids = [doc.id for doc in docs]
        for doc_id in ids:
            try:
                table.delete(f"id = '{doc_id}'")
            except Exception:
                pass

        # Insert all documents
        table.add([doc.to_dict() for doc in docs])

        return ids

    async def search(
        self,
        query: str,
        project_id: str | None = None,
        doc_type: DocumentType | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Perform semantic similarity search.

        Args:
            query: Search query text.
            project_id: Optional project filter.
            doc_type: Optional document type filter.
            limit: Maximum number of results.

        Returns:
            List of search results ordered by similarity.
        """
        table = self._ensure_connected()

        # Generate query embedding
        query_vector = await self.embedder.embed(query)

        # Build search query
        search_query = table.search(query_vector).limit(limit)

        # Apply filters
        filters = []
        if project_id:
            filters.append(f"project_id = '{project_id}'")
        if doc_type:
            filters.append(f"doc_type = '{doc_type.value}'")

        if filters:
            search_query = search_query.where(" AND ".join(filters))

        # Execute search
        results = search_query.to_pandas()

        # Convert to SearchResult objects
        search_results = []
        for _, row in results.iterrows():
            search_results.append(
                SearchResult(
                    id=row["id"],
                    text=row["text"],
                    score=row["_distance"],
                    project_id=row["project_id"],
                    file_path=row["file_path"],
                    doc_type=row["doc_type"],
                    chunk_type=row["chunk_type"],
                    language=row["language"],
                    name=row["name"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    server_name=row.get("server_name", ""),
                    parameters_schema=row.get("parameters_schema", ""),
                )
            )

        return search_results

    async def delete_by_file(self, file_path: str, project_id: str) -> int:
        """Delete all documents for a specific file.

        Args:
            file_path: Path to the file.
            project_id: Project ID.

        Returns:
            Number of documents deleted.
        """
        table = self._ensure_connected()

        # Get count before deletion
        try:
            count_before = len(
                table.search()
                .where(f"file_path = '{file_path}' AND project_id = '{project_id}'")
                .to_pandas()
            )
        except Exception:
            count_before = 0

        # Delete documents
        try:
            table.delete(f"file_path = '{file_path}' AND project_id = '{project_id}'")
        except Exception:
            pass

        return count_before

    async def delete_by_project(self, project_id: str) -> int:
        """Delete all documents for a project.

        Args:
            project_id: Project ID.

        Returns:
            Number of documents deleted.
        """
        table = self._ensure_connected()

        # Get count before deletion
        try:
            count_before = len(table.search().where(f"project_id = '{project_id}'").to_pandas())
        except Exception:
            count_before = 0

        # Delete documents
        try:
            table.delete(f"project_id = '{project_id}'")
        except Exception:
            pass

        return count_before

    async def get_project_stats(self, project_id: str | None = None) -> dict[str, int]:
        """Get statistics for a project or all projects.

        Args:
            project_id: Project ID. If None, returns stats for all projects.

        Returns:
            Dictionary with counts by document type.
        """
        table = self._ensure_connected()

        try:
            # Get all data as pandas DataFrame
            df = table.to_pandas()

            # Filter by project_id if specified
            if project_id:
                df = df[df["project_id"] == project_id]

            # Group by document type
            if len(df) == 0:
                return {"total": 0}

            stats = df.groupby("doc_type").size().to_dict()
            stats["total"] = len(df)
            return stats
        except Exception as e:
            # Return error details for debugging
            import logging

            logging.error(f"Failed to get project stats: {e}", exc_info=True)
            return {"total": 0}

    async def get_recent_lessons(
        self,
        project_id: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Get recent lessons ordered by timestamp.

        Args:
            project_id: Optional project filter.
            limit: Maximum number of results.

        Returns:
            List of recent lessons.
        """
        table = self._ensure_connected()

        try:
            query = table.search()
            filters = [f"doc_type = '{DocumentType.LESSON.value}'"]

            if project_id:
                filters.append(f"project_id = '{project_id}'")

            df = query.where(" AND ".join(filters)).limit(limit * 2).to_pandas()

            # Sort by timestamp (descending) and limit
            df = df.sort_values("timestamp", ascending=False).head(limit)

            results = []
            for _, row in df.iterrows():
                results.append(
                    SearchResult(
                        id=row["id"],
                        text=row["text"],
                        score=0.0,
                        project_id=row["project_id"],
                        file_path=row["file_path"],
                        doc_type=row["doc_type"],
                        chunk_type=row["chunk_type"],
                        language=row["language"],
                        name=row["name"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        server_name=row.get("server_name", ""),
                        parameters_schema=row.get("parameters_schema", ""),
                    )
                )
            return results
            return results
        except Exception:
            return []


def generate_document_id(
    project_id: str,
    file_path: str,
    chunk_name: str,
    start_line: int,
) -> str:
    """Generate a deterministic document ID.

    This allows for idempotent updates when re-indexing the same code.

    Args:
        project_id: Project ID.
        file_path: File path.
        chunk_name: Name of the chunk (function/class name).
        start_line: Starting line number.

    Returns:
        Deterministic UUID based on input parameters.
    """
    # Create a deterministic ID from the combination
    key = f"{project_id}:{file_path}:{chunk_name}:{start_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def tool_document_from_schema(
    server_name: str,
    tool_name: str,
    schema: dict[str, Any],
    vector: list[float],
) -> ToolDocument:
    """Create ToolDocument from MCP tool schema.

    Args:
        server_name: Name of the MCP server.
        tool_name: Name of the tool.
        schema: MCP tool schema dictionary.
        vector: Embedding vector for the tool.

    Returns:
        ToolDocument instance.
    """
    return ToolDocument(
        id=f"{server_name}:{tool_name}",
        server_name=server_name,
        tool_name=tool_name,
        description=schema.get("description", ""),
        parameters=schema.get("inputSchema", {}),
        vector=vector,
    )
