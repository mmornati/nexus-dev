"""Base classes for code chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ChunkType(str, Enum):
    """Type of code chunk."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    DOCUMENTATION = "documentation"
    SECTION = "section"  # For documentation sections
    LESSON = "lesson"


@dataclass
class CodeChunk:
    """Represents a semantic code chunk.

    Attributes:
        content: The actual code/text content.
        chunk_type: Type of chunk (function, class, etc.).
        name: Name of the code element.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        language: Programming language identifier.
        file_path: Source file path.
        parent: Parent element name (e.g., class name for methods).
        docstring: Extracted docstring if available.
        imports: List of imports used in this chunk (for context).
        signature: Function/method signature.
    """

    content: str
    chunk_type: ChunkType
    name: str
    start_line: int
    end_line: int
    language: str
    file_path: str = ""
    parent: str | None = None
    docstring: str | None = None
    imports: list[str] = field(default_factory=list)
    signature: str | None = None

    def get_searchable_text(self) -> str:
        """Get text optimized for embedding and search.

        Combines content with metadata for better semantic matching.
        """
        parts = []

        # Add signature/name for context
        if self.signature:
            parts.append(f"# {self.signature}")
        elif self.name:
            parts.append(f"# {self.chunk_type.value}: {self.name}")

        # Add docstring if available
        if self.docstring:
            parts.append(self.docstring)

        # Add content
        parts.append(self.content)

        return "\n".join(parts)


class BaseChunker(ABC):
    """Abstract base class for all code chunkers.

    To add support for a new language:
    1. Create a new class inheriting from BaseChunker
    2. Implement supported_extensions property
    3. Implement chunk_file and chunk_content methods
    4. Register the chunker in ChunkerRegistry
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this chunker handles.

        Returns:
            List of extensions including the dot (e.g., ['.py', '.pyw']).
        """

    @property
    def language_name(self) -> str:
        """Human-readable name of the language.

        Returns:
            Language name (e.g., 'Python', 'JavaScript').
        """
        return self.__class__.__name__.replace("Chunker", "")

    @abstractmethod
    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Parse a file and extract semantic chunks.

        Args:
            file_path: Path to the file (for metadata).
            content: File content as string.

        Returns:
            List of extracted code chunks.
        """

    def chunk_content(self, content: str, file_name: str = "unknown") -> list[CodeChunk]:
        """Parse content string directly.

        Args:
            content: Code/text content.
            file_name: Filename for metadata.

        Returns:
            List of extracted code chunks.
        """
        return self.chunk_file(file_name, content)


class FallbackChunker(BaseChunker):
    """Fallback chunker for unsupported file types.

    Uses simple character-based chunking with overlap.
    """

    MAX_CHUNK_SIZE = 1500
    OVERLAP_SIZE = 200

    @property
    def supported_extensions(self) -> list[str]:
        return []  # Matches nothing, used as fallback

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Chunk content by character count with overlap.

        Args:
            file_path: Path to the file.
            content: File content.

        Returns:
            List of text chunks.
        """
        if not content.strip():
            return []

        # For small files, return as single chunk
        if len(content) <= self.MAX_CHUNK_SIZE:
            return [
                CodeChunk(
                    content=content,
                    chunk_type=ChunkType.MODULE,
                    name=file_path.split("/")[-1] if "/" in file_path else file_path,
                    start_line=1,
                    end_line=content.count("\n") + 1,
                    language="unknown",
                    file_path=file_path,
                )
            ]

        # Split into overlapping chunks
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.MAX_CHUNK_SIZE, len(content))

            # Try to break at a newline
            if end < len(content):
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start + self.MAX_CHUNK_SIZE // 2:
                    end = newline_pos + 1

            chunk_text = content[start:end]
            start_line = content[:start].count("\n") + 1
            end_line = start_line + chunk_text.count("\n")

            chunks.append(
                CodeChunk(
                    content=chunk_text,
                    chunk_type=ChunkType.MODULE,
                    name=f"{file_path.split('/')[-1]}:chunk_{chunk_index}",
                    start_line=start_line,
                    end_line=end_line,
                    language="unknown",
                    file_path=file_path,
                )
            )

            # Move start with overlap
            start = end - self.OVERLAP_SIZE if end < len(content) else end
            chunk_index += 1

        return chunks
