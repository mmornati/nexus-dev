"""Chunker registry and utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseChunker, ChunkType, CodeChunk, FallbackChunker
from .docs_chunker import DocumentationChunker
from .java_chunker import JavaChunker
from .javascript_chunker import JavaScriptChunker, TypeScriptChunker
from .python_chunker import PythonChunker

if TYPE_CHECKING:
    pass

__all__ = [
    "BaseChunker",
    "ChunkType",
    "CodeChunk",
    "ChunkerRegistry",
    "DocumentationChunker",
    "FallbackChunker",
    "JavaChunker",
    "JavaScriptChunker",
    "PythonChunker",
    "TypeScriptChunker",
]


class ChunkerRegistry:
    """Registry for file extension to chunker mapping.

    The registry automatically maps file extensions to appropriate chunkers.
    Use `get_chunker()` to get a chunker for a specific file, or
    `chunk_file()` to directly chunk a file.
    """

    _chunkers: dict[str, BaseChunker] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure default chunkers are registered."""
        if not cls._initialized:
            cls._register_default_chunkers()
            cls._initialized = True

    @classmethod
    def _register_default_chunkers(cls) -> None:
        """Register all built-in chunkers."""
        cls.register(PythonChunker())
        cls.register(JavaScriptChunker())
        cls.register(TypeScriptChunker())
        cls.register(JavaChunker())
        cls.register(DocumentationChunker())

    @classmethod
    def register(cls, chunker: BaseChunker) -> None:
        """Register a chunker for its supported extensions.

        Args:
            chunker: Chunker instance to register.
        """
        for ext in chunker.supported_extensions:
            ext_lower = ext.lower()
            if not ext_lower.startswith("."):
                ext_lower = f".{ext_lower}"
            cls._chunkers[ext_lower] = chunker

    @classmethod
    def get_chunker(cls, file_path: str | Path) -> BaseChunker | None:
        """Get the appropriate chunker for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Chunker instance or None if no specific chunker matches.
        """
        cls._ensure_initialized()

        path = Path(file_path)
        ext = path.suffix.lower()

        return cls._chunkers.get(ext)

    @classmethod
    def chunk_file(cls, file_path: str | Path, content: str) -> list[CodeChunk]:
        """Chunk a file using the appropriate chunker.

        Args:
            file_path: Path to the file.
            content: File content.

        Returns:
            List of code chunks.
        """
        cls._ensure_initialized()

        file_path_str = str(file_path)
        chunker = cls.get_chunker(file_path)

        if chunker is not None:
            return chunker.chunk_file(file_path_str, content)

        # Use fallback chunker for unknown types
        return FallbackChunker().chunk_file(file_path_str, content)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of all supported file extensions.

        Returns:
            List of supported extensions.
        """
        cls._ensure_initialized()
        return list(cls._chunkers.keys())

    @classmethod
    def is_supported(cls, file_path: str | Path) -> bool:
        """Check if a file type is supported.

        Args:
            file_path: Path to check.

        Returns:
            True if the file type has a registered chunker.
        """
        cls._ensure_initialized()

        path = Path(file_path)
        ext = path.suffix.lower()

        return ext in cls._chunkers

    @classmethod
    def get_language(cls, file_path: str | Path) -> str:
        """Get the language identifier for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Language identifier string.
        """
        ext_to_language = {
            ".py": "python",
            ".pyw": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mts": "typescript",
            ".cts": "typescript",
            ".java": "java",
            ".md": "markdown",
            ".markdown": "markdown",
            ".rst": "rst",
            ".txt": "text",
        }

        path = Path(file_path)
        ext = path.suffix.lower()

        return ext_to_language.get(ext, "unknown")
