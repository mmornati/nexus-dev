"""Tests for the chunker registry."""

from nexus_dev.chunkers import (
    ChunkerRegistry,
    DocumentationChunker,
    JavaChunker,
    JavaScriptChunker,
    PythonChunker,
    TypeScriptChunker,
)


class TestChunkerRegistry:
    """Test suite for ChunkerRegistry."""

    def test_registry_initialized(self):
        """Test that default chunkers are registered."""
        extensions = ChunkerRegistry.get_supported_extensions()

        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions
        assert ".java" in extensions
        assert ".md" in extensions

    def test_get_chunker_python(self):
        """Test getting Python chunker."""
        chunker = ChunkerRegistry.get_chunker("test.py")
        assert isinstance(chunker, PythonChunker)

    def test_get_chunker_javascript(self):
        """Test getting JavaScript chunker."""
        chunker = ChunkerRegistry.get_chunker("test.js")
        assert isinstance(chunker, JavaScriptChunker)

    def test_get_chunker_typescript(self):
        """Test getting TypeScript chunker."""
        chunker = ChunkerRegistry.get_chunker("test.ts")
        assert isinstance(chunker, TypeScriptChunker)

    def test_get_chunker_java(self):
        """Test getting Java chunker."""
        chunker = ChunkerRegistry.get_chunker("test.java")
        assert isinstance(chunker, JavaChunker)

    def test_get_chunker_markdown(self):
        """Test getting documentation chunker for Markdown."""
        chunker = ChunkerRegistry.get_chunker("README.md")
        assert isinstance(chunker, DocumentationChunker)

    def test_get_chunker_unsupported(self):
        """Test getting chunker for unsupported extension."""
        chunker = ChunkerRegistry.get_chunker("test.xyz")
        assert chunker is None

    def test_is_supported(self):
        """Test checking if file type is supported."""
        assert ChunkerRegistry.is_supported("test.py")
        assert ChunkerRegistry.is_supported("test.js")
        assert not ChunkerRegistry.is_supported("test.xyz")

    def test_get_language(self):
        """Test getting language identifier."""
        assert ChunkerRegistry.get_language("test.py") == "python"
        assert ChunkerRegistry.get_language("test.js") == "javascript"
        assert ChunkerRegistry.get_language("test.ts") == "typescript"
        assert ChunkerRegistry.get_language("test.java") == "java"
        assert ChunkerRegistry.get_language("test.md") == "markdown"
        assert ChunkerRegistry.get_language("test.xyz") == "unknown"

    def test_chunk_file_python(self, sample_python_code):
        """Test chunking Python file through registry."""
        chunks = ChunkerRegistry.chunk_file("test.py", sample_python_code)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.language == "python"

    def test_chunk_file_unsupported_uses_fallback(self):
        """Test that unsupported files use fallback chunker."""
        content = "Some content in an unknown file type."
        chunks = ChunkerRegistry.chunk_file("test.xyz", content)

        assert len(chunks) >= 1
        assert chunks[0].language == "unknown"

    def test_case_insensitive_extensions(self):
        """Test that extensions are case-insensitive."""
        assert ChunkerRegistry.is_supported("test.PY")
        assert ChunkerRegistry.is_supported("test.Js")
        assert ChunkerRegistry.get_chunker("TEST.JAVA") is not None
