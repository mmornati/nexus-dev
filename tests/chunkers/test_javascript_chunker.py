"""Tests for JavaScript/TypeScript chunkers."""

import pytest

from nexus_dev.chunkers.base import ChunkType
from nexus_dev.chunkers.javascript_chunker import JavaScriptChunker, TypeScriptChunker


class TestJavaScriptChunker:
    """Test suite for JavaScriptChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a JavaScriptChunker instance."""
        return JavaScriptChunker()

    def test_supported_extensions(self, chunker):
        """Test that JavaScript extensions are supported."""
        assert ".js" in chunker.supported_extensions
        assert ".jsx" in chunker.supported_extensions
        assert ".mjs" in chunker.supported_extensions
        assert ".cjs" in chunker.supported_extensions

    def test_chunk_function_declaration(self, chunker):
        """Test extracting function declarations."""
        code = """
function greet(name) {
    return "Hello, " + name;
}
"""
        chunks = chunker.chunk_content(code, "test.js")

        func_chunk = next(c for c in chunks if c.name == "greet")
        assert func_chunk.chunk_type == ChunkType.FUNCTION
        assert func_chunk.language == "javascript"

    def test_chunk_arrow_function(self, chunker, sample_javascript_code):
        """Test extracting arrow functions."""
        chunks = chunker.chunk_content(sample_javascript_code, "test.js")

        arrow_chunks = [c for c in chunks if c.name == "add"]
        assert len(arrow_chunks) >= 1

    def test_chunk_class(self, chunker, sample_javascript_code):
        """Test extracting JavaScript classes."""
        chunks = chunker.chunk_content(sample_javascript_code, "test.js")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

        calc_class = next(c for c in class_chunks if c.name == "Calculator")
        assert "class Calculator" in calc_class.content

    def test_chunk_class_methods(self, chunker, sample_javascript_code):
        """Test extracting class methods."""
        chunks = chunker.chunk_content(sample_javascript_code, "test.js")

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        method_names = [c.name for c in method_chunks]

        assert "add" in method_names
        assert "subtract" in method_names

    def test_chunk_empty_file(self, chunker):
        """Test handling of empty files."""
        chunks = chunker.chunk_content("", "empty.js")
        assert chunks == []

    def test_const_arrow_function(self, chunker):
        """Test const with arrow function."""
        code = """
const multiply = (a, b) => a * b;
"""
        chunks = chunker.chunk_content(code, "test.js")

        multi_chunk = next((c for c in chunks if c.name == "multiply"), None)
        assert multi_chunk is not None
        assert multi_chunk.chunk_type == ChunkType.FUNCTION


class TestTypeScriptChunker:
    """Test suite for TypeScriptChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a TypeScriptChunker instance."""
        return TypeScriptChunker()

    def test_supported_extensions(self, chunker):
        """Test that TypeScript extensions are supported."""
        assert ".ts" in chunker.supported_extensions
        assert ".tsx" in chunker.supported_extensions
        assert ".mts" in chunker.supported_extensions
        assert ".cts" in chunker.supported_extensions

    def test_chunk_typed_function(self, chunker, sample_typescript_code):
        """Test extracting typed functions."""
        chunks = chunker.chunk_content(sample_typescript_code, "test.ts")

        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) >= 1

    def test_chunk_typescript_class(self, chunker, sample_typescript_code):
        """Test extracting TypeScript classes with modifiers."""
        chunks = chunker.chunk_content(sample_typescript_code, "test.ts")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

        # Find any class with Calculator in name or content
        calc_class = next((c for c in class_chunks if "Calculator" in c.content), None)
        assert calc_class is not None
        assert "value" in calc_class.content

    def test_language_identifier(self, chunker, sample_typescript_code):
        """Test that language is correctly identified as typescript."""
        chunks = chunker.chunk_content(sample_typescript_code, "test.ts")

        for chunk in chunks:
            assert chunk.language == "typescript"
