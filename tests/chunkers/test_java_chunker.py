"""Tests for Java chunker."""

import pytest

from nexus_dev.chunkers.base import ChunkType
from nexus_dev.chunkers.java_chunker import JavaChunker


class TestJavaChunker:
    """Test suite for JavaChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a JavaChunker instance."""
        return JavaChunker()

    def test_supported_extensions(self, chunker):
        """Test that Java extensions are supported."""
        assert ".java" in chunker.supported_extensions

    def test_chunk_class(self, chunker, sample_java_code):
        """Test extracting Java classes."""
        chunks = chunker.chunk_content(sample_java_code, "Calculator.java")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

        calc_class = next(c for c in class_chunks if c.name == "Calculator")
        assert "public class Calculator" in calc_class.content
        assert calc_class.language == "java"

    def test_chunk_methods(self, chunker, sample_java_code):
        """Test extracting Java methods."""
        chunks = chunker.chunk_content(sample_java_code, "Calculator.java")

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        method_names = [c.name for c in method_chunks]

        assert "add" in method_names
        assert "subtract" in method_names

    def test_chunk_constructor(self, chunker, sample_java_code):
        """Test extracting Java constructors."""
        chunks = chunker.chunk_content(sample_java_code, "Calculator.java")

        # Constructor should be treated as a method with class name
        constructor_chunks = [
            c for c in chunks if "Calculator" in c.name and "initial" in c.content
        ]
        assert len(constructor_chunks) >= 1

    def test_chunk_javadoc(self, chunker, sample_java_code):
        """Test that Javadoc comments are extracted."""
        chunks = chunker.chunk_content(sample_java_code, "Calculator.java")

        calc_class = next(
            c for c in chunks if c.name == "Calculator" and c.chunk_type == ChunkType.CLASS
        )
        assert calc_class.docstring is not None
        assert "simple calculator" in calc_class.docstring.lower()

    def test_chunk_interface(self, chunker):
        """Test extracting Java interfaces."""
        code = """
public interface Greeter {
    String greet(String name);
}
"""
        chunks = chunker.chunk_content(code, "Greeter.java")

        interface_chunks = [c for c in chunks if c.name == "Greeter"]
        assert len(interface_chunks) >= 1

    def test_chunk_enum(self, chunker):
        """Test extracting Java enums."""
        code = """
public enum Color {
    RED,
    GREEN,
    BLUE
}
"""
        chunks = chunker.chunk_content(code, "Color.java")

        enum_chunks = [c for c in chunks if c.name == "Color"]
        assert len(enum_chunks) >= 1

    def test_chunk_empty_file(self, chunker):
        """Test handling of empty files."""
        chunks = chunker.chunk_content("", "Empty.java")
        assert chunks == []

    def test_method_parent_class(self, chunker, sample_java_code):
        """Test that methods have correct parent class."""
        chunks = chunker.chunk_content(sample_java_code, "Calculator.java")

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        for method in method_chunks:
            assert method.parent == "Calculator"
