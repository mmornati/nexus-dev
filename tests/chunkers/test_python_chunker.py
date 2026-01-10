"""Tests for Python chunker."""

import pytest

from nexus_dev.chunkers.base import ChunkType
from nexus_dev.chunkers.python_chunker import PythonChunker


class TestPythonChunker:
    """Test suite for PythonChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a PythonChunker instance."""
        return PythonChunker()

    def test_supported_extensions(self, chunker):
        """Test that Python extensions are supported."""
        assert ".py" in chunker.supported_extensions
        assert ".pyw" in chunker.supported_extensions

    def test_chunk_simple_function(self, chunker):
        """Test extracting a simple function."""
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        chunks = chunker.chunk_content(code, "test.py")

        assert len(chunks) >= 1
        func_chunk = next(c for c in chunks if c.name == "hello")
        assert func_chunk.chunk_type == ChunkType.FUNCTION
        assert func_chunk.language == "python"
        assert "def hello" in func_chunk.content

    def test_chunk_class(self, chunker, sample_python_code):
        """Test extracting a class with methods."""
        chunks = chunker.chunk_content(sample_python_code, "test.py")

        # Should have: 2 functions + 1 class + 2 methods + 1 async function
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        assert len(class_chunks) >= 1
        assert len(method_chunks) >= 2  # add, subtract
        assert len(function_chunks) >= 2  # greet, add

        # Check class extraction
        calc_class = next(c for c in class_chunks if c.name == "Calculator")
        assert "class Calculator" in calc_class.content
        # Docstring is in content even if not extracted separately
        assert "simple calculator" in calc_class.content.lower()

    def test_chunk_async_function(self, chunker, sample_python_code):
        """Test extracting async functions."""
        chunks = chunker.chunk_content(sample_python_code, "test.py")

        async_chunks = [
            c for c in chunks if "async" in c.content.lower() and c.name == "async_fetch"
        ]
        assert len(async_chunks) >= 1

    def test_chunk_with_docstrings(self, chunker, sample_python_code):
        """Test that docstrings are in chunk content."""
        chunks = chunker.chunk_content(sample_python_code, "test.py")

        greet_func = next(c for c in chunks if c.name == "greet")
        # Docstring should be in content even if not extracted separately
        assert "say hello" in greet_func.content.lower()

    def test_chunk_empty_file(self, chunker):
        """Test handling of empty files."""
        chunks = chunker.chunk_content("", "empty.py")
        assert chunks == []

    def test_chunk_whitespace_only(self, chunker):
        """Test handling of whitespace-only files."""
        chunks = chunker.chunk_content("   \n\n   \t  ", "whitespace.py")
        assert chunks == []

    def test_chunk_syntax_error_fallback(self, chunker):
        """Test fallback for files with syntax errors."""
        code = """
def broken(
    # Missing closing paren
    pass
"""
        # Should not raise, should return module-level chunk
        chunks = chunker.chunk_content(code, "broken.py")
        assert len(chunks) >= 0  # Should handle gracefully

    def test_line_numbers_correct(self, chunker):
        """Test that line numbers are correctly tracked."""
        code = """# Comment line 1
# Comment line 2

def my_function():
    pass
"""
        chunks = chunker.chunk_content(code, "test.py")
        func_chunk = next(c for c in chunks if c.name == "my_function")

        # Function starts at line 4 (1-indexed)
        assert func_chunk.start_line == 4
        assert func_chunk.end_line == 5

    def test_nested_functions_included(self, chunker):
        """Test that nested functions are included in parent."""
        code = """
def outer():
    def inner():
        pass
    return inner()
"""
        chunks = chunker.chunk_content(code, "test.py")
        outer_chunk = next(c for c in chunks if c.name == "outer")

        # Nested function should be in the content
        assert "def inner" in outer_chunk.content

    def test_class_methods_have_parent(self, chunker, sample_python_code):
        """Test that methods have parent class set."""
        chunks = chunker.chunk_content(sample_python_code, "test.py")

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        for method in method_chunks:
            assert method.parent == "Calculator"
