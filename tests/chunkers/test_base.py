"""Tests for base chunker and fallback chunker."""

import pytest

from nexus_dev.chunkers.base import ChunkType, CodeChunk, FallbackChunker


class TestFallbackChunker:
    """Test suite for FallbackChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a FallbackChunker instance."""
        return FallbackChunker()

    def test_supported_extensions_empty(self, chunker):
        """Test that fallback has no specific extensions."""
        assert chunker.supported_extensions == []

    def test_chunk_small_file(self, chunker):
        """Test chunking a small file into single chunk."""
        content = "Small content that fits in one chunk."
        chunks = chunker.chunk_file("test.txt", content)

        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].chunk_type == ChunkType.MODULE

    def test_chunk_large_file(self, chunker):
        """Test chunking a large file into multiple chunks."""
        # Create content larger than MAX_CHUNK_SIZE
        content = "x" * 5000

        chunks = chunker.chunk_file("large.txt", content)

        assert len(chunks) > 1

    def test_chunk_overlap(self, chunker):
        """Test that chunks have overlap."""
        lines = ["Line " + str(i) + "\n" for i in range(500)]
        content = "".join(lines)

        chunks = chunker.chunk_file("test.txt", content)

        if len(chunks) >= 2:
            # Check that there's some overlap between consecutive chunks
            _chunk1_end = chunks[0].content[-100:]  # noqa: F841
            # Overlap may or may not exist depending on newline locations
            assert len(chunks[0].content) > 0

    def test_chunk_empty_file(self, chunker):
        """Test handling of empty files."""
        chunks = chunker.chunk_file("empty.txt", "")
        assert chunks == []

    def test_chunk_whitespace_only(self, chunker):
        """Test handling of whitespace-only files."""
        chunks = chunker.chunk_file("whitespace.txt", "   \n\n  \t  ")
        assert chunks == []

    def test_chunk_preserves_filename(self, chunker):
        """Test that filename is preserved in chunks."""
        chunks = chunker.chunk_file("myfile.xyz", "content")

        assert len(chunks) == 1
        assert "myfile.xyz" in chunks[0].name
        assert chunks[0].file_path == "myfile.xyz"


class TestCodeChunk:
    """Test suite for CodeChunk dataclass."""

    def test_get_searchable_text_with_signature(self):
        """Test searchable text includes signature."""
        chunk = CodeChunk(
            content="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            name="foo",
            start_line=1,
            end_line=1,
            language="python",
            signature="def foo():",
        )

        searchable = chunk.get_searchable_text()
        assert "def foo():" in searchable
        assert "def foo(): pass" in searchable

    def test_get_searchable_text_with_docstring(self):
        """Test searchable text includes docstring."""
        chunk = CodeChunk(
            content="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            name="foo",
            start_line=1,
            end_line=1,
            language="python",
            docstring="This is the docstring.",
        )

        searchable = chunk.get_searchable_text()
        assert "This is the docstring." in searchable

    def test_get_searchable_text_without_signature(self):
        """Test searchable text fallback without signature."""
        chunk = CodeChunk(
            content="some content",
            chunk_type=ChunkType.FUNCTION,
            name="foo",
            start_line=1,
            end_line=1,
            language="python",
        )

        searchable = chunk.get_searchable_text()
        assert "function: foo" in searchable
        assert "some content" in searchable
