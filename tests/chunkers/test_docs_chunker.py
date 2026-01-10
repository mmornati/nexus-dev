"""Tests for documentation chunker."""

import pytest

from nexus_dev.chunkers.base import ChunkType
from nexus_dev.chunkers.docs_chunker import DocumentationChunker


class TestDocumentationChunker:
    """Test suite for DocumentationChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a DocumentationChunker instance."""
        return DocumentationChunker()

    def test_supported_extensions(self, chunker):
        """Test that documentation extensions are supported."""
        assert ".md" in chunker.supported_extensions
        assert ".markdown" in chunker.supported_extensions
        assert ".rst" in chunker.supported_extensions
        assert ".txt" in chunker.supported_extensions

    def test_chunk_markdown_sections(self, chunker, sample_markdown):
        """Test extracting Markdown sections by headers."""
        chunks = chunker.chunk_content(sample_markdown, "README.md")

        # Should have sections for each header
        section_names = [c.name for c in chunks]

        assert "Project Documentation" in section_names
        assert "Installation" in section_names
        assert "Configuration" in section_names

    def test_chunk_rst_sections(self, chunker, sample_rst):
        """Test extracting RST sections by underlines."""
        chunks = chunker.chunk_content(sample_rst, "README.rst")

        section_names = [c.name for c in chunks]

        assert "Project Documentation" in section_names
        assert "Installation" in section_names
        assert "Configuration" in section_names

    def test_markdown_preserves_code_blocks(self, chunker, sample_markdown):
        """Test that code blocks are preserved in sections."""
        chunks = chunker.chunk_content(sample_markdown, "README.md")

        install_section = next(c for c in chunks if c.name == "Installation")
        assert "pip install" in install_section.content

    def test_chunk_type_is_section(self, chunker, sample_markdown):
        """Test that chunks have correct type."""
        chunks = chunker.chunk_content(sample_markdown, "README.md")

        for chunk in chunks:
            assert chunk.chunk_type in (ChunkType.SECTION, ChunkType.DOCUMENTATION)

    def test_language_identifier(self, chunker, sample_markdown):
        """Test that language is correctly identified."""
        chunks = chunker.chunk_content(sample_markdown, "README.md")

        for chunk in chunks:
            assert chunk.language == "markdown"

    def test_rst_language_identifier(self, chunker, sample_rst):
        """Test that RST language is correctly identified."""
        chunks = chunker.chunk_content(sample_rst, "README.rst")

        for chunk in chunks:
            assert chunk.language == "rst"

    def test_chunk_empty_file(self, chunker):
        """Test handling of empty files."""
        chunks = chunker.chunk_content("", "empty.md")
        assert chunks == []

    def test_chunk_no_headers(self, chunker):
        """Test handling of markdown without headers."""
        content = "This is just some text without headers."
        chunks = chunker.chunk_content(content, "simple.md")

        assert len(chunks) >= 1
        # Without headers, it returns SECTION or DOCUMENTATION
        assert chunks[0].chunk_type in (ChunkType.DOCUMENTATION, ChunkType.SECTION)

    def test_nested_headers(self, chunker, sample_markdown):
        """Test handling of nested headers (h2, h3)."""
        chunks = chunker.chunk_content(sample_markdown, "README.md")

        # Database Settings is h3 under Configuration
        db_section = next((c for c in chunks if c.name == "Database Settings"), None)
        assert db_section is not None

    def test_plain_text_chunking(self, chunker):
        """Test chunking plain text files."""
        content = """First paragraph with some content.

Second paragraph has different information.

Third paragraph concludes the text."""

        chunks = chunker.chunk_content(content, "notes.txt")

        assert len(chunks) >= 1
        assert chunks[0].language == "text"
