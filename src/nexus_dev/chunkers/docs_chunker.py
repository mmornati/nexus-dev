"""Documentation chunker for Markdown and RST files."""

from __future__ import annotations

import re

from .base import BaseChunker, ChunkType, CodeChunk


class DocumentationChunker(BaseChunker):
    """Chunker for documentation files (Markdown, RST, plain text).

    Splits documentation by headers/sections while keeping code blocks intact.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown", ".rst", ".txt"]

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Parse documentation file and extract sections as chunks.

        Args:
            file_path: Path to the documentation file.
            content: File content.

        Returns:
            List of documentation chunks split by headers.
        """
        if not content.strip():
            return []

        ext = file_path.split(".")[-1].lower()

        if ext in ("md", "markdown"):
            return self._chunk_markdown(file_path, content)
        elif ext == "rst":
            return self._chunk_rst(file_path, content)
        else:
            return self._chunk_plain_text(file_path, content)

    def _chunk_markdown(self, file_path: str, content: str) -> list[CodeChunk]:
        """Chunk Markdown content by headers.

        Args:
            file_path: File path.
            content: Markdown content.

        Returns:
            List of section chunks.
        """
        lines = content.split("\n")
        chunks: list[CodeChunk] = []

        # Pattern for Markdown headers (## or ###)
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        current_section: list[str] = []
        current_header = ""
        current_start_line = 1
        parent_header: str | None = None

        for i, line in enumerate(lines):
            match = header_pattern.match(line)

            if match:
                # Save previous section if it has content
                if current_section:
                    section_content = "\n".join(current_section).strip()
                    if section_content:
                        chunks.append(
                            CodeChunk(
                                content=section_content,
                                chunk_type=ChunkType.SECTION,
                                name=current_header or file_path.split("/")[-1],
                                start_line=current_start_line,
                                end_line=i,
                                language="markdown",
                                file_path=file_path,
                                parent=parent_header,
                            )
                        )

                # Start new section
                level = len(match.group(1))
                header_text = match.group(2).strip()

                # Track parent for level 2+ headers
                if level == 1:
                    parent_header = None
                elif level == 2:
                    parent_header = current_header if current_header else None

                current_header = header_text
                current_section = [line]
                current_start_line = i + 1
            else:
                current_section.append(line)

        # Save final section
        if current_section:
            section_content = "\n".join(current_section).strip()
            if section_content:
                chunks.append(
                    CodeChunk(
                        content=section_content,
                        chunk_type=ChunkType.SECTION,
                        name=current_header or file_path.split("/")[-1],
                        start_line=current_start_line,
                        end_line=len(lines),
                        language="markdown",
                        file_path=file_path,
                        parent=parent_header,
                    )
                )

        # If no headers found, return whole file as one chunk
        if not chunks:
            chunks.append(
                CodeChunk(
                    content=content,
                    chunk_type=ChunkType.DOCUMENTATION,
                    name=file_path.split("/")[-1] if "/" in file_path else file_path,
                    start_line=1,
                    end_line=len(lines),
                    language="markdown",
                    file_path=file_path,
                )
            )

        return chunks

    def _chunk_rst(self, file_path: str, content: str) -> list[CodeChunk]:
        """Chunk RST content by title underlines.

        Args:
            file_path: File path.
            content: RST content.

        Returns:
            List of section chunks.
        """
        lines = content.split("\n")
        chunks: list[CodeChunk] = []

        # RST title underlines: =, -, ~, ^, ", etc.
        underline_pattern = re.compile(r"^[=\-~^\"\'`\*\+#]+$")

        current_section: list[str] = []
        current_header = ""
        current_start_line = 1

        i = 0
        while i < len(lines):
            # Check if next line is an underline (making current line a header)
            if (
                i + 1 < len(lines)
                and lines[i].strip()
                and underline_pattern.match(lines[i + 1].strip())
                and len(lines[i + 1].strip()) >= len(lines[i].strip())
            ):
                # Save previous section
                if current_section:
                    section_content = "\n".join(current_section).strip()
                    if section_content:
                        chunks.append(
                            CodeChunk(
                                content=section_content,
                                chunk_type=ChunkType.SECTION,
                                name=current_header or file_path.split("/")[-1],
                                start_line=current_start_line,
                                end_line=i,
                                language="rst",
                                file_path=file_path,
                            )
                        )

                # Start new section with header
                current_header = lines[i].strip()
                current_section = [lines[i], lines[i + 1]]
                current_start_line = i + 1
                i += 2
            else:
                current_section.append(lines[i])
                i += 1

        # Save final section
        if current_section:
            section_content = "\n".join(current_section).strip()
            if section_content:
                chunks.append(
                    CodeChunk(
                        content=section_content,
                        chunk_type=ChunkType.SECTION,
                        name=current_header or file_path.split("/")[-1],
                        start_line=current_start_line,
                        end_line=len(lines),
                        language="rst",
                        file_path=file_path,
                    )
                )

        if not chunks:
            chunks.append(
                CodeChunk(
                    content=content,
                    chunk_type=ChunkType.DOCUMENTATION,
                    name=file_path.split("/")[-1] if "/" in file_path else file_path,
                    start_line=1,
                    end_line=len(lines),
                    language="rst",
                    file_path=file_path,
                )
            )

        return chunks

    def _chunk_plain_text(self, file_path: str, content: str) -> list[CodeChunk]:
        """Chunk plain text by paragraph breaks or fixed size.

        Args:
            file_path: File path.
            content: Text content.

        Returns:
            List of text chunks.
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)

        chunks: list[CodeChunk] = []
        current_chunk: list[str] = []
        current_length = 0
        chunk_index = 0
        max_chunk_size = 1500

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_length + len(para) > max_chunk_size and current_chunk:
                # Save current chunk
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        chunk_type=ChunkType.DOCUMENTATION,
                        name=f"{file_path.split('/')[-1]}:section_{chunk_index}",
                        start_line=1,  # Not tracking exact lines for plain text
                        end_line=1,
                        language="text",
                        file_path=file_path,
                    )
                )
                current_chunk = []
                current_length = 0
                chunk_index += 1

            current_chunk.append(para)
            current_length += len(para)

        # Save final chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(
                CodeChunk(
                    content=chunk_content,
                    chunk_type=ChunkType.DOCUMENTATION,
                    name=f"{file_path.split('/')[-1]}:section_{chunk_index}",
                    start_line=1,
                    end_line=1,
                    language="text",
                    file_path=file_path,
                )
            )

        if not chunks:
            chunks.append(
                CodeChunk(
                    content=content,
                    chunk_type=ChunkType.DOCUMENTATION,
                    name=file_path.split("/")[-1] if "/" in file_path else file_path,
                    start_line=1,
                    end_line=content.count("\n") + 1,
                    language="text",
                    file_path=file_path,
                )
            )

        return chunks
