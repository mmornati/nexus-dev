"""Java code chunker using tree-sitter."""

from __future__ import annotations

from typing import Any

from tree_sitter_language_pack import get_parser

from .base import BaseChunker, ChunkType, CodeChunk


class JavaChunker(BaseChunker):
    """Tree-sitter based chunker for Java files.

    Extracts classes, interfaces, methods, and constructors as semantic chunks.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".java"]

    def __init__(self) -> None:
        """Initialize the Java parser."""
        self._parser = get_parser("java")

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Parse Java file and extract semantic chunks.

        Args:
            file_path: Path to the Java file.
            content: File content.

        Returns:
            List of code chunks.
        """
        if not content.strip():
            return []

        try:
            tree = self._parser.parse(content.encode("utf-8"))
        except Exception:
            return [self._create_module_chunk(file_path, content)]

        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        # Extract package and imports for context
        imports = self._extract_imports(tree.root_node, lines)

        self._walk_tree(tree.root_node, lines, file_path, chunks, imports, parent=None)

        if not chunks:
            return [self._create_module_chunk(file_path, content)]

        return chunks

    def _walk_tree(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        chunks: list[CodeChunk],
        imports: list[str],
        parent: str | None,
    ) -> None:
        """Recursively walk the AST to find classes and methods.

        Args:
            node: Current AST node.
            lines: Source lines.
            file_path: File path.
            chunks: List to append chunks to.
            imports: Package and import statements.
            parent: Parent class name if inside a class.
        """
        if node.type == "class_declaration":
            class_name = self._get_identifier(node)

            # Extract the whole class
            chunk = self._extract_class(node, lines, file_path, imports)
            if chunk:
                chunks.append(chunk)

            # Extract methods and constructors
            for child in node.children:
                if child.type == "class_body":
                    for member in child.children:
                        if member.type == "method_declaration":
                            method_chunk = self._extract_method(
                                member, lines, file_path, imports, class_name
                            )
                            if method_chunk:
                                chunks.append(method_chunk)
                        elif member.type == "constructor_declaration":
                            constructor_chunk = self._extract_constructor(
                                member, lines, file_path, imports, class_name
                            )
                            if constructor_chunk:
                                chunks.append(constructor_chunk)
                        # Recurse for inner classes
                        elif member.type == "class_declaration":
                            self._walk_tree(member, lines, file_path, chunks, imports, class_name)

        elif node.type == "interface_declaration":
            chunk = self._extract_interface(node, lines, file_path, imports)
            if chunk:
                chunks.append(chunk)

        elif node.type == "enum_declaration":
            chunk = self._extract_enum(node, lines, file_path, imports)
            if chunk:
                chunks.append(chunk)

        else:
            for child in node.children:
                self._walk_tree(child, lines, file_path, chunks, imports, parent)

    def _extract_imports(self, root: Any, lines: list[str]) -> list[str]:
        """Extract package and import statements.

        Args:
            root: Root AST node.
            lines: Source lines.

        Returns:
            List of package and import statements.
        """
        imports = []
        for child in root.children:
            if child.type == "package_declaration" or child.type == "import_declaration":
                start = child.start_point[0]
                end = child.end_point[0]
                imports.append("\n".join(lines[start : end + 1]))
        return imports

    def _get_identifier(self, node: Any) -> str:
        """Get identifier (name) from a declaration."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return "Unknown"

    def _get_javadoc(self, node: Any, lines: list[str]) -> str | None:
        """Extract Javadoc comment if present before the node.

        Args:
            node: AST node.
            lines: Source lines.

        Returns:
            Javadoc text or None.
        """
        start_line = node.start_point[0]
        if start_line == 0:
            return None

        # Look for block comment ending just before this node
        for i in range(start_line - 1, max(-1, start_line - 20), -1):
            line = lines[i].strip()
            if line.endswith("*/"):
                # Found end of comment, find start
                doc_lines: list[str] = []
                for j in range(i, max(-1, i - 50), -1):
                    doc_lines.insert(0, lines[j])
                    if lines[j].strip().startswith("/**"):
                        return "\n".join(doc_lines)
            elif line and not line.startswith("*") and not line.startswith("@"):
                break

        return None

    def _extract_class(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        imports: list[str],
    ) -> CodeChunk | None:
        """Extract a class as a chunk."""
        try:
            name = self._get_identifier(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            docstring = self._get_javadoc(node, lines)

            # Build signature from first line
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.CLASS,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language="java",
                file_path=file_path,
                docstring=docstring,
                imports=imports[:5],
                signature=signature,
            )
        except Exception:
            return None

    def _extract_interface(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        imports: list[str],
    ) -> CodeChunk | None:
        """Extract an interface as a chunk."""
        try:
            name = self._get_identifier(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            docstring = self._get_javadoc(node, lines)
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.CLASS,  # Treat interface as class type
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language="java",
                file_path=file_path,
                docstring=docstring,
                imports=imports[:5],
                signature=signature,
            )
        except Exception:
            return None

    def _extract_enum(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        imports: list[str],
    ) -> CodeChunk | None:
        """Extract an enum as a chunk."""
        try:
            name = self._get_identifier(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            docstring = self._get_javadoc(node, lines)
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.CLASS,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language="java",
                file_path=file_path,
                docstring=docstring,
                imports=imports[:5],
                signature=signature,
            )
        except Exception:
            return None

    def _extract_method(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        imports: list[str],
        parent: str,
    ) -> CodeChunk | None:
        """Extract a method as a chunk."""
        try:
            name = self._get_identifier(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            docstring = self._get_javadoc(node, lines)
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.METHOD,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language="java",
                file_path=file_path,
                parent=parent,
                docstring=docstring,
                imports=imports[:3],
                signature=signature,
            )
        except Exception:
            return None

    def _extract_constructor(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        imports: list[str],
        parent: str,
    ) -> CodeChunk | None:
        """Extract a constructor as a chunk."""
        try:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            docstring = self._get_javadoc(node, lines)
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.METHOD,
                name=f"{parent}",  # Constructor has same name as class
                start_line=start_line + 1,
                end_line=end_line + 1,
                language="java",
                file_path=file_path,
                parent=parent,
                docstring=docstring,
                imports=imports[:3],
                signature=signature,
            )
        except Exception:
            return None

    def _create_module_chunk(self, file_path: str, content: str) -> CodeChunk:
        """Create a module-level chunk for the entire file."""
        name = file_path.split("/")[-1] if "/" in file_path else file_path
        return CodeChunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            name=name,
            start_line=1,
            end_line=content.count("\n") + 1,
            language="java",
            file_path=file_path,
        )
