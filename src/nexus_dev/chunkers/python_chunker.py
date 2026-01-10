"""Python code chunker using tree-sitter."""

from __future__ import annotations

from typing import Any

from tree_sitter_language_pack import get_parser

from .base import BaseChunker, ChunkType, CodeChunk


class PythonChunker(BaseChunker):
    """Tree-sitter based chunker for Python files.

    Extracts functions, classes, and methods as semantic chunks.
    Preserves docstrings and signatures for better search quality.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".py", ".pyw"]

    def __init__(self) -> None:
        """Initialize the Python parser."""
        self._parser = get_parser("python")

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Parse Python file and extract semantic chunks.

        Args:
            file_path: Path to the Python file.
            content: File content.

        Returns:
            List of code chunks (functions, classes, methods).
        """
        if not content.strip():
            return []

        try:
            tree = self._parser.parse(content.encode("utf-8"))
        except Exception:
            # Fall back to returning whole file as single chunk
            return [self._create_module_chunk(file_path, content)]

        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        # Extract imports for context
        imports = self._extract_imports(tree.root_node, lines)

        # Walk the tree to find functions and classes
        self._walk_tree(tree.root_node, lines, file_path, chunks, imports, parent=None)

        # If no chunks found, return whole file as module
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
        """Recursively walk the AST to find functions and classes.

        Args:
            node: Current AST node.
            lines: Source lines.
            file_path: File path.
            chunks: List to append chunks to.
            imports: List of import statements.
            parent: Parent class name if inside a class.
        """
        if node.type == "function_definition" or node.type == "async_function_definition":
            chunk = self._extract_function(node, lines, file_path, imports, parent)
            if chunk:
                chunks.append(chunk)

        elif node.type == "class_definition":
            class_name = self._get_node_name(node)

            # Extract the whole class as one chunk
            chunk = self._extract_class(node, lines, file_path, imports)
            if chunk:
                chunks.append(chunk)

            # Also extract individual methods
            for child in node.children:
                if child.type == "block":
                    for block_child in child.children:
                        if block_child.type in ("function_definition", "async_function_definition"):
                            method_chunk = self._extract_function(
                                block_child, lines, file_path, imports, class_name
                            )
                            if method_chunk:
                                chunks.append(method_chunk)

        else:
            # Recurse into other nodes
            for child in node.children:
                self._walk_tree(child, lines, file_path, chunks, imports, parent)

    def _extract_imports(self, root: Any, lines: list[str]) -> list[str]:
        """Extract import statements from the module.

        Args:
            root: Root AST node.
            lines: Source lines.

        Returns:
            List of import statements.
        """
        imports = []
        for child in root.children:
            if child.type in ("import_statement", "import_from_statement"):
                start = child.start_point[0]
                end = child.end_point[0]
                import_text = "\n".join(lines[start : end + 1])
                imports.append(import_text)
        return imports

    def _get_node_name(self, node: Any) -> str:
        """Get the name from a function or class definition.

        Args:
            node: AST node.

        Returns:
            Name of the function/class.
        """
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return "unknown"

    def _get_docstring(self, node: Any, lines: list[str]) -> str | None:
        """Extract docstring from a function or class.

        Args:
            node: AST node.
            lines: Source lines.

        Returns:
            Docstring text or None.
        """
        for child in node.children:
            if child.type == "block":
                for block_child in child.children:
                    if block_child.type == "expression_statement":
                        for expr_child in block_child.children:
                            if expr_child.type == "string":
                                start = expr_child.start_point[0]
                                end = expr_child.end_point[0]
                                docstring = "\n".join(lines[start : end + 1])
                                # Clean up the docstring (remove triple quotes)
                                docstring = docstring.strip()
                                for quote in ('"""', "'''"):
                                    if docstring.startswith(quote):
                                        docstring = docstring[3:]
                                    if docstring.endswith(quote):
                                        docstring = docstring[:-3]
                                return docstring.strip()
                        break
                break
        return None

    def _get_signature(self, node: Any, lines: list[str]) -> str:
        """Extract function/method signature.

        Args:
            node: AST node.
            lines: Source lines.

        Returns:
            Signature string.
        """
        # Get the first line(s) up to the colon
        start_line = node.start_point[0]
        sig_parts = []
        for i in range(start_line, min(start_line + 5, len(lines))):
            line = lines[i]
            sig_parts.append(line)
            if ":" in line and not line.strip().endswith(":"):
                continue
            if line.rstrip().endswith(":"):
                break

        signature = " ".join(sig_parts)
        # Clean up
        if ":" in signature:
            signature = signature[: signature.rfind(":") + 1]
        return signature.strip()

    def _extract_function(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        imports: list[str],
        parent: str | None,
    ) -> CodeChunk | None:
        """Extract a function as a chunk.

        Args:
            node: Function AST node.
            lines: Source lines.
            file_path: File path.
            imports: Import statements.
            parent: Parent class name if method.

        Returns:
            CodeChunk or None if extraction failed.
        """
        try:
            name = self._get_node_name(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            content = "\n".join(lines[start_line : end_line + 1])
            docstring = self._get_docstring(node, lines)
            signature = self._get_signature(node, lines)

            chunk_type = ChunkType.METHOD if parent else ChunkType.FUNCTION

            return CodeChunk(
                content=content,
                chunk_type=chunk_type,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language="python",
                file_path=file_path,
                parent=parent,
                docstring=docstring,
                imports=imports[:5],  # Limit imports for context
                signature=signature,
            )
        except Exception:
            return None

    def _extract_class(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        imports: list[str],
    ) -> CodeChunk | None:
        """Extract a class as a chunk.

        Args:
            node: Class AST node.
            lines: Source lines.
            file_path: File path.
            imports: Import statements.

        Returns:
            CodeChunk or None if extraction failed.
        """
        try:
            name = self._get_node_name(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            content = "\n".join(lines[start_line : end_line + 1])
            docstring = self._get_docstring(node, lines)
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.CLASS,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language="python",
                file_path=file_path,
                docstring=docstring,
                imports=imports[:5],
                signature=signature,
            )
        except Exception:
            return None

    def _create_module_chunk(self, file_path: str, content: str) -> CodeChunk:
        """Create a module-level chunk for the entire file.

        Args:
            file_path: File path.
            content: File content.

        Returns:
            Module chunk.
        """
        name = file_path.split("/")[-1] if "/" in file_path else file_path
        return CodeChunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            name=name,
            start_line=1,
            end_line=content.count("\n") + 1,
            language="python",
            file_path=file_path,
        )
