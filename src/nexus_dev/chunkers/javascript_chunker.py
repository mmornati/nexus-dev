"""JavaScript/TypeScript code chunker using tree-sitter."""

from __future__ import annotations

from typing import Any

from tree_sitter_language_pack import get_parser

from .base import BaseChunker, ChunkType, CodeChunk


class JavaScriptChunker(BaseChunker):
    """Tree-sitter based chunker for JavaScript and TypeScript files.

    Extracts functions, classes, methods, and arrow functions as semantic chunks.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".js", ".jsx", ".mjs", ".cjs"]

    def __init__(self) -> None:
        """Initialize the JavaScript parser."""
        self._parser = get_parser("javascript")

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Parse JavaScript file and extract semantic chunks.

        Args:
            file_path: Path to the JavaScript file.
            content: File content.

        Returns:
            List of code chunks.
        """
        if not content.strip():
            return []

        try:
            tree = self._parser.parse(content.encode("utf-8"))
        except Exception:
            return [self._create_module_chunk(file_path, content, "javascript")]

        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        self._walk_tree(tree.root_node, lines, file_path, chunks, "javascript", parent=None)

        if not chunks:
            return [self._create_module_chunk(file_path, content, "javascript")]

        return chunks

    def _walk_tree(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        chunks: list[CodeChunk],
        language: str,
        parent: str | None,
    ) -> None:
        """Recursively walk the AST to find functions and classes.

        Args:
            node: Current AST node.
            lines: Source lines.
            file_path: File path.
            chunks: List to append chunks to.
            language: Language identifier.
            parent: Parent class name if inside a class.
        """
        if node.type == "function_declaration":
            chunk = self._extract_function(node, lines, file_path, language, parent)
            if chunk:
                chunks.append(chunk)

        elif node.type == "class_declaration":
            class_name = self._get_class_name(node)
            chunk = self._extract_class(node, lines, file_path, language)
            if chunk:
                chunks.append(chunk)

            # Extract methods
            for child in node.children:
                if child.type == "class_body":
                    for method in child.children:
                        if method.type == "method_definition":
                            method_chunk = self._extract_method(
                                method, lines, file_path, language, class_name
                            )
                            if method_chunk:
                                chunks.append(method_chunk)

        elif node.type == "lexical_declaration":
            # Handle const/let with arrow functions: const foo = () => {}
            chunk = self._extract_arrow_function(node, lines, file_path, language)
            if chunk:
                chunks.append(chunk)

        elif node.type == "variable_declaration":
            # Handle var with arrow functions
            chunk = self._extract_arrow_function(node, lines, file_path, language)
            if chunk:
                chunks.append(chunk)

        elif node.type == "export_statement":
            # Handle exported declarations
            for child in node.children:
                self._walk_tree(child, lines, file_path, chunks, language, parent)

        else:
            for child in node.children:
                self._walk_tree(child, lines, file_path, chunks, language, parent)

    def _get_function_name(self, node: Any) -> str:
        """Get function name from declaration."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return "anonymous"

    def _get_class_name(self, node: Any) -> str:
        """Get class name from declaration."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return "AnonymousClass"

    def _get_method_name(self, node: Any) -> str:
        """Get method name from definition."""
        for child in node.children:
            if child.type == "property_identifier":
                return child.text.decode("utf-8")
        return "anonymous"

    def _extract_function(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        language: str,
        parent: str | None,
    ) -> CodeChunk | None:
        """Extract a function declaration as a chunk."""
        try:
            name = self._get_function_name(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.FUNCTION,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language=language,
                file_path=file_path,
                parent=parent,
                signature=signature,
            )
        except Exception:
            return None

    def _extract_class(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        language: str,
    ) -> CodeChunk | None:
        """Extract a class as a chunk."""
        try:
            name = self._get_class_name(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.CLASS,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language=language,
                file_path=file_path,
                signature=signature,
            )
        except Exception:
            return None

    def _extract_method(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        language: str,
        parent: str,
    ) -> CodeChunk | None:
        """Extract a class method as a chunk."""
        try:
            name = self._get_method_name(node)
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "\n".join(lines[start_line : end_line + 1])
            signature = lines[start_line].strip()

            return CodeChunk(
                content=content,
                chunk_type=ChunkType.METHOD,
                name=name,
                start_line=start_line + 1,
                end_line=end_line + 1,
                language=language,
                file_path=file_path,
                parent=parent,
                signature=signature,
            )
        except Exception:
            return None

    def _extract_arrow_function(
        self,
        node: Any,
        lines: list[str],
        file_path: str,
        language: str,
    ) -> CodeChunk | None:
        """Extract an arrow function from const/let/var declaration."""
        try:
            # Look for pattern: const name = () => {} or const name = function() {}
            for child in node.children:
                if child.type == "variable_declarator":
                    name = None
                    has_function = False

                    for subchild in child.children:
                        if subchild.type == "identifier":
                            name = subchild.text.decode("utf-8")
                        elif subchild.type in ("arrow_function", "function_expression"):
                            has_function = True

                    if name and has_function:
                        start_line = node.start_point[0]
                        end_line = node.end_point[0]
                        content = "\n".join(lines[start_line : end_line + 1])
                        signature = lines[start_line].strip()

                        return CodeChunk(
                            content=content,
                            chunk_type=ChunkType.FUNCTION,
                            name=name,
                            start_line=start_line + 1,
                            end_line=end_line + 1,
                            language=language,
                            file_path=file_path,
                            signature=signature,
                        )
        except Exception:
            pass
        return None

    def _create_module_chunk(self, file_path: str, content: str, language: str) -> CodeChunk:
        """Create a module-level chunk for the entire file."""
        name = file_path.split("/")[-1] if "/" in file_path else file_path
        return CodeChunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            name=name,
            start_line=1,
            end_line=content.count("\n") + 1,
            language=language,
            file_path=file_path,
        )


class TypeScriptChunker(JavaScriptChunker):
    """Tree-sitter based chunker for TypeScript files.

    Inherits from JavaScriptChunker with TypeScript-specific parser.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".mts", ".cts"]

    def __init__(self) -> None:
        """Initialize the TypeScript parser."""
        self._parser = get_parser("typescript")

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Parse TypeScript file and extract semantic chunks."""
        if not content.strip():
            return []

        try:
            tree = self._parser.parse(content.encode("utf-8"))
        except Exception:
            return [self._create_module_chunk(file_path, content, "typescript")]

        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        self._walk_tree(tree.root_node, lines, file_path, chunks, "typescript", parent=None)

        if not chunks:
            return [self._create_module_chunk(file_path, content, "typescript")]

        return chunks
