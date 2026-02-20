"""AST-based code graph extraction for Python.

This module provides PythonGraphBuilder which parses Python files using tree-sitter
and extracts structural relationships into the FalkorDB graph store.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tree_sitter_language_pack import get_parser

if TYPE_CHECKING:
    from .graph_store import GraphStore

logger = logging.getLogger(__name__)


class PythonGraphBuilder:
    """Extract Python code relationships using tree-sitter.

    Parses Python files and creates graph nodes/edges for:
    - Files (with language and project_id)
    - Functions (with signatures, line numbers)
    - Classes (with line numbers)
    - Import relationships (IMPORTS)
    - Function call relationships (CALLS)
    - Class inheritance relationships (INHERITS)

    Attributes:
        graph: GraphStore instance for Cypher queries
        project_id: Project identifier for scoping
    """

    def __init__(self, graph: GraphStore, project_id: str) -> None:
        """Initialize the graph builder.

        Args:
            graph: GraphStore instance
            project_id: Project identifier
        """
        self.graph = graph
        self.project_id = project_id
        self._parser = get_parser("python")

    def index_file(self, file_path: Path) -> dict[str, int]:
        """Parse Python file and update graph.

        Args:
            file_path: Path to the Python file

        Returns:
            Statistics: {"functions": N, "classes": N, "imports": N, "calls": N}
        """
        content_bytes = file_path.read_bytes()
        try:
            tree = self._parser.parse(content_bytes)
        except Exception:
            logger.warning(f"Failed to parse {file_path}", exc_info=True)
            return {"functions": 0, "classes": 0, "imports": 0, "calls": 0}

        stats = {"functions": 0, "classes": 0, "imports": 0, "calls": 0}
        rel_path = str(file_path)

        # Track imported names to resolve calls: name -> module_path
        imported_symbols: dict[str, str] = {}

        # Create file node
        self._add_file(rel_path, "python")

        # Walk Tree
        self._visit_node(tree.root_node, rel_path, content_bytes, stats, imported_symbols)

        return stats

    def _visit_node(
        self,
        node: Any,
        rel_path: str,
        content_bytes: bytes,
        stats: dict[str, int],
        imported_symbols: dict[str, str],
    ) -> None:
        """Recursively visit nodes to extract graph elements."""
        if node.type in ("import_statement", "import_from_statement"):
            self._handle_import(node, rel_path, content_bytes, stats, imported_symbols)

        elif node.type in ("function_definition", "async_function_definition"):
            self._handle_function(node, rel_path, content_bytes, stats, imported_symbols)

        elif node.type == "class_definition":
            self._handle_class(node, rel_path, content_bytes, stats)

        # Recurse
        for child in node.children:
            self._visit_node(child, rel_path, content_bytes, stats, imported_symbols)

    def _handle_import(
        self,
        node: Any,
        rel_path: str,
        content_bytes: bytes,
        stats: dict[str, int],
        imported_symbols: dict[str, str],
    ) -> None:
        """Process import statements."""
        if node.type == "import_statement":
            # import os
            # import sys as system
            for child in node.children:
                if child.type == "dotted_name":
                    module_name = child.text.decode("utf-8")
                    self._add_import_ref(rel_path, module_name, stats, imported_symbols)
                elif child.type == "aliased_import":
                    # dotted_name as identifier
                    dotted_name = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    if dotted_name:
                        module_name = dotted_name.text.decode("utf-8")
                        alias_name = alias.text.decode("utf-8") if alias else module_name
                        self._add_import_ref(
                            rel_path, module_name, stats, imported_symbols, alias_name
                        )

        elif node.type == "import_from_statement":
            # from . import x
            # from .sub import x

            # Count leading dots
            dots = ""
            for child in node.children:
                if child.type == ".":
                    dots += "."
                elif child.type == "dotted_name":
                    # This is the module name part
                    break

            module_name_node = node.child_by_field_name("module_name")
            name_part = module_name_node.text.decode("utf-8") if module_name_node else ""

            module_name = dots + name_part

            # Fallback if structure is unexpected (e.g. no dotted_name field but has dots)
            if not module_name:
                parts = []
                for child in node.children:
                    if child.type == "import":
                        break
                    if child.type != "from":
                        parts.append(child.text.decode("utf-8"))
                module_name = "".join(parts)

            for child in node.children:
                if child.type == "dotted_name" and child != module_name_node:
                    # from x import y
                    name = child.text.decode("utf-8")
                    self._add_import_ref(rel_path, module_name, stats, imported_symbols, name)
                elif child.type == "aliased_import":
                    # from x import y as z
                    dotted_name = child.child_by_field_name("name")
                    alias = child.child_by_field_name("alias")
                    if dotted_name:
                        name = dotted_name.text.decode("utf-8")
                        alias_name = alias.text.decode("utf-8") if alias else name
                        self._add_import_ref(
                            rel_path, module_name, stats, imported_symbols, alias_name
                        )

    def _add_import_ref(
        self,
        rel_path: str,
        module_name: str,
        stats: dict[str, int],
        imported_symbols: dict[str, str],
        alias_name: str | None = None,
    ) -> None:
        """Helper to add import to graph and stats."""
        # If it's a 'from module import name', we might be importing a submodule or a member.
        # But broadly, we resolve 'module'.
        # For 'from . import x', module_name is '.'

        # If alias_name is provided (the name used in code), map it.
        # If not provided, use module_name (for 'import os').

        resolved_path = self._resolve_module_path(rel_path, module_name)
        self._add_import(rel_path, resolved_path)
        stats["imports"] += 1

        name_to_track = alias_name or module_name
        imported_symbols[name_to_track] = resolved_path

    def _handle_function(
        self,
        node: Any,
        rel_path: str,
        content_bytes: bytes,
        stats: dict[str, int],
        imported_symbols: dict[str, str],
    ) -> None:
        """Process function definitions."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        func_name = name_node.text.decode("utf-8")
        func_id = f"{rel_path}:{func_name}"

        is_async = node.type == "async_function_definition"
        if not is_async:
            # Check for 'async' keyword child node
            for child in node.children:
                if child.type == "async":
                    is_async = True
                    break

        # Determine start/end including decorators
        start_node = node
        if node.parent and node.parent.type == "decorated_definition":
            start_node = node.parent

        # Extract signature
        # Everything up to the colon (excluding decorators, but including async)
        # Using node.start_byte to exclude decorators
        colon_found = False
        sig_end = node.end_byte
        for child in node.children:
            if child.type == ":":
                sig_end = child.end_byte
                colon_found = True
                break

        if not colon_found:
            # Fallback: finding block
            body = node.child_by_field_name("body")
            if body:
                sig_end = body.start_byte

        signature = content_bytes[node.start_byte : sig_end].decode("utf-8").strip()

        self._add_function(
            func_id,
            func_name,
            rel_path,
            start_node.start_point[0] + 1,
            start_node.end_point[0] + 1,
            signature,
            is_async,
        )
        stats["functions"] += 1

        # Track calls within function (recursively in subtree)
        self._find_calls(node, func_id, rel_path, content_bytes, stats, imported_symbols)

    def _find_calls(
        self,
        node: Any,
        func_id: str,
        rel_path: str,
        content_bytes: bytes,
        stats: dict[str, int],
        imported_symbols: dict[str, str],
    ) -> None:
        """Recursively find calls in a function's body."""
        if node.type == "call":
            self._handle_call(node, func_id, rel_path, content_bytes, stats, imported_symbols)

        for child in node.children:
            self._find_calls(child, func_id, rel_path, content_bytes, stats, imported_symbols)

    def _handle_call(
        self,
        node: Any,
        caller_id: str,
        rel_path: str,
        content_bytes: bytes,
        stats: dict[str, int],
        imported_symbols: dict[str, str],
    ) -> None:
        """Process function calls."""
        func_node = node.child_by_field_name("function")
        if not func_node:
            return

        call_name = self._get_node_text(func_node, content_bytes)
        if call_name:
            self._add_call(caller_id, call_name, rel_path, imported_symbols)
            stats["calls"] += 1

    def _handle_class(
        self,
        node: Any,
        rel_path: str,
        content_bytes: bytes,
        stats: dict[str, int],
    ) -> None:
        """Process class definitions."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return

        class_name = name_node.text.decode("utf-8")
        class_id = f"{rel_path}:{class_name}"

        # Determine start/end including decorators
        start_node = node
        if node.parent and node.parent.type == "decorated_definition":
            start_node = node.parent

        self._add_class(
            class_id,
            class_name,
            rel_path,
            start_node.start_point[0] + 1,
            start_node.end_point[0] + 1,
        )
        stats["classes"] += 1

        # Track inheritance
        superclasses_node = node.child_by_field_name("superclasses")
        if superclasses_node:
            for child in superclasses_node.children:
                base_name = self._get_node_text(child, content_bytes)
                if base_name:
                    self._add_inheritance(class_id, base_name)

    def _get_node_text(self, node: Any, content_bytes: bytes) -> str | None:
        """Extract text from node."""
        if node.type == "identifier":
            return str(node.text.decode("utf-8"))
        elif node.type == "attribute":
            # For calls like obj.method(), we want 'method' usually?
            # AST: _get_call_name(ast.Attribute) -> node.attr
            attr = node.child_by_field_name("attribute")
            if attr:
                return str(attr.text.decode("utf-8"))
        return None

    def _resolve_module_path(self, from_file: str, module_name: str) -> str:
        """Resolve module name to absolute file path.

        Args:
            from_file: Importing file path
            module_name: Module to resolve

        Returns:
            Absolute path to the module file
        """
        from_path = Path(from_file)
        from_dir = from_path.parent

        if module_name.startswith("."):
            # Relative import
            dots = len(module_name) - len(module_name.lstrip("."))
            name_part = module_name.lstrip(".")

            target_dir = from_dir
            for _ in range(dots - 1):
                target_dir = target_dir.parent

            if not name_part:
                # from . import x -> module is the package
                module_path = target_dir / "__init__.py"
            else:
                module_path = (target_dir / name_part.replace(".", "/")).with_suffix(".py")
        else:
            # Absolute import (try relative to current dir first)
            module_path = (from_dir / module_name.replace(".", "/")).with_suffix(".py")

        return str(module_path)

    def _add_file(self, file_path: str, language: str) -> None:
        """Add file node to graph.

        Args:
            file_path: File path (used as identifier)
            language: Programming language
        """
        self.graph.query(
            """
            MERGE (f:File {path: $path})
            SET f.language = $language, f.project_id = $project_id
            """,
            {"path": file_path, "language": language, "project_id": self.project_id},
        )

    def _add_import(self, from_file: str, module_path: str) -> None:
        """Add import relationship.

        Args:
            from_file: File that contains the import (absolute path)
            module_path: Resolved absolute path of imported module
        """
        # Ensure target file node exists
        self.graph.query(
            """
            MERGE (f:File {path: $path})
            SET f.language = 'python', f.project_id = $project_id
            """,
            {"path": module_path, "project_id": self.project_id},
        )

        # Create import relationship
        self.graph.query(
            """
            MATCH (from:File {path: $from_path})
            MATCH (to:File {path: $to_path})
            MERGE (from)-[:IMPORTS]->(to)
            """,
            {"from_path": from_file, "to_path": module_path},
        )

    def _add_function(
        self,
        func_id: str,
        name: str,
        file_path: str,
        start_line: int,
        end_line: int,
        signature: str,
        is_async: bool = False,
    ) -> None:
        """Add function node.

        Args:
            func_id: Unique function identifier
            name: Function name
            file_path: Path to the containing file
            start_line: Start line number
            end_line: End line number
            signature: Function signature
            is_async: Whether function is async
        """
        self.graph.query(
            """
            MERGE (f:Function {id: $id})
            SET f.name = $name,
                f.file_path = $path,
                f.start_line = $start,
                f.end_line = $end,
                f.signature = $sig,
                f.async_func = $async_func
            """,
            {
                "id": func_id,
                "name": name,
                "path": file_path,
                "start": start_line,
                "end": end_line,
                "sig": signature,
                "async_func": is_async,
            },
        )

        # Connect to file
        self.graph.query(
            """
            MATCH (file:File {path: $path}), (func:Function {id: $id})
            MERGE (file)-[:DEFINES]->(func)
            """,
            {"path": file_path, "id": func_id},
        )

    def _add_class(
        self, class_id: str, name: str, file_path: str, start_line: int, end_line: int
    ) -> None:
        """Add class node.

        Args:
            class_id: Unique class identifier
            name: Class name
            file_path: Path to the containing file
            start_line: Start line
            end_line: End line
        """
        self.graph.query(
            """
            MERGE (c:Class {id: $id})
            SET c.name = $name,
                c.file_path = $path,
                c.start_line = $start,
                c.end_line = $end
            """,
            {
                "id": class_id,
                "name": name,
                "path": file_path,
                "start": start_line,
                "end": end_line,
            },
        )

        # Connect to file
        self.graph.query(
            """
            MATCH (file:File {path: $path}), (cls:Class {id: $id})
            MERGE (file)-[:DEFINES]->(cls)
            """,
            {"path": file_path, "id": class_id},
        )

    def _add_call(
        self, caller_id: str, callee_name: str, current_file: str, imports: dict[str, str]
    ) -> None:
        """Add function call relationship.

        Args:
            caller_id: ID of the calling function
            callee_name: Name of the called function
            current_file: Path of the file containing the call
            imports: Map of imported names to module paths
        """
        # Resolve callee ID
        if callee_name in imports:
            # Imported function: module_path:function_name
            callee_id = f"{imports[callee_name]}:{callee_name}"
        else:
            # Assumed local function: current_file:function_name
            callee_id = f"{current_file}:{callee_name}"

        # Create Check/Merge relationship using IDs
        # We use MERGE for the callee node to support forward references
        self.graph.query(
            """
            MATCH (caller:Function {id: $caller})
            MERGE (callee:Function {id: $callee_id})
            ON CREATE SET callee.name = $callee_name
            MERGE (caller)-[:CALLS]->(callee)
            """,
            {"caller": caller_id, "callee_id": callee_id, "callee_name": callee_name},
        )

    def _add_inheritance(self, child_id: str, parent_name: str) -> None:
        """Add class inheritance relationship.

        Args:
            child_id: ID of the child class
            parent_name: Name of the parent class
        """
        self.graph.query(
            """
            MATCH (child:Class {id: $child})
            MATCH (parent:Class)
            WHERE parent.name = $parent_name
            MERGE (child)-[:INHERITS]->(parent)
            """,
            {"child": child_id, "parent_name": parent_name},
        )
