"""AST-based code graph extraction for Python.

This module provides PythonGraphBuilder which parses Python files using AST
and extracts structural relationships into the FalkorDB graph store.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore

logger = logging.getLogger(__name__)


class PythonGraphBuilder:
    """Extract Python code relationships using AST.

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

    def index_file(self, file_path: Path) -> dict[str, int]:
        """Parse Python file and update graph.

        Args:
            file_path: Path to the Python file

        Returns:
            Statistics: {"functions": N, "classes": N, "imports": N, "calls": N}
        """
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        stats = {"functions": 0, "classes": 0, "imports": 0, "calls": 0}
        rel_path = str(file_path)

        # Track imported names to resolve calls: name -> module_path
        imported_symbols: dict[str, str] = {}

        # Create file node
        self._add_file(rel_path, "python")

        # Walk AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_path = self._resolve_module_path(rel_path, alias.name)
                    self._add_import(rel_path, module_path)
                    stats["imports"] += 1
                    # Track imported names
                    name = alias.asname or alias.name
                    imported_symbols[name] = module_path
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_path = self._resolve_module_path(rel_path, node.module)
                    self._add_import(rel_path, module_path)
                    stats["imports"] += 1
                    # Track imported names
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imported_symbols[name] = module_path
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                func_id = f"{rel_path}:{node.name}"
                is_async = isinstance(node, ast.AsyncFunctionDef)
                self._add_function(func_id, node, rel_path, is_async)
                stats["functions"] += 1

                # Track calls within function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        call_name = self._get_call_name(child)
                        if call_name:
                            self._add_call(func_id, call_name, rel_path, imported_symbols)
                            stats["calls"] += 1
            elif isinstance(node, ast.ClassDef):
                class_id = f"{rel_path}:{node.name}"
                self._add_class(class_id, node, rel_path)
                stats["classes"] += 1

                # Track inheritance
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name:
                        self._add_inheritance(class_id, base_name)

        return stats

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
            module_path = (from_dir / module_name.lstrip(".")).with_suffix(".py")
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
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        is_async: bool = False,
    ) -> None:
        """Add function node.

        Args:
            func_id: Unique function identifier
            node: AST node for the function
            file_path: Path to the containing file
            is_async: Whether function is async
        """
        # Build signature
        args = [arg.arg for arg in node.args.args]
        signature = f"{'async ' if is_async else ''}def {node.name}({', '.join(args)})"

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
                "name": node.name,
                "path": file_path,
                "start": node.lineno,
                "end": node.end_lineno or node.lineno,
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

    def _add_class(self, class_id: str, node: ast.ClassDef, file_path: str) -> None:
        """Add class node.

        Args:
            class_id: Unique class identifier
            node: AST node for the class
            file_path: Path to the containing file
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
                "name": node.name,
                "path": file_path,
                "start": node.lineno,
                "end": node.end_lineno or node.lineno,
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

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract function name from Call node.

        Args:
            node: AST Call node

        Returns:
            Function name or None if cannot be determined
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _get_name(self, node: ast.expr) -> str | None:
        """Extract name from AST expression.

        Args:
            node: AST expression node

        Returns:
            Name string or None if cannot be determined
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None
