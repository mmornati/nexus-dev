"""JavaScript/TypeScript code graph extraction."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_store import GraphStore

logger = logging.getLogger(__name__)


class JSGraphBuilder:
    """Extract JS/TS code relationships using regex patterns.

    Note: For production use, consider tree-sitter-javascript.
    This implementation uses regex for simplicity as per Phase 2 requirements.
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
        """Parse JS/TS file and update graph.

        Args:
            file_path: Path to the file

        Returns:
            Statistics: {"functions": N, "classes": N, "imports": N}
        """
        content = file_path.read_text(encoding="utf-8")
        ext = file_path.suffix.lower()
        language = "typescript" if ext in (".ts", ".tsx") else "javascript"

        stats = {"functions": 0, "classes": 0, "imports": 0}
        rel_path = str(file_path)

        # Create file node
        self._add_file(rel_path, language)

        # Extract imports
        # import X from 'module'
        # import { X } from 'module'
        # const X = require('module')
        import_patterns = [
            r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
            r"import\s+['\"]([^'\"]+)['\"]",
            r"require\(['\"]([^'\"]+)['\"]\)",
        ]

        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                module_name = match.group(1)
                self._add_import(rel_path, module_name)
                stats["imports"] += 1

        # Extract functions
        # function name() {}
        # const name = () => {}
        # const name = function() {}
        # async function name() {}
        func_patterns = [
            r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            r"(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)",
        ]

        for pattern in func_patterns:
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                self._add_function(rel_path, func_name, match.start())
                stats["functions"] += 1

        # Extract classes
        class_pattern = r"class\s+([A-Z][a-zA-Z0-9]*)(?:\s+extends\s+([A-Z][a-zA-Z0-9]*))?\s*\{"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            parent_name = match.group(2)
            self._add_class(rel_path, class_name, parent_name)
            stats["classes"] += 1

        return stats

    def _add_file(self, file_path: str, language: str) -> None:
        """Add file node to graph."""
        self.graph.query(
            """
            MERGE (f:File {path: $path})
            SET f.language = $language, f.project_id = $project_id
            """,
            {"path": file_path, "language": language, "project_id": self.project_id},
        )

    def _add_import(self, from_file: str, module_name: str) -> None:
        """Add import relationship."""
        # Resolve relative imports
        if module_name.startswith("."):
            # Relative import - convert to file path guess
            base_dir = Path(from_file).parent
            # Just a best-effort guess at the resolved path
            # For now, let's just use the strict path logic from the proposal
            module_path = str((base_dir / module_name).with_suffix(".js"))
        else:
            # Node module or absolute
            module_path = f"node_modules/{module_name}"

        # Create target file node (placeholder)
        self.graph.query(
            """
            MERGE (f:File {path: $path})
            SET f.project_id = $project_id
            """,
            {"path": module_path, "project_id": self.project_id},
        )

        self.graph.query(
            """
            MATCH (from:File {path: $from_path})
            MATCH (to:File {path: $to_path})
            MERGE (from)-[:IMPORTS]->(to)
            """,
            {"from_path": from_file, "to_path": module_path},
        )

    def _add_function(self, file_path: str, func_name: str, char_pos: int) -> None:
        """Add function node."""
        func_id = f"{file_path}:{func_name}"

        self.graph.query(
            """
            MERGE (f:Function {id: $id})
            SET f.name = $name, f.file_path = $path
            """,
            {"id": func_id, "name": func_name, "path": file_path},
        )

        self.graph.query(
            """
            MATCH (file:File {path: $path}), (func:Function {id: $id})
            MERGE (file)-[:DEFINES]->(func)
            """,
            {"path": file_path, "id": func_id},
        )

    def _add_class(self, file_path: str, class_name: str, parent_name: str | None) -> None:
        """Add class node and inheritance."""
        class_id = f"{file_path}:{class_name}"

        self.graph.query(
            """
            MERGE (c:Class {id: $id})
            SET c.name = $name, f.file_path = $path
            """,
            {"id": class_id, "name": class_name, "path": file_path},
        )

        self.graph.query(
            """
            MATCH (file:File {path: $path}), (cls:Class {id: $id})
            MERGE (file)-[:DEFINES]->(cls)
            """,
            {"path": file_path, "id": class_id},
        )

        if parent_name:
            self.graph.query(
                """
                MATCH (child:Class {id: $child})
                MATCH (parent:Class) WHERE parent.name = $parent_name
                MERGE (child)-[:INHERITS]->(parent)
                """,
                {"child": class_id, "parent_name": parent_name},
            )
