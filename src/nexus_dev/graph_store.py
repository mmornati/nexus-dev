"""KùzuDB-based graph store for code structure and relationships."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import kuzu

logger = logging.getLogger(__name__)


class GraphStore:
    """KùzuDB-based graph store for code relationships.

    Manages the graph database schema and connections for:
    - Code structure (Files, Functions, Classes)
    - Relationships (DEFINES, IMPORTS, CALLS, INHERITS_FROM)

    Attributes:
        db_path: Path to KùzuDB database directory
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize Graph store.

        Args:
            db_path: Path to KùzuDB database directory
        """
        self.db_path = db_path
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    def connect(self) -> None:
        """Connect and initialize schema.

        Creates database directory and schema if they don't exist.
        """
        import kuzu

        # KùzuDB creates the directory automatically
        try:
            self._db = kuzu.Database(str(self.db_path))
            self._conn = kuzu.Connection(self._db)
            self._init_schema()
        except Exception as e:
            logger.error(f"Failed to connect to graph store at {self.db_path}: {e}")
            raise

    def _init_schema(self) -> None:
        """Create graph schema if not exist."""
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        # Node Tables
        # -----------

        # File Node
        # Stores information about source files
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS File (
                path STRING,
                language STRING,
                size INT64,
                last_modified STRING,
                PRIMARY KEY (path)
            )
        """)

        # Module Node
        # Stores Python modules / packages
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Module (
                name STRING,
                PRIMARY KEY (name)
            )
        """)

        # Class Node
        # Stores class definitions
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Class (
                id STRING,
                name STRING,
                start_line INT64,
                end_line INT64,
                PRIMARY KEY (id)
            )
        """)

        # Function Node
        # Stores function/method definitions
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Function (
                id STRING,
                name STRING,
                signature STRING,
                async_func BOOL,
                start_line INT64,
                end_line INT64,
                PRIMARY KEY (id)
            )
        """)

        # Relationship Tables (Edges)
        # ---------------------------

        # DEFINES: File -> Function/Class
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS DEFINES (
                FROM File TO Function,
                FROM File TO Class
            )
        """)

        # IMPORTS: File -> Module
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS IMPORTS (
                FROM File TO Module
            )
        """)

        # CALLS: Function -> Function
        # Captures static call graph
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CALLS (
                FROM Function TO Function
            )
        """)

        # INHERITS_FROM: Class -> Class
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS INHERITS_FROM (
                FROM Class TO Class
            )
        """)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Query parameters (not yet fully supported by Kùzu Python API in all versions)

        Returns:
            QueryResult
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        # Note: Kùzu Python API handling of params varies by version.
        # For safety in this MVP, we rely on the connection.execute method.
        # In production, ensure proper parameter binding to prevent injection.
        return self._conn.execute(cypher, params or {})

    def close(self) -> None:
        """Close database connection."""
        self._conn = None
        self._db = None

    def reset(self) -> None:
        """Delete the entire graph database (Dangerous!)."""
        self.close()
        if self.db_path.exists():
            shutil.rmtree(self.db_path)

    def __enter__(self) -> GraphStore:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
