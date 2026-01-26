"""Hybrid database manager coordinating KV, Vector, and Graph stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import kuzu

    from .config import NexusConfig
    from .database import NexusDatabase

from .kv_store import KVStore


class HybridDatabase:
    """Coordinates SQLite (KV), LanceDB (Vector), and KùzuDB (Graph).

    This class provides a unified interface to three complementary database systems:
    - SQLite (KV): Fast exact lookups for session state and chat history
    - LanceDB (Vector): Semantic search via embeddings (existing)
    - KùzuDB (Graph): Code relationships and dependency graphs

    Attributes:
        config: Nexus-Dev configuration
    """

    def __init__(self, config: NexusConfig) -> None:
        """Initialize hybrid database manager.

        Args:
            config: Nexus-Dev configuration
        """
        self.config = config
        self._kv_store: KVStore | None = None
        self._vector: NexusDatabase | None = None
        self._graph_db: kuzu.Database | None = None
        self._graph_conn: kuzu.Connection | None = None

    def connect(self) -> None:
        """Initialize all database connections.

        Creates database directories and initializes schemas if needed.
        Only connects to enabled databases.
        """
        if not self.config.enable_hybrid_db:
            return

        db_path = self.config.get_db_path()

        # KV Store (SQLite)
        kv_path = db_path / "state.db"
        self._kv_store = KVStore(kv_path)
        self._kv_store.connect()

        # Graph Store (KùzuDB)
        # KùzuDB creates the directory automatically - don't mkdir first
        graph_path = db_path / "graph_db"

        # Lazy import kuzu to avoid dependency when hybrid mode is disabled
        import kuzu

        self._graph_db = kuzu.Database(str(graph_path))
        self._graph_conn = kuzu.Connection(self._graph_db)
        self._init_graph_schema()

    def _init_graph_schema(self) -> None:
        """Create graph schema if not exist.

        Note: Full schema implementation will be added in Phase 2 (#59).
        For now, just verifies connection works.
        """
        # Placeholder - will be implemented in #59
        pass

    @property
    def kv(self) -> KVStore:
        """Get KV store.

        Returns:
            KVStore instance

        Raises:
            RuntimeError: If hybrid mode is not enabled
        """
        if not self.config.enable_hybrid_db:
            raise RuntimeError(
                "Hybrid database is not enabled. Set enable_hybrid_db=True in config."
            )

        if self._kv_store is None:
            self.connect()

        if self._kv_store is None:
            raise RuntimeError("Failed to initialize KV store")

        return self._kv_store

    @property
    def graph(self) -> Any:  # kuzu.Connection
        """Get graph connection.

        Returns:
            KùzuDB connection

        Raises:
            RuntimeError: If hybrid mode is not enabled
        """
        if not self.config.enable_hybrid_db:
            raise RuntimeError(
                "Hybrid database is not enabled. Set enable_hybrid_db=True in config."
            )

        if self._graph_conn is None:
            self.connect()

        if self._graph_conn is None:
            raise RuntimeError("Failed to initialize graph store")

        return self._graph_conn

    def close(self) -> None:
        """Close all database connections."""
        if self._kv_store:
            self._kv_store.close()
            self._kv_store = None

        if self._graph_conn:
            self._graph_conn = None

        if self._graph_db:
            self._graph_db = None

    def __enter__(self) -> HybridDatabase:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
