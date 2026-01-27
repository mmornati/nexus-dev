"""Hybrid database manager coordinating KV, Vector, and Graph stores."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redislite import FalkorDB

    from .config import NexusConfig

from .graph_store import GraphStore
from .kv_store import KVStore

logger = logging.getLogger(__name__)


class HybridDatabase:
    """Coordinates Redis (KV + Graph) and LanceDB (Vector).

    This class provides a unified interface to the hybrid database system:
    - FalkorDBLite (Redis):
        - KV: Fast exact lookups for session state and chat history
        - Graph: Code relationships and dependency graphs (Cypher)
    - LanceDB: Semantic search via embeddings (existing)

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
        self._graph_store: GraphStore | None = None
        self._falkor_db: FalkorDB | None = None

    def connect(self) -> None:
        """Initialize all database connections.

        Creates database directories and initializes schemas if needed.
        Only connects to enabled databases.
        """
        if not self.config.enable_hybrid_db:
            return

        # Check if already connected
        if self._kv_store is not None:
            return

        db_path = self.config.get_db_path()
        # Ensure db directory exists
        db_path.mkdir(parents=True, exist_ok=True)

        # Initialize FalkorDBLite (Redis + Graph)
        try:
            # Monkeypatch redislite/redis compatibility issues
            # redislite passes 'dir' and other args to redis.Redis, which Redis 5+ rejects
            try:
                import redis
                import redislite.client

                # Fix AttributeError in __del__
                original_cleanup = redislite.client.RedisMixin._cleanup

                def patched_cleanup(self: Any, *args: Any, **kwargs: Any) -> None:
                    try:
                        original_cleanup(self, *args, **kwargs)
                    except AttributeError:
                        pass
                    except Exception:
                        pass

                redislite.client.RedisMixin._cleanup = patched_cleanup

                # Fix TypeError in __init__
                original_redis_init = redis.Redis.__init__

                def patched_redis_init(self: Any, *args: Any, **kwargs: Any) -> None:
                    # Remove arguments that redislite passes but redis doesn't accept
                    kwargs.pop("dir", None)
                    kwargs.pop("dbfilename", None)
                    kwargs.pop("serverconfig", None)

                    original_redis_init(self, *args, **kwargs)

                redis.Redis.__init__ = patched_redis_init  # type: ignore[method-assign]
                redis.StrictRedis.__init__ = patched_redis_init  # type: ignore[method-assign]

            except ImportError:
                pass

            from redislite import FalkorDB

            # FalkorDBLite manages the server process in the specified directory
            self._falkor_db = FalkorDB(dir=str(db_path))

            # Initialize stores
            # KVStore needs standard Redis commands -> use .client
            self._kv_store = KVStore(self._falkor_db.client)
            # GraphStore needs Graph commands -> use FalkorDB object
            self._graph_store = GraphStore(self._falkor_db)

            # Initialize schemas/indices
            self._kv_store.connect()
            self._graph_store.connect()

        except Exception as e:
            logger.error(f"Failed to initialize FalkorDBLite: {e}")
            raise

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
    def graph(self) -> GraphStore:
        """Get graph store.

        Returns:
            GraphStore instance

        Raises:
            RuntimeError: If hybrid mode is not enabled
        """
        if not self.config.enable_hybrid_db:
            raise RuntimeError(
                "Hybrid database is not enabled. Set enable_hybrid_db=True in config."
            )

        if self._graph_store is None:
            self.connect()

        if self._graph_store is None:
            raise RuntimeError("Failed to initialize graph store")

        return self._graph_store

    def close(self) -> None:
        """Close all database connections."""
        if self._kv_store:
            self._kv_store.close()
            self._kv_store = None

        if self._graph_store:
            self._graph_store.close()
            self._graph_store = None

        if self._falkor_db:
            try:
                self._falkor_db.close()
            except Exception:
                pass
            self._falkor_db = None
            self._falkor_db = None

    def __enter__(self) -> HybridDatabase:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
