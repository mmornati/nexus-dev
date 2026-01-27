"""FalkorDB-based graph store for code structure and relationships."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redislite import FalkorDB

logger = logging.getLogger(__name__)


class GraphStore:
    """FalkorDB-based graph store for code relationships.

    Manages the graph database schema and connections for:
    - Code structure (Files, Functions, Classes)
    - Relationships (DEFINES, IMPORTS, CALLS, INHERITS_FROM)

    Attributes:
        client: FalkorDB client (Redis client with graph capabilities)
        graph_name: Name of the graph key in Redis
    """

    def __init__(self, client: FalkorDB, graph_name: str = "nexus_code_graph") -> None:
        """Initialize Graph store.

        Args:
            client: FalkorDB client
            graph_name: Key for the graph data
        """
        self.client = client
        self.graph_name = graph_name
        self._graph = client.select_graph(graph_name)

    def connect(self) -> None:
        """Initialize schema.

        FalkorDB is schema-less but supports indices.
        """
        self._init_schema()

    def _init_schema(self) -> None:
        """Create graph indices."""
        # Create indices for faster lookup
        indices = [
            ("File", "path"),
            ("Module", "name"),
            ("Class", "id"),
            ("Function", "id"),
        ]

        for label, prop in indices:
            try:
                # This creates an index on :Label(prop)
                self._graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
            except Exception as e:
                # Indices might already exist
                logger.debug(f"Index creation for {label}.{prop} returned: {e}")

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            QueryResult
        """
        return self._graph.query(cypher, params or {})

    def close(self) -> None:
        """Close database connection."""
        pass

    def delete_graph(self) -> None:
        """Delete the entire graph."""
        try:
            self._graph.delete()
        except Exception:
            pass

    def __enter__(self) -> GraphStore:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
