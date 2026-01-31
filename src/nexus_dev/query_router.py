"""Query router for intelligent search dispatch.

This module provides the HybridQueryRouter which analyzes user queries to determine
the most appropriate search strategy (Graph, KV, or Vector) and extracts relevant
entities (file names, function names) to optimize the search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class QueryType(Enum):
    """Type of query intent."""

    VECTOR = "vector"  # Semantic search (default)
    GRAPH = "graph"  # Structural queries (dependencies, callers)
    KV = "kv"  # Session context/history


@dataclass
class QueryIntent:
    """Analyzed query intent."""

    query_type: QueryType
    extracted_entity: str | None = None
    original_query: str = ""
    confidence: float = 1.0


class HybridQueryRouter:
    """Routes queries to the appropriate database engine."""

    # Regex patterns for intent detection
    # Each tuple is (Category, List[Patterns])

    GRAPH_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        # Callers
        (r"who calls\s+([a-zA-Z0-9_.]+)", "callers"),
        (r"callers of\s+([a-zA-Z0-9_.]+)", "callers"),
        (r"where is\s+([a-zA-Z0-9_.]+)\s+called", "callers"),
        # Implementations/Inheritance
        (r"who implements\s+([a-zA-Z0-9_.]+)", "implementations"),
        (r"implementations of\s+([a-zA-Z0-9_.]+)", "implementations"),
        (r"subclasses of\s+([a-zA-Z0-9_.]+)", "implementations"),
        (r"classes extending\s+([a-zA-Z0-9_.]+)", "implementations"),
        # Dependencies (Imports)
        (r"what imports\s+([a-zA-Z0-9_./]+)", "imported_by"),
        (r"who imports\s+([a-zA-Z0-9_./]+)", "imported_by"),
        (r"imports of\s+([a-zA-Z0-9_./]+)", "imports"),
        (r"dependencies of\s+([a-zA-Z0-9_./]+)", "imports"),
        (r"what does\s+([a-zA-Z0-9_./]+)\s+import", "imports"),
    ]

    KV_PATTERNS: ClassVar[list[str]] = [
        r"last message",
        r"previous message",
        r"recent context",
        r"session history",
        r"conversation history",
        r"what did we just do",
        r"what did I just ask",
        r"summarize the session",
    ]

    def route(self, query: str) -> QueryIntent:
        """Analyze query and return routing plan.

        Args:
            query: User search query.

        Returns:
            QueryIntent with classification and extraction.
        """
        query_lower = query.lower()

        # 1. Check Graph Patterns (Entity extraction)
        for pattern, _category in self.GRAPH_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entity = match.group(1)
                return QueryIntent(
                    query_type=QueryType.GRAPH,
                    extracted_entity=entity,
                    original_query=query,
                    confidence=0.9,
                )

        # 2. Check KV Patterns (Session context)
        for pattern in self.KV_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryIntent(query_type=QueryType.KV, original_query=query, confidence=0.85)

        # 3. Default to Vector (Semantic Search)
        return QueryIntent(query_type=QueryType.VECTOR, original_query=query, confidence=0.5)
