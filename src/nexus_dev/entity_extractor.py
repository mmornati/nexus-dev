"""LLM-based entity extraction from conversations.

This module provides regex-based entity extraction for code-related entities
from conversation text, and stores entity relationships in the FalkorDB graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_store import GraphStore


@dataclass
class ExtractedEntity:
    """An entity extracted from text.

    Attributes:
        name: The entity name (e.g., "auth.py", "UserController")
        entity_type: Type of entity: 'file', 'function', 'class', 'error'
        confidence: Extraction confidence (0.0-1.0)
    """

    name: str
    entity_type: str  # 'file', 'function', 'class', 'error'
    confidence: float


class EntityExtractor:
    """Extract entities from text using regex patterns.

    Provides pattern-based extraction for common code entities:
    - Files: Matches file paths with code extensions
    - Functions: Matches function call patterns (name followed by parenthesis)
    - Classes: Matches PascalCase identifiers
    - Errors: Matches common error/exception types
    """

    # Regex patterns for common code entities
    # Note: Use non-capturing groups (?:...) when you don't want group(1) to match it
    PATTERNS: dict[str, str] = {
        "file": r"[\w/.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|cpp|c|h)\b",
        "function": r"\b([a-z_][a-z0-9_]*)\s*\(",
        "class": r"\b([A-Z][a-zA-Z0-9]+)\b",
        "error": (
            r"\b(Error|Exception|TypeError|ValueError|KeyError|AttributeError|"
            r"RuntimeError|ImportError|IndexError|NameError|OSError|IOError)\b"
        ),
    }

    def extract_regex(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using regex patterns.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities with types and confidence scores
        """
        entities: list[ExtractedEntity] = []

        for entity_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                # Get the captured group if exists, otherwise the full match
                name = match.group(1) if match.lastindex else match.group(0)
                entities.append(
                    ExtractedEntity(
                        name=name,
                        entity_type=entity_type,
                        confidence=0.7,  # Regex-based = medium confidence
                    )
                )

        # Deduplicate by (name, type)
        seen: set[tuple[str, str]] = set()
        unique: list[ExtractedEntity] = []
        for entity in entities:
            key = (entity.name, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    async def extract_with_llm(
        self,
        text: str,
        llm_client: Any | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities using LLM (optional enhancement).

        Falls back to regex if no LLM client provided.

        Args:
            text: Text to extract entities from
            llm_client: Optional LLM client for enhanced extraction

        Returns:
            List of extracted entities
        """
        if not llm_client:
            return self.extract_regex(text)

        # LLM-based extraction (future enhancement)
        # Prompt: "Extract code-related entities from this text..."
        # For now, use regex
        return self.extract_regex(text)


class EntityGraphManager:
    """Manage entity relationships in the graph database.

    Stores extracted entities as graph nodes and creates relationships:
    - DISCUSSED: Links entities that co-occur in the same message
    - RELATED_TO: Links Entity nodes to File nodes when entity is a file path

    Attributes:
        graph: GraphStore instance for Cypher queries
        session_id: Session identifier for scoping entities
        extractor: EntityExtractor instance
    """

    def __init__(self, graph: GraphStore, session_id: str) -> None:
        """Initialize entity graph manager.

        Args:
            graph: GraphStore instance
            session_id: Session identifier
        """
        self.graph = graph
        self.session_id = session_id
        self.extractor = EntityExtractor()

    def process_message(self, content: str) -> list[ExtractedEntity]:
        """Extract entities from message and add to graph.

        Args:
            content: Message content to process

        Returns:
            List of extracted entities
        """
        entities = self.extractor.extract_regex(content)

        for entity in entities:
            self._add_entity(entity)

        # Link related entities (co-occurrence)
        self._link_entities(entities)

        return entities

    def _add_entity(self, entity: ExtractedEntity) -> None:
        """Add entity node to graph.

        Creates an Entity node with session scope. For file entities,
        also creates a RELATED_TO relationship to matching File nodes.

        Args:
            entity: Entity to add
        """
        entity_id = f"{self.session_id}:{entity.entity_type}:{entity.name}"

        # Create or update entity node
        self.graph.query(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.entity_type = $type,
                e.session_id = $session,
                e.confidence = $confidence
            """,
            {
                "id": entity_id,
                "name": entity.name,
                "type": entity.entity_type,
                "session": self.session_id,
                "confidence": entity.confidence,
            },
        )

        # Link to File nodes if entity type is file
        if entity.entity_type == "file":
            self.graph.query(
                """
                MATCH (e:Entity {id: $id}), (f:File)
                WHERE f.path CONTAINS $name
                MERGE (e)-[:RELATED_TO {weight: 1.0}]->(f)
                """,
                {"id": entity_id, "name": entity.name},
            )

    def _link_entities(self, entities: list[ExtractedEntity]) -> None:
        """Link co-occurring entities with DISCUSSED relationship.

        Creates bidirectional DISCUSSED edges between all entity pairs
        that appear in the same message.

        Args:
            entities: List of entities to link
        """
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                id1 = f"{self.session_id}:{e1.entity_type}:{e1.name}"
                id2 = f"{self.session_id}:{e2.entity_type}:{e2.name}"

                self.graph.query(
                    """
                    MATCH (a:Entity {id: $id1}), (b:Entity {id: $id2})
                    MERGE (a)-[:DISCUSSED]->(b)
                    """,
                    {"id1": id1, "id2": id2},
                )
