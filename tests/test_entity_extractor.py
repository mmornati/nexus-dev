"""Tests for entity_extractor module."""

from pathlib import Path

import pytest

from nexus_dev.entity_extractor import EntityExtractor, EntityGraphManager, ExtractedEntity
from nexus_dev.graph_store import GraphStore


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_create_entity(self) -> None:
        """Test creating an ExtractedEntity."""
        entity = ExtractedEntity(name="auth.py", entity_type="file", confidence=0.7)
        assert entity.name == "auth.py"
        assert entity.entity_type == "file"
        assert entity.confidence == 0.7


class TestEntityExtractor:
    """Tests for EntityExtractor regex patterns."""

    def test_extract_files(self) -> None:
        """Test extraction of file paths."""
        extractor = EntityExtractor()
        text = "The bug is in auth.py and also affects utils.js"
        entities = extractor.extract_regex(text)

        file_entities = [e for e in entities if e.entity_type == "file"]
        names = {e.name for e in file_entities}
        assert "auth.py" in names
        assert "utils.js" in names

    def test_extract_functions(self) -> None:
        """Test extraction of function calls."""
        extractor = EntityExtractor()
        text = "The function validate_user() is called by authenticate()"
        entities = extractor.extract_regex(text)

        func_entities = [e for e in entities if e.entity_type == "function"]
        names = {e.name for e in func_entities}
        assert "validate_user" in names
        assert "authenticate" in names

    def test_extract_classes(self) -> None:
        """Test extraction of PascalCase class names."""
        extractor = EntityExtractor()
        text = "The UserController extends BaseController and uses AuthService"
        entities = extractor.extract_regex(text)

        class_entities = [e for e in entities if e.entity_type == "class"]
        names = {e.name for e in class_entities}
        assert "UserController" in names
        assert "BaseController" in names
        assert "AuthService" in names

    def test_extract_errors(self) -> None:
        """Test extraction of error/exception types."""
        extractor = EntityExtractor()
        text = "Caught TypeError when None was passed, also handle ValueError"
        entities = extractor.extract_regex(text)

        error_entities = [e for e in entities if e.entity_type == "error"]
        names = {e.name for e in error_entities}
        assert "TypeError" in names
        assert "ValueError" in names

    def test_deduplication(self) -> None:
        """Test that duplicate entities are removed."""
        extractor = EntityExtractor()
        text = "Call validate() then validate() again and validate() once more"
        entities = extractor.extract_regex(text)

        func_entities = [e for e in entities if e.name == "validate"]
        # Should only have one entry despite three mentions
        assert len(func_entities) == 1

    def test_combined_extraction(self) -> None:
        """Test extraction of mixed entity types from Issue 62 example."""
        extractor = EntityExtractor()
        text = "The UserController in auth.py calls validate_user() and raises ValueError"
        entities = extractor.extract_regex(text)

        names = {e.name for e in entities}
        assert "auth.py" in names
        assert "UserController" in names
        assert "validate_user" in names
        assert "ValueError" in names

    def test_confidence_score(self) -> None:
        """Test that regex extractions have medium confidence."""
        extractor = EntityExtractor()
        text = "Check auth.py"
        entities = extractor.extract_regex(text)

        for entity in entities:
            assert entity.confidence == 0.7


class TestEntityGraphManager:
    """Tests for EntityGraphManager with FalkorDB graph."""

    def test_process_message_stores_entities(self, graph_client) -> None:
        """Test that entities are stored as graph nodes."""
        gs = GraphStore(graph_client, "test_entity_graph")
        gs.connect()
        manager = EntityGraphManager(gs, "session-123")

        text = "Working on auth.py with UserController"
        entities = manager.process_message(text)

        assert len(entities) >= 2  # auth.py and UserController

        # Verify entities in graph
        result = gs.query("MATCH (e:Entity) RETURN e.name, e.entity_type ORDER BY e.name")
        names = [row[0] for row in result.result_set]
        assert "UserController" in names
        assert "auth.py" in names

    def test_entities_scoped_to_session(self, graph_client) -> None:
        """Test that entities have session_id property."""
        gs = GraphStore(graph_client, "test_session_scope_graph")
        gs.connect()
        manager = EntityGraphManager(gs, "my-session")

        manager.process_message("Check utils.py")

        result = gs.query("MATCH (e:Entity) RETURN e.session_id")
        sessions = [row[0] for row in result.result_set]
        assert all(s == "my-session" for s in sessions)

    def test_co_occurring_entities_linked(self, graph_client) -> None:
        """Test that DISCUSSED edges are created between co-occurring entities."""
        gs = GraphStore(graph_client, "test_discussed_graph")
        gs.connect()
        manager = EntityGraphManager(gs, "sess-1")

        # Two entities in same message should be linked
        manager.process_message("Error in auth.py causes ValueError")

        result = gs.query(
            """
            MATCH (a:Entity)-[:DISCUSSED]->(b:Entity)
            RETURN a.name, b.name
            """
        )
        assert len(result.result_set) >= 1  # At least one DISCUSSED edge

    def test_file_entity_links_to_file_node(self, graph_client, tmp_path: Path) -> None:
        """Test that file entities link to existing File nodes via RELATED_TO."""
        gs = GraphStore(graph_client, "test_file_link_graph")
        gs.connect()

        # First, create a File node (simulating indexed file)
        gs.query(
            """
            MERGE (f:File {path: $path})
            SET f.language = 'python', f.project_id = 'test'
            """,
            {"path": "/path/to/service.py"},
        )

        # Now process a message mentioning that file
        manager = EntityGraphManager(gs, "sess-2")
        manager.process_message("The bug is in service.py")

        # Check for RELATED_TO edge
        result = gs.query(
            """
            MATCH (e:Entity)-[:RELATED_TO]->(f:File)
            RETURN e.name, f.path
            """
        )
        # The entity should link to the File node
        assert len(result.result_set) >= 1

    def test_multiple_messages_accumulate(self, graph_client) -> None:
        """Test that entities from multiple messages accumulate."""
        gs = GraphStore(graph_client, "test_accumulate_graph")
        gs.connect()
        manager = EntityGraphManager(gs, "sess-3")

        manager.process_message("Working on auth.py")
        manager.process_message("Now checking utils.py")

        result = gs.query("MATCH (e:Entity) WHERE e.entity_type = 'file' RETURN e.name")
        file_names = {row[0] for row in result.result_set}
        assert "auth.py" in file_names
        assert "utils.py" in file_names


class TestEntityExtractorAsync:
    """Tests for async entity extraction."""

    @pytest.mark.asyncio
    async def test_extract_with_llm_fallback(self) -> None:
        """Test that extract_with_llm falls back to regex without LLM client."""
        extractor = EntityExtractor()
        text = "Check auth.py"

        # Without LLM client, should fall back to regex
        entities = await extractor.extract_with_llm(text, llm_client=None)
        names = {e.name for e in entities}
        assert "auth.py" in names
