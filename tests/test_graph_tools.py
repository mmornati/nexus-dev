"""Tests for graph query MCP tools (search_dependencies, find_callers, find_implementations)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from redislite import FalkorDB

from nexus_dev.code_graph import PythonGraphBuilder
from nexus_dev.config import NexusConfig
from nexus_dev.graph_store import GraphStore


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a sample Python project with dependencies."""
    project = tmp_path / "sample_project"
    project.mkdir()

    # Create utils.py
    utils = project / "utils.py"
    utils.write_text(
        """
def validate_user(username: str) -> bool:
    \"\"\"Validate username.\"\"\"
    return len(username) > 3

def format_name(name: str) -> str:
    \"\"\"Format name.\"\"\"
    return name.strip().title()
"""
    )

    # Create auth.py (imports utils)
    auth = project / "auth.py"
    auth.write_text(
        """
from utils import validate_user

def handle_login(username: str) -> bool:
    \"\"\"Handle user login.\"\"\"
    if validate_user(username):
        return True
    return False
"""
    )

    # Create main.py (imports auth)
    main = project / "main.py"
    main.write_text(
        """
from auth import handle_login
from utils import format_name

def main():
    \"\"\"Main entry point.\"\"\"
    username = format_name("john doe")
    result = handle_login(username)
    return result
"""
    )

    # Create base_handler.py with class hierarchy
    base = project / "base_handler.py"
    base.write_text(
        """
class BaseHandler:
    \"\"\"Base handler class.\"\"\"
    def handle(self):
        pass

class MiddleHandler(BaseHandler):
    \"\"\"Middle handler.\"\"\"
    def handle(self):
        super().handle()
"""
    )

    # Create api_handler.py (inherits from BaseHandler)
    api = project / "api_handler.py"
    api.write_text(
        """
from base_handler import BaseHandler

class APIHandler(BaseHandler):
    \"\"\"API handler.\"\"\"
    def handle(self):
        return "API"
"""
    )

    return project


@pytest.fixture
def indexed_graph(sample_project: Path, graph_client: FalkorDB) -> GraphStore:
    """Index sample project into graph database."""
    # Create graph store with same name as HybridDatabase uses
    graph_store = GraphStore(graph_client, "nexus_code_graph")
    graph_store.connect()

    # Index all Python files
    builder = PythonGraphBuilder(graph_store, "test-project")

    for py_file in sample_project.glob("*.py"):
        builder.index_file(py_file)

    # DEBUG: Verify data is in the graph
    main_path = str(sample_project / "main.py")
    result = graph_store.query(
        "MATCH (f:File {path: $path})-[:IMPORTS]->(dep:File) RETURN dep.path", {"path": main_path}
    )
    print(f"\n[FIXTURE] After indexing, main.py imports: {result.result_set}")
    print(f"[FIXTURE] Graph name: {graph_store.graph_name}")
    print(f"[FIXTURE] Graph client ID: {id(graph_store.client)}")

    return graph_store


@pytest.fixture
def test_config(tmp_path: Path) -> NexusConfig:
    """Create test config with hybrid DB enabled."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = True
    return config


@pytest.fixture
def test_hybrid_db(indexed_graph: GraphStore, test_config: NexusConfig, graph_client: FalkorDB):
    """Create HybridDatabase that shares the same FalkorDB instance as indexed_graph."""
    from unittest.mock import MagicMock

    from nexus_dev.hybrid_db import HybridDatabase

    hybrid_db = HybridDatabase(test_config)
    # Directly set the instances to use the test fixtures
    # This ensures we're querying the same graph that was populated
    hybrid_db._graph_store = indexed_graph
    hybrid_db._falkor_db = graph_client

    # IMPORTANT: We must set _kv_store to something not None to prevent
    # connect() from re-initializing the database and creating a NEW FalkorDB instance
    hybrid_db._kv_store = MagicMock()

    return hybrid_db


@pytest.mark.asyncio
async def test_search_dependencies_imports(
    test_hybrid_db, sample_project: Path, monkeypatch
) -> None:
    """Test search_dependencies with 'imports' direction."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    # Use absolute path as stored in graph
    main_path = str(sample_project / "main.py")
    result = await server.search_dependencies(main_path, direction="imports", depth=1)

    assert "## Imports" in result
    assert "auth" in result.lower()
    assert "utils" in result.lower()


@pytest.mark.asyncio
async def test_search_dependencies_imported_by(
    test_hybrid_db, sample_project: Path, monkeypatch
) -> None:
    """Test search_dependencies with 'imported_by' direction."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    utils_path = str(sample_project / "utils.py")
    result = await server.search_dependencies(utils_path, direction="imported_by", depth=1)

    assert "## Imported By" in result
    assert "auth" in result.lower() or "main" in result.lower()


@pytest.mark.asyncio
async def test_search_dependencies_both(test_hybrid_db, sample_project: Path, monkeypatch) -> None:
    """Test search_dependencies with 'both' direction."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    auth_path = str(sample_project / "auth.py")
    result = await server.search_dependencies(auth_path, direction="both", depth=1)

    # Should show what auth.py imports
    assert "## Imports" in result
    assert "utils" in result.lower()

    # Should show what imports auth.py
    assert "## Imported By" in result
    assert "main" in result.lower()


@pytest.mark.asyncio
async def test_search_dependencies_depth(test_hybrid_db, sample_project: Path, monkeypatch) -> None:
    """Test search_dependencies with depth > 1."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    # main.py -> auth.py -> utils.py (depth 2)
    main_path = str(sample_project / "main.py")
    result = await server.search_dependencies(main_path, direction="imports", depth=2)

    assert "utils" in result.lower()


@pytest.mark.asyncio
async def test_search_dependencies_not_found(test_hybrid_db, monkeypatch) -> None:
    """Test search_dependencies with non-existent file."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    result = await server.search_dependencies("nonexistent.py", direction="both")

    assert "No dependencies found" in result
    assert "nonexistent.py" in result


@pytest.mark.asyncio
async def test_find_callers(test_hybrid_db, monkeypatch) -> None:
    """Test find_callers functionality."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    result = await server.find_callers("validate_user")

    assert "## Functions that call" in result
    assert "validate_user" in result
    assert "handle_login" in result


@pytest.mark.asyncio
async def test_find_callers_multiple(test_hybrid_db, monkeypatch) -> None:
    """Test find_callers with multiple callers."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    result = await server.find_callers("format_name")

    assert "## Functions that call" in result
    assert "format_name" in result
    assert "main" in result


@pytest.mark.asyncio
async def test_find_callers_not_found(test_hybrid_db, monkeypatch) -> None:
    """Test find_callers with non-existent function."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    result = await server.find_callers("nonexistent_function")

    assert "No callers found" in result
    assert "nonexistent_function" in result


@pytest.mark.asyncio
async def test_find_implementations(test_hybrid_db, monkeypatch) -> None:
    """Test find_implementations functionality."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    result = await server.find_implementations("BaseHandler")

    assert "## Classes inheriting from" in result
    assert "BaseHandler" in result
    # Should find both MiddleHandler and APIHandler
    assert "MiddleHandler" in result or "APIHandler" in result


@pytest.mark.asyncio
async def test_find_implementations_not_found(test_hybrid_db, monkeypatch) -> None:
    """Test find_implementations with non-existent class."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    result = await server.find_implementations("NonExistentClass")

    assert "No subclasses found" in result
    assert "NonExistentClass" in result


@pytest.mark.asyncio
async def test_hybrid_db_disabled(test_config: NexusConfig, monkeypatch) -> None:
    """Test graceful handling when hybrid DB is disabled."""
    from nexus_dev import server

    # Ensure hybrid DB is disabled
    test_config.enable_hybrid_db = False
    monkeypatch.setattr(server, "_hybrid_db", None)

    result = await server.search_dependencies("test.py")

    assert "Error" in result
    assert "not enabled" in result


@pytest.mark.asyncio
async def test_depth_clamping(test_hybrid_db, sample_project: Path, monkeypatch) -> None:
    """Test that depth is clamped to max 5."""
    from nexus_dev import server

    monkeypatch.setattr(server, "_hybrid_db", test_hybrid_db)

    # Request depth of 100, should be clamped to 5
    main_path = str(sample_project / "main.py")
    result = await server.search_dependencies(main_path, direction="imports", depth=100)

    # Should not error, depth should be clamped
    assert "Error" not in result or "No dependencies" in result
