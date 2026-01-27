"""Tests for hybrid database module using FalkorDBLite."""

from pathlib import Path

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.hybrid_db import HybridDatabase


def test_hybrid_database_disabled_by_default(tmp_path: Path) -> None:
    """Test that hybrid database is disabled by default."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")

    db = HybridDatabase(config)

    # Should not connect when disabled
    db.connect()
    assert db._kv_store is None
    assert db._graph_store is None
    # No server/client should be init
    assert db._falkor_db is None


def test_hybrid_database_requires_flag(tmp_path: Path) -> None:
    """Test that accessing databases without enable flag raises error."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = False

    db = HybridDatabase(config)

    with pytest.raises(RuntimeError, match="Hybrid database is not enabled"):
        _ = db.kv

    with pytest.raises(RuntimeError, match="Hybrid database is not enabled"):
        _ = db.graph


def test_hybrid_database_initialization(tmp_path: Path) -> None:
    """Test hybrid database initializes all components when enabled."""
    config = NexusConfig.create_new("test-project")
    # Use unique path for test
    config.db_path = str(tmp_path / "hybrid_init_db")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)
    db.connect()

    try:
        # Check FalkorDB components
        assert db._falkor_db is not None

        # KV store should be initialized
        assert db._kv_store is not None
        from nexus_dev.kv_store import KVStore

        assert isinstance(db._kv_store, KVStore)
        # Check client sharing - KVStore uses .client attribute of FalkorDB object
        assert db._kv_store.client == db._falkor_db.client

        # Graph store should be initialized
        assert db._graph_store is not None
        from nexus_dev.graph_store import GraphStore

        assert isinstance(db._graph_store, GraphStore)
        # Check client sharing - GraphStore uses FalkorDB object itself
        assert db._graph_store.client == db._falkor_db

        # Verify functionality via high-level properties
        db.kv.create_session("test-session", "test-project")
        session = db.kv.get_session("test-session")
        assert session is not None

    finally:
        db.close()


def test_kv_property_lazy_initialization(tmp_path: Path) -> None:
    """Test KV property initializes connection on first access."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "hybrid_lazy_kv")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)

    # Should initialize on property access
    kv = db.kv
    assert kv is not None
    assert db._falkor_db is not None

    db.close()


def test_graph_property_lazy_initialization(tmp_path: Path) -> None:
    """Test graph property initializes connection on first access."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "hybrid_lazy_graph")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)

    # Should initialize on property access
    graph = db.graph
    assert graph is not None
    assert db._falkor_db is not None

    db.close()


def test_context_manager(tmp_path: Path) -> None:
    """Test hybrid database works as context manager."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "hybrid_cm")
    config.enable_hybrid_db = True

    with HybridDatabase(config) as db:
        # Should be connected
        assert db._kv_store is not None
        kv = db.kv
        assert kv is not None

    # Should be closed after context
    # Client should be closed/None
    assert db._kv_store is None
    assert db._falkor_db is None


def test_database_directories_created(tmp_path: Path) -> None:
    """Test that database directories are created on initialization."""
    config = NexusConfig.create_new("test-project")
    db_path = tmp_path / "hybrid_dir_test"
    config.db_path = str(db_path)
    config.enable_hybrid_db = True

    db = HybridDatabase(config)
    db.connect()

    try:
        assert db_path.exists()
        assert db_path.is_dir()
    finally:
        db.close()


def test_close_idempotent(tmp_path: Path) -> None:
    """Test that close() can be called multiple times safely."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "hybrid_close")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)
    db.connect()

    # Should not raise on multiple closes
    db.close()
    db.close()
    db.close()
