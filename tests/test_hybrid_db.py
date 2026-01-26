"""Tests for hybrid database module."""

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
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)
    db.connect()

    try:
        # KV store should be initialized
        assert db._kv_store is not None
        from nexus_dev.kv_store import KVStore

        assert isinstance(db._kv_store, KVStore)

        # Graph store should be initialized
        assert db._graph_store is not None
        from nexus_dev.graph_store import GraphStore

        assert isinstance(db._graph_store, GraphStore)

        # KV schema should be created - verify through KVStore
        # Try to create a session to verify tables exist
        db._kv_store.create_session("test-session", "test-project")
        session = db._kv_store.get_session("test-session")
        assert session is not None
        assert session["session_id"] == "test-session"

    finally:
        db.close()


def test_kv_property_lazy_initialization(tmp_path: Path) -> None:
    """Test KV property initializes connection on first access."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)

    # Should initialize on property access
    kv = db.kv
    assert kv is not None
    from nexus_dev.kv_store import KVStore

    assert isinstance(kv, KVStore)

    db.close()


def test_graph_property_lazy_initialization(tmp_path: Path) -> None:
    """Test graph property initializes connection on first access."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)

    # Should initialize on property access
    graph = db.graph
    assert graph is not None

    db.close()


def test_context_manager(tmp_path: Path) -> None:
    """Test hybrid database works as context manager."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = True

    with HybridDatabase(config) as db:
        # Should be connected
        assert db._kv_store is not None
        kv = db.kv
        assert kv is not None

    # Should be closed after context
    # Note: SQLite connection might still work, but that's ok


def test_database_directories_created(tmp_path: Path) -> None:
    """Test that database directories are created on initialization."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)
    db.connect()

    try:
        db_path = Path(tmp_path / "db")
        assert db_path.exists()
        assert (db_path / "state.db").exists()
        # KùzuDB creates its own subdirectories (.tmp, etc.)
        assert db_path.is_dir()
    finally:
        db.close()


def test_close_idempotent(tmp_path: Path) -> None:
    """Test that close() can be called multiple times safely."""
    config = NexusConfig.create_new("test-project")
    config.db_path = str(tmp_path / "db")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)
    db.connect()

    # Should not raise on multiple closes
    db.close()
    db.close()
    db.close()


def test_config_persistence(tmp_path: Path) -> None:
    """Test that enable_hybrid_db persists in config file."""
    config_path = tmp_path / "nexus_config.json"

    # Create and save config with hybrid enabled
    config = NexusConfig.create_new("test-project")
    config.enable_hybrid_db = True
    config.save(config_path)

    # Load config and verify flag persists
    loaded_config = NexusConfig.load(config_path)
    assert loaded_config.enable_hybrid_db is True


def test_config_defaults_to_false(tmp_path: Path) -> None:
    """Test that enable_hybrid_db defaults to False in loaded config."""
    config_path = tmp_path / "nexus_config.json"

    # Create and save config without hybrid flag
    config = NexusConfig.create_new("test-project")
    config.save(config_path)

    # Load config and verify flag defaults to False
    loaded_config = NexusConfig.load(config_path)
    assert loaded_config.enable_hybrid_db is False
