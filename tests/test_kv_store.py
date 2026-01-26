"""Tests for KV store module."""

from pathlib import Path
from time import sleep

import pytest

from nexus_dev.kv_store import KVStore


def test_kv_store_initialization(tmp_path: Path) -> None:
    """Test KV store creates database and schema."""
    db_path = tmp_path / "test.db"

    kv = KVStore(db_path)
    kv.connect()

    try:
        assert db_path.exists()
        # Verify tables were created by trying to use them
        kv.create_session("sess-1", "proj-1")
        session = kv.get_session("sess-1")
        assert session is not None
    finally:
        kv.close()


def test_context_manager(tmp_path: Path) -> None:
    """Test KV store works as context manager."""
    db_path = tmp_path / "test.db"

    with KVStore(db_path) as kv:
        kv.create_session("sess-1", "proj-1")

    # Should be able to reopen
    with KVStore(db_path) as kv:
        session = kv.get_session("sess-1")
        assert session is not None


# Session tests


def test_create_and_get_session(tmp_path: Path) -> None:
    """Test creating and retrieving sessions."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1")

        session = kv.get_session("sess-1")
        assert session is not None
        assert session["session_id"] == "sess-1"
        assert session["project_id"] == "proj-1"
        assert "created_at" in session
        assert "updated_at" in session
        assert session["metadata"] == {}


def test_create_session_with_metadata(tmp_path: Path) -> None:
    """Test creating session with metadata."""
    metadata = {"user": "alice", "theme": "dark"}

    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1", metadata)

        session = kv.get_session("sess-1")
        assert session is not None
        assert session["metadata"] == metadata


def test_update_session_metadata(tmp_path: Path) -> None:
    """Test updating session metadata."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1", {"version": "1.0"})

        # Update metadata
        kv.update_session("sess-1", {"version": "2.0", "updated": True})

        session = kv.get_session("sess-1")
        assert session is not None
        assert session["metadata"] == {"version": "2.0", "updated": True}


def test_get_nonexistent_session(tmp_path: Path) -> None:
    """Test getting session that doesn't exist."""
    with KVStore(tmp_path / "test.db") as kv:
        session = kv.get_session("nonexistent")
        assert session is None


def test_delete_session(tmp_path: Path) -> None:
    """Test deleting session."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1")
        kv.add_message("sess-1", "user", "Hello")

        # Delete session
        kv.delete_session("sess-1")

        # Should not exist
        assert kv.get_session("sess-1") is None

        # Chat history should also be deleted
        messages = kv.get_recent_messages("sess-1")
        assert len(messages) == 0


# Chat history tests


def test_add_and_get_messages(tmp_path: Path) -> None:
    """Test adding and retrieving messages."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1")

        msg_id1 = kv.add_message("sess-1", "user", "Hello")
        msg_id2 = kv.add_message("sess-1", "assistant", "Hi there!")

        assert msg_id1 > 0
        assert msg_id2 > msg_id1

        messages = kv.get_recent_messages("sess-1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"


def test_get_recent_messages_limit(tmp_path: Path) -> None:
    """Test limiting number of messages returned."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1")

        # Add 5 messages
        for i in range(5):
            kv.add_message("sess-1", "user", f"Message {i}")

        # Get only 3
        messages = kv.get_recent_messages("sess-1", limit=3)
        assert len(messages) == 3
        # Should be most recent in chronological order
        assert messages[0]["content"] == "Message 2"
        assert messages[1]["content"] == "Message 3"
        assert messages[2]["content"] == "Message 4"


def test_get_messages_chronological_order(tmp_path: Path) -> None:
    """Test messages are returned in chronological order."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1")

        kv.add_message("sess-1", "user", "First")
        kv.add_message("sess-1", "assistant", "Second")
        kv.add_message("sess-1", "user", "Third")

        messages = kv.get_recent_messages("sess-1")
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"


def test_get_message_count(tmp_path: Path) -> None:
    """Test getting message count for session."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1")

        assert kv.get_message_count("sess-1") == 0

        kv.add_message("sess-1", "user", "Message 1")
        kv.add_message("sess-1", "user", "Message 2")

        assert kv.get_message_count("sess-1") == 2


def test_messages_across_sessions(tmp_path: Path) -> None:
    """Test messages are isolated by session."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.create_session("sess-1", "proj-1")
        kv.create_session("sess-2", "proj-1")

        kv.add_message("sess-1", "user", "Message for session 1")
        kv.add_message("sess-2", "user", "Message for session 2")

        messages1 = kv.get_recent_messages("sess-1")
        messages2 = kv.get_recent_messages("sess-2")

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0]["content"] == "Message for session 1"
        assert messages2[0]["content"] == "Message for session 2"


# Config cache tests


def test_set_and_get_cache(tmp_path: Path) -> None:
    """Test setting and getting cache entries."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.set_cache("key1", {"foo": "bar"})
        kv.set_cache("key2", [1, 2, 3])

        assert kv.get_cache("key1") == {"foo": "bar"}
        assert kv.get_cache("key2") == [1, 2, 3]


def test_cache_overwrites(tmp_path: Path) -> None:
    """Test cache key can be overwritten."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.set_cache("key1", "value1")
        kv.set_cache("key1", "value2")

        assert kv.get_cache("key1") == "value2"


def test_get_nonexistent_cache(tmp_path: Path) -> None:
    """Test getting cache key that doesn't exist."""
    with KVStore(tmp_path / "test.db") as kv:
        assert kv.get_cache("nonexistent") is None


def test_cache_ttl_expiration(tmp_path: Path) -> None:
    """Test cache entries expire after TTL."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.set_cache("key1", "value1", ttl_seconds=1)

        # Should exist immediately
        assert kv.get_cache("key1") == "value1"

        # Wait for expiration
        sleep(1.1)

        # Should be expired
        assert kv.get_cache("key1") is None


def test_cache_no_ttl(tmp_path: Path) -> None:
    """Test cache without TTL never expires."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.set_cache("key1", "value1")

        # Wait a bit
        sleep(0.5)

        # Should still exist
        assert kv.get_cache("key1") == "value1"


def test_delete_cache(tmp_path: Path) -> None:
    """Test deleting cache entry."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.set_cache("key1", "value1")
        kv.delete_cache("key1")

        assert kv.get_cache("key1") is None


def test_cleanup_expired_cache(tmp_path: Path) -> None:
    """Test cleanup removes only expired entries."""
    with KVStore(tmp_path / "test.db") as kv:
        kv.set_cache("expired1", "value1", ttl_seconds=1)
        kv.set_cache("expired2", "value2", ttl_seconds=1)
        kv.set_cache("permanent", "value3")  # No TTL

        sleep(1.1)

        count = kv.cleanup_expired()
        assert count == 2

        # Permanent should still exist
        assert kv.get_cache("permanent") == "value3"
        assert kv.get_cache("expired1") is None
        assert kv.get_cache("expired2") is None


def test_cache_json_serialization(tmp_path: Path) -> None:
    """Test cache handles complex JSON types."""
    with KVStore(tmp_path / "test.db") as kv:
        complex_data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        kv.set_cache("complex", complex_data)
        retrieved = kv.get_cache("complex")

        assert retrieved == complex_data


def test_close_idempotent(tmp_path: Path) -> None:
    """Test close() can be called multiple times."""
    kv = KVStore(tmp_path / "test.db")
    kv.connect()

    kv.close()
    kv.close()
    kv.close()


def test_operations_raise_when_not_connected(tmp_path: Path) -> None:
    """Test operations raise error when not connected."""
    kv = KVStore(tmp_path / "test.db")

    with pytest.raises(RuntimeError, match="Not connected"):
        kv.create_session("sess-1", "proj-1")

    with pytest.raises(RuntimeError, match="Not connected"):
        kv.get_session("sess-1")
