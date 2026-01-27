"""Tests for KV store module using Redislite/FalkorDBLite."""

from time import sleep

from nexus_dev.kv_store import KVStore

# Note: redis_client fixture is provided by conftest.py with module-scoped server


def test_kv_store_initialization(redis_client) -> None:
    """Test KV store initialization."""
    kv = KVStore(redis_client)
    kv.connect()

    # Write something
    kv.create_session("sess-1", "proj-1")
    session = kv.get_session("sess-1")
    assert session is not None
    assert session["session_id"] == "sess-1"


def test_context_manager(redis_client) -> None:
    """Test KV store works as context manager."""
    with KVStore(redis_client) as kv:
        kv.create_session("sess-1", "proj-1")

    # Should be able to access via plain client
    kv2 = KVStore(redis_client)
    session = kv2.get_session("sess-1")
    assert session is not None


# Session tests


def test_create_and_get_session(redis_client) -> None:
    """Test creating and retrieving sessions."""
    kv = KVStore(redis_client)
    kv.create_session("sess-1", "proj-1")

    session = kv.get_session("sess-1")
    assert session is not None
    assert session["session_id"] == "sess-1"
    assert session["project_id"] == "proj-1"
    assert "created_at" in session
    assert "updated_at" in session
    assert session["metadata"] == {}


def test_create_session_with_metadata(redis_client) -> None:
    """Test creating session with metadata."""
    metadata = {"user": "alice", "theme": "dark"}
    kv = KVStore(redis_client)
    kv.create_session("sess-1", "proj-1", metadata)

    session = kv.get_session("sess-1")
    assert session is not None
    assert session["metadata"] == metadata


def test_update_session_metadata(redis_client) -> None:
    """Test updating session metadata."""
    kv = KVStore(redis_client)
    kv.create_session("sess-1", "proj-1", {"version": "1.0"})

    # Update metadata
    kv.update_session("sess-1", {"version": "2.0", "updated": True})

    session = kv.get_session("sess-1")
    assert session is not None
    assert session["metadata"] == {"version": "2.0", "updated": True}


def test_get_nonexistent_session(redis_client) -> None:
    """Test getting session that doesn't exist."""
    kv = KVStore(redis_client)
    session = kv.get_session("nonexistent")
    assert session is None


def test_delete_session(redis_client) -> None:
    """Test deleting session."""
    kv = KVStore(redis_client)
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


def test_add_and_get_messages(redis_client) -> None:
    """Test adding and retrieving messages."""
    kv = KVStore(redis_client)
    kv.create_session("sess-1", "proj-1")

    count1 = kv.add_message("sess-1", "user", "Hello")
    count2 = kv.add_message("sess-1", "assistant", "Hi there!")

    assert count1 == 1
    assert count2 == 2

    messages = kv.get_recent_messages("sess-1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there!"


def test_get_recent_messages_limit(redis_client) -> None:
    """Test limiting number of messages returned."""
    kv = KVStore(redis_client)
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


def test_get_message_count(redis_client) -> None:
    """Test getting message count for session."""
    kv = KVStore(redis_client)

    assert kv.get_message_count("sess-1") == 0

    kv.add_message("sess-1", "user", "Message 1")
    kv.add_message("sess-1", "user", "Message 2")

    assert kv.get_message_count("sess-1") == 2


def test_messages_across_sessions(redis_client) -> None:
    """Test messages are isolated by session."""
    kv = KVStore(redis_client)
    # Sessions don't strictly need to exist to add messages in Redis (List key)
    # but practically we create them.
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


def test_set_and_get_cache(redis_client) -> None:
    """Test setting and getting cache entries."""
    kv = KVStore(redis_client)
    kv.set_cache("key1", {"foo": "bar"})
    kv.set_cache("key2", [1, 2, 3])

    assert kv.get_cache("key1") == {"foo": "bar"}
    assert kv.get_cache("key2") == [1, 2, 3]


def test_cache_overwrites(redis_client) -> None:
    """Test cache key can be overwritten."""
    kv = KVStore(redis_client)
    kv.set_cache("key1", "value1")
    kv.set_cache("key1", "value2")

    assert kv.get_cache("key1") == "value2"


def test_get_nonexistent_cache(redis_client) -> None:
    """Test getting cache key that doesn't exist."""
    kv = KVStore(redis_client)
    assert kv.get_cache("nonexistent") is None


def test_cache_ttl_expiration(redis_client) -> None:
    """Test cache entries expire after TTL."""
    kv = KVStore(redis_client)
    kv.set_cache("key1", "value1", ttl_seconds=1)

    # Should exist immediately
    assert kv.get_cache("key1") == "value1"

    # Wait for expiration
    sleep(1.2)

    # Should be expired
    assert kv.get_cache("key1") is None


def test_delete_cache(redis_client) -> None:
    """Test deleting cache entry."""
    kv = KVStore(redis_client)
    kv.set_cache("key1", "value1")
    kv.delete_cache("key1")

    assert kv.get_cache("key1") is None
