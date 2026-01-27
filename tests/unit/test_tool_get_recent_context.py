"""Tests for get_recent_context tool."""

from unittest.mock import MagicMock, patch

import pytest

from nexus_dev import server
from nexus_dev.config import NexusConfig
from nexus_dev.hybrid_db import HybridDatabase


@pytest.fixture
def mock_hybrid_db(redis_client):
    """Create a HybridDatabase using the shared Redis client."""
    config = NexusConfig.create_new("test-project")
    config.enable_hybrid_db = True

    # Create a hybrid database but use the existing shared redis client
    db = HybridDatabase(config)
    # Manually initialize with the shared client instead of creating new FalkorDB
    from nexus_dev.kv_store import KVStore

    # Create falkor db wrapper with the existing client
    db._falkor_db = MagicMock()
    db._kv_store = KVStore(redis_client)
    db._graph_store = MagicMock()  # Not used in this test

    # Add some sample data
    db.kv.create_session("test-session", "test-project")
    db.kv.add_message("test-session", "user", "Hello ai")
    db.kv.add_message("test-session", "assistant", "Hello human")

    return db


@pytest.mark.asyncio
async def test_get_recent_context_success(mock_hybrid_db):
    """Test successful retrieval of context."""
    # Patch the server's hybrid db getter
    with patch("nexus_dev.server._get_hybrid_db", return_value=mock_hybrid_db):
        result = await server.get_recent_context(session_id="test-session")

        assert "## Recent Context (Session: test-session)" in result
        assert "### USER" in result
        assert "Hello ai" in result
        assert "### ASSISTANT" in result
        assert "Hello human" in result


@pytest.mark.asyncio
async def test_get_recent_context_disabled(mock_hybrid_db):
    """Test behavior when hybrid DB is disabled."""
    mock_hybrid_db.config.enable_hybrid_db = False

    with patch("nexus_dev.server._get_hybrid_db", return_value=mock_hybrid_db):
        result = await server.get_recent_context(session_id="test-session")

        assert "Hybrid database is not enabled" in result


@pytest.mark.asyncio
async def test_get_recent_context_no_history(mock_hybrid_db):
    """Test retrieval for empty session."""
    mock_hybrid_db.kv.create_session("empty-session", "test-project")

    with patch("nexus_dev.server._get_hybrid_db", return_value=mock_hybrid_db):
        result = await server.get_recent_context(session_id="empty-session")

        assert "No chat history found" in result


@pytest.mark.asyncio
async def test_get_recent_context_limit(mock_hybrid_db):
    """Test limit parameter."""
    # Add more messages
    for i in range(5):
        mock_hybrid_db.kv.add_message("test-session", "user", f"msg {i}")

    with patch("nexus_dev.server._get_hybrid_db", return_value=mock_hybrid_db):
        # Limit to 2 most recent + 2 existing = 4 messages?
        # Wait, get_recent_messages returns *recent* ones.
        # We added 2 initial, then 5 more. Total 7.
        # Limit 3 should return last 3.

        result = await server.get_recent_context(session_id="test-session", limit=3)

        # Should contain msg 4, msg 3, msg 2
        assert "msg 4" in result
        assert "msg 3" in result
        assert "msg 2" in result
        # Should NOT contain "Hello ai" (oldest)
        assert "Hello ai" not in result
