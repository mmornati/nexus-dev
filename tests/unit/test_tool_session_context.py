"""Tests for session context tools."""

from unittest.mock import MagicMock, patch

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.hybrid_db import HybridDatabase
from nexus_dev.kv_store import KVStore
from nexus_dev.tools.context import (
    get_search_suggestions,
    get_session_context,
    set_session_context,
)


@pytest.fixture
def mock_hybrid_db(redis_client):
    """Create a HybridDatabase using the shared Redis client."""
    config = NexusConfig.create_new("test-project")
    config.enable_hybrid_db = True

    db = HybridDatabase(config)
    db._falkor_db = MagicMock()
    db._kv_store = KVStore(redis_client)
    db._graph_store = MagicMock()

    db.kv.create_session("test-session", "test-project")

    return db


@pytest.mark.asyncio
async def test_set_session_context_with_task(mock_hybrid_db):
    """Test setting session context with current task."""
    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await set_session_context(
            session_id="test-session",
            current_task="Implementing user authentication",
        )

        assert "Session Context Updated" in result
        assert "Implementing user authentication" in result


@pytest.mark.asyncio
async def test_set_session_context_with_files(mock_hybrid_db):
    """Test setting session context with recent files."""
    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await set_session_context(
            session_id="test-session",
            recent_files=["src/auth.py", "src/user.py"],
        )

        assert "Session Context Updated" in result
        assert "src/auth.py" in result
        assert "src/user.py" in result


@pytest.mark.asyncio
async def test_set_session_context_with_metadata(mock_hybrid_db):
    """Test setting session context with metadata."""
    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await set_session_context(
            session_id="test-session",
            metadata={"language": "python", "framework": "fastapi"},
        )

        assert "Session Context Updated" in result
        assert "python" in result


@pytest.mark.asyncio
async def test_set_session_context_error_no_params(mock_hybrid_db):
    """Test error when no parameters provided."""
    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await set_session_context(session_id="test-session")

        assert "Error" in result


@pytest.mark.asyncio
async def test_get_session_context_success(mock_hybrid_db):
    """Test getting session context."""
    mock_hybrid_db.kv.set_session_context(
        session_id="test-session",
        current_task="Implementing user auth",
        recent_files=["src/auth.py"],
    )

    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await get_session_context(session_id="test-session")

        assert "Session Context" in result
        assert "Implementing user auth" in result
        assert "src/auth.py" in result


@pytest.mark.asyncio
async def test_get_session_context_empty(mock_hybrid_db):
    """Test getting empty session context."""
    mock_hybrid_db.kv.create_session("empty-context-session", "test-project")

    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await get_session_context(session_id="empty-context-session")

        assert "No session context found" in result


@pytest.mark.asyncio
async def test_get_session_context_disabled(mock_hybrid_db):
    """Test behavior when hybrid DB is disabled."""
    mock_hybrid_db.config.enable_hybrid_db = False

    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await get_session_context(session_id="test-session")

        assert "Hybrid database is not enabled" in result


@pytest.mark.asyncio
async def test_get_search_suggestions_with_context(mock_hybrid_db):
    """Test getting search suggestions with context."""
    mock_hybrid_db.kv.set_session_context(
        session_id="test-session",
        current_task="User authentication",
        recent_files=["src/auth.py", "tests/test_auth.py"],
    )

    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await get_search_suggestions(session_id="test-session")

        assert "Search Suggestions" in result
        assert "User authentication" in result


@pytest.mark.asyncio
async def test_get_search_suggestions_limit(mock_hybrid_db):
    """Test suggestions respect limit parameter."""
    mock_hybrid_db.kv.set_session_context(
        session_id="test-session",
        current_task="Implementing feature",
        recent_files=["src/a.py", "src/b.py", "src/c.py", "src/d.py"],
    )

    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await get_search_suggestions(session_id="test-session", limit=2)

        lines = result.split("\n")
        suggestion_lines = [line for line in lines if line.strip().startswith(("1.", "2.", "3."))]
        assert len(suggestion_lines) <= 2


@pytest.mark.asyncio
async def test_get_search_suggestions_no_context(mock_hybrid_db):
    """Test suggestions when no context available."""
    mock_hybrid_db.kv.create_session("no-context-session", "test-project")

    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await get_search_suggestions(session_id="no-context-session")

        assert "No session context available" in result


@pytest.mark.asyncio
async def test_get_search_suggestions_disabled(mock_hybrid_db):
    """Test behavior when hybrid DB is disabled."""
    mock_hybrid_db.config.enable_hybrid_db = False

    with patch("nexus_dev.tools.context.get_hybrid_db", return_value=mock_hybrid_db):
        result = await get_search_suggestions(session_id="test-session")

        assert "Hybrid database is not enabled" in result


@pytest.mark.asyncio
async def test_add_recent_file(mock_hybrid_db):
    """Test adding recent files maintains order and dedupes."""
    mock_hybrid_db.kv.set_session_context(
        session_id="test-session",
        recent_files=["src/a.py"],
    )

    mock_hybrid_db.kv.add_recent_file("test-session", "src/b.py")
    mock_hybrid_db.kv.add_recent_file("test-session", "src/c.py")
    mock_hybrid_db.kv.add_recent_file("test-session", "src/a.py")

    context = mock_hybrid_db.kv.get_session_context("test-session")
    recent = context["recent_files"]

    assert recent[0] == "src/a.py"
    assert "src/b.py" in recent
    assert "src/c.py" in recent
    assert recent.count("src/a.py") == 1
