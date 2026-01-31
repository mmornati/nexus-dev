"""Integration tests for smart_search tool."""

from unittest.mock import MagicMock, patch

import pytest

from nexus_dev.query_router import QueryIntent, QueryType


@pytest.mark.asyncio
class TestSmartSearch:
    """Test suite for smart_search tool."""

    @patch("nexus_dev.server.HybridQueryRouter")
    @patch("nexus_dev.server.find_callers")
    async def test_smart_search_graph_callers(self, mock_find_callers, mock_router_cls):
        """Test smart_search routes to find_callers for graph intent."""
        from nexus_dev.server import smart_search

        # Setup router mock
        mock_router = MagicMock()
        mock_intent = QueryIntent(
            query_type=QueryType.GRAPH,
            extracted_entity="my_function",
            original_query="who calls my_function",
        )
        mock_router.route.return_value = mock_intent
        mock_router_cls.return_value = mock_router

        # Setup tool mock
        mock_find_callers.return_value = "Callers found"

        result = await smart_search("who calls my_function")

        assert result == "Callers found"
        mock_find_callers.assert_called_once_with("my_function", None)

    @patch("nexus_dev.server.HybridQueryRouter")
    @patch("nexus_dev.server.search_dependencies")
    async def test_smart_search_graph_imports(self, mock_search_deps, mock_router_cls):
        """Test smart_search routes to search_dependencies for import intent."""
        from nexus_dev.server import smart_search

        mock_router = MagicMock()
        mock_intent = QueryIntent(
            query_type=QueryType.GRAPH,
            extracted_entity="main.py",
            original_query="what imports main.py",
        )
        mock_router.route.return_value = mock_intent
        mock_router_cls.return_value = mock_router

        mock_search_deps.return_value = "Imports found"

        result = await smart_search("what imports main.py")

        assert result == "Imports found"
        # "what imports" -> direction="imported_by"
        mock_search_deps.assert_called_once_with(
            "main.py", direction="imported_by", project_id=None
        )

    @patch("nexus_dev.server.HybridQueryRouter")
    @patch("nexus_dev.server.get_recent_context")
    async def test_smart_search_kv_context(self, mock_get_context, mock_router_cls):
        """Test smart_search routes to get_recent_context for KV intent."""
        from nexus_dev.server import smart_search

        mock_router = MagicMock()
        mock_intent = QueryIntent(query_type=QueryType.KV, original_query="show recent context")
        mock_router.route.return_value = mock_intent
        mock_router_cls.return_value = mock_router

        mock_get_context.return_value = "Recent context..."

        # With session_id
        result = await smart_search("show recent context", session_id="sess-123")
        assert result == "Recent context..."
        mock_get_context.assert_called_once_with("sess-123")

    @patch("nexus_dev.server.HybridQueryRouter")
    async def test_smart_search_kv_missing_session(self, mock_router_cls):
        """Test smart_search handles missing session_id for KV intent."""
        from nexus_dev.server import smart_search

        mock_router = MagicMock()
        mock_intent = QueryIntent(query_type=QueryType.KV, original_query="show history")
        mock_router.route.return_value = mock_intent
        mock_router_cls.return_value = mock_router

        # Without session_id
        result = await smart_search("show history")
        assert "no 'session_id' was provided" in result

    @patch("nexus_dev.server.HybridQueryRouter")
    @patch("nexus_dev.server.search_knowledge")
    async def test_smart_search_vector_fallback(self, mock_search_knowledge, mock_router_cls):
        """Test smart_search falls back to vector search."""
        from nexus_dev.server import smart_search

        mock_router = MagicMock()
        mock_intent = QueryIntent(query_type=QueryType.VECTOR, original_query="how to install")
        mock_router.route.return_value = mock_intent
        mock_router_cls.return_value = mock_router

        mock_search_knowledge.return_value = "Docs found"

        result = await smart_search("how to install")
        assert result == "Docs found"
        mock_search_knowledge.assert_called_once_with("how to install", project_id=None)
