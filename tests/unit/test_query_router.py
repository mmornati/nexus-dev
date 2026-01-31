"""Tests for HybridQueryRouter.

Verifies that queries are correctly classified into Graph, KV, and Vector intents,
and that entities are extracted accurately.
"""

from nexus_dev.query_router import HybridQueryRouter, QueryType


def test_router_graph_detection() -> None:
    """Test graph query detection and entity extraction."""
    router = HybridQueryRouter()

    # Callers
    result = router.route("Who calls validate_user?")
    assert result.query_type == QueryType.GRAPH
    assert result.extracted_entity == "validate_user"

    result = router.route("Show me callers of auth.login")
    assert result.query_type == QueryType.GRAPH
    assert result.extracted_entity == "auth.login"

    # Imports
    result = router.route("What imports main.py?")
    assert result.query_type == QueryType.GRAPH
    assert result.extracted_entity == "main.py"

    result = router.route("Dependencies of src/utils.ts")
    assert result.query_type == QueryType.GRAPH
    assert result.extracted_entity == "src/utils.ts"

    # Implementations
    result = router.route("Who implements BaseHandler?")
    assert result.query_type == QueryType.GRAPH
    assert result.extracted_entity == "BaseHandler"


def test_router_kv_detection() -> None:
    """Test KV (session context) query detection."""
    router = HybridQueryRouter()

    kv_queries = [
        "What was the last message?",
        "Show me recent context",
        "Summarize the session history",
        "What did we just do?",
    ]

    for query in kv_queries:
        result = router.route(query)
        assert result.query_type == QueryType.KV, f"Failed for query: {query}"


def test_router_vector_fallback() -> None:
    """Test fallback to vector search for general queries."""
    router = HybridQueryRouter()

    vector_queries = [
        "How do I configure the database?",
        "Explain the authentication flow",
        "Where is the error handling logic?",
        "Find code about logging",
    ]

    for query in vector_queries:
        result = router.route(query)
        assert result.query_type == QueryType.VECTOR, f"Failed for query: {query}"


def test_router_complex_cases() -> None:
    """Test edge cases and complex queries."""
    router = HybridQueryRouter()

    # Graph query with extra words
    result = router.route("Please tell me who calls get_user_id now")
    # Our regex is `who calls ([a-zA-Z0-9_.]+)` -> takes "get_user_id"
    # Ensure it doesn't capture "now" if regex is tight or loose
    # Current regex: `who calls\s+([a-zA-Z0-9_.]+)`
    # It might match "get_user_id" and ignore "now" if we use search() not fullmatch()
    assert result.query_type == QueryType.GRAPH
    assert result.extracted_entity == "get_user_id"

    # Case insensitivity
    result = router.route("WHO CALLS MYFUNCTION")
    assert result.query_type == QueryType.GRAPH
    assert result.extracted_entity == "MYFUNCTION"  # Regex capture group preserves case?
    # Actually our regex uses re.IGNORECASE search, but capture group returns original text slice?
    # Typically match.group(1) returns the substring from the original string.
