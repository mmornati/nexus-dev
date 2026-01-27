"""Tests for GraphStore module using FalkorDBLite."""

from nexus_dev.graph_store import GraphStore

# Note: graph_client fixture is provided by conftest.py with module-scoped server


def test_graph_store_initialization(graph_client) -> None:
    """Test graph store creates indices."""
    gs = GraphStore(graph_client, "test_graph")
    gs.connect()

    # Verify schema by inserting nodes
    # FalkorDB Cypher
    gs.query("CREATE (:File {path: 'test.py', language: 'python', size: 100})")

    gs.query(
        "CREATE (:Function {id: 'func1', name: 'my_func', "
        "signature: 'def my_func():', async_func: false, "
        "start_line: 1, end_line: 10})"
    )

    # Verify relationships
    gs.query(
        "MATCH (f:File), (fn:Function) "
        "WHERE f.path = 'test.py' AND fn.id = 'func1' "
        "CREATE (f)-[:DEFINES]->(fn)"
    )

    # Query back
    res = gs.query("MATCH (f:File)-[:DEFINES]->(fn:Function) RETURN f.path, fn.name")

    # FalkorDB result iteration (use .result_set)
    results = res.result_set
    assert len(results) == 1
    assert results[0][0] == "test.py"
    assert results[0][1] == "my_func"


def test_context_manager(graph_client) -> None:
    """Test context manager support."""
    with GraphStore(graph_client, "ctx_graph") as gs:
        gs.query("CREATE (:File {path: 'test.py', language: 'python'})")

    # Reopen (reuse client)
    gs2 = GraphStore(graph_client, "ctx_graph")
    res = gs2.query("MATCH (n:File) RETURN n.path")
    results = res.result_set
    assert len(results) == 1
    assert results[0][0] == "test.py"


def test_delete_graph(graph_client) -> None:
    """Test deleting graph."""
    gs = GraphStore(graph_client, "del_graph")
    gs.connect()
    gs.query("CREATE (:Node {id: 1})")

    gs.delete_graph()

    # Should be empty or not exist
    res = gs.query("MATCH (n) RETURN n")
    assert len(res.result_set) == 0
