"""Tests for GraphStore module."""

from pathlib import Path

from nexus_dev.graph_store import GraphStore


def test_graph_store_initialization(tmp_path: Path) -> None:
    """Test graph store creates database and schema."""
    db_path = tmp_path / "graph_db"

    gs = GraphStore(db_path)
    gs.connect()

    try:
        assert db_path.exists()

        # Verify schema by inserting nodes
        gs.query("CREATE (:File {path: 'test.py', language: 'python', size: 100})")
        gs.query(
            "CREATE (:Function {id: 'func1', name: 'my_func', "
            "signature: 'def my_func():', async_func: false, "
            "start_line: 1, end_line: 10})"
        )
        gs.query("CREATE (:Class {id: 'class1', name: 'MyClass', start_line: 20, end_line: 30})")

        # Verify schema by inserting relationships
        gs.query(
            "MATCH (f:File), (fn:Function) "
            "WHERE f.path = 'test.py' AND fn.id = 'func1' "
            "CREATE (f)-[:DEFINES]->(fn)"
        )

        # Query back
        res = gs.query("MATCH (f:File)-[:DEFINES]->(fn:Function) RETURN f.path, fn.name")
        while res.has_next():
            row = res.get_next()
            assert row[0] == "test.py"
            assert row[1] == "my_func"

    finally:
        gs.close()


def test_context_manager(tmp_path: Path) -> None:
    """Test context manager support."""
    db_path = tmp_path / "graph_db"

    with GraphStore(db_path) as gs:
        gs.query("CREATE (:File {path: 'test.py', language: 'python'})")

    # Reopen and check
    with GraphStore(db_path) as gs:
        res = gs.query("MATCH (n:File) RETURN n.path")
        assert res.has_next()
        assert res.get_next()[0] == "test.py"


def test_idempotent_close(tmp_path: Path) -> None:
    """Test close can be called multiple times."""
    db_path = tmp_path / "graph_db"
    gs = GraphStore(db_path)
    gs.connect()
    gs.close()
    gs.close()
