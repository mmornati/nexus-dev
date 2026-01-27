"""Tests for code_graph module - AST-based code graph extraction."""

from pathlib import Path

from nexus_dev.code_graph import PythonGraphBuilder
from nexus_dev.graph_store import GraphStore

# Note: graph_client fixture is provided by conftest.py with module-scoped server


class TestPythonGraphBuilder:
    """Tests for PythonGraphBuilder class."""

    def test_index_file_extracts_functions(self, graph_client, tmp_path: Path) -> None:
        """Test that functions are extracted with signatures."""
        # Create test file
        test_file = tmp_path / "example.py"
        test_file.write_text("""
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a, b):
    return a + b
""")

        gs = GraphStore(graph_client, "test_func_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["functions"] == 2
        assert stats["classes"] == 0

        # Verify functions in graph
        res = gs.query("MATCH (f:Function) RETURN f.name, f.signature ORDER BY f.name")
        results = res.result_set
        assert len(results) == 2
        assert results[0][0] == "add"
        assert results[1][0] == "greet"
        # Check signature includes def
        assert "def greet" in results[1][1]

    def test_index_file_extracts_classes(self, graph_client, tmp_path: Path) -> None:
        """Test that classes are extracted with line numbers."""
        test_file = tmp_path / "models.py"
        test_file.write_text("""
class BaseModel:
    pass

class UserModel:
    def __init__(self):
        self.name = ""
""")

        gs = GraphStore(graph_client, "test_class_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["classes"] == 2
        # __init__ is inside UserModel
        assert stats["functions"] == 1

        # Verify classes in graph
        res = gs.query("MATCH (c:Class) RETURN c.name ORDER BY c.name")
        results = res.result_set
        assert len(results) == 2
        assert results[0][0] == "BaseModel"
        assert results[1][0] == "UserModel"

    def test_index_file_extracts_imports(self, graph_client, tmp_path: Path) -> None:
        """Test that import relationships are tracked."""
        test_file = tmp_path / "service.py"
        test_file.write_text("""
import os
from pathlib import Path
from typing import List
""")

        gs = GraphStore(graph_client, "test_import_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["imports"] == 3

        # Verify import relationships
        res = gs.query("MATCH (f:File)-[:IMPORTS]->(m:File) RETURN f.path, m.path")
        results = res.result_set
        assert len(results) == 3

    def test_function_call_tracking(self, graph_client, tmp_path: Path) -> None:
        """Test that CALLS relationships are created."""
        test_file = tmp_path / "calculator.py"
        test_file.write_text("""
def helper():
    return 42

def main():
    x = helper()
    return x
""")

        gs = GraphStore(graph_client, "test_call_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["functions"] == 2
        assert stats["calls"] >= 1

        # Verify call relationship
        res = gs.query(
            "MATCH (caller:Function)-[:CALLS]->(callee:Function) RETURN caller.name, callee.name"
        )
        results = res.result_set
        assert len(results) == 1
        assert results[0][0] == "main"
        assert results[0][1] == "helper"

    def test_class_inheritance_tracking(self, graph_client, tmp_path: Path) -> None:
        """Test that INHERITS relationships are created."""
        test_file = tmp_path / "handlers.py"
        test_file.write_text("""
class BaseHandler:
    pass

class UserHandler(BaseHandler):
    pass

class AdminHandler(UserHandler):
    pass
""")

        gs = GraphStore(graph_client, "test_inherit_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["classes"] == 3

        # Verify inheritance relationships
        res = gs.query(
            "MATCH (child:Class)-[:INHERITS]->(parent:Class) "
            "RETURN child.name, parent.name ORDER BY child.name"
        )
        results = res.result_set
        assert len(results) == 2
        # AdminHandler -> UserHandler
        assert results[0][0] == "AdminHandler"
        assert results[0][1] == "UserHandler"
        # UserHandler -> BaseHandler
        assert results[1][0] == "UserHandler"
        assert results[1][1] == "BaseHandler"

    def test_async_function_support(self, graph_client, tmp_path: Path) -> None:
        """Test that async functions are handled correctly."""
        test_file = tmp_path / "async_service.py"
        test_file.write_text("""
async def fetch_data(url: str):
    return f"Data from {url}"

async def process():
    data = await fetch_data("http://example.com")
    return data
""")

        gs = GraphStore(graph_client, "test_async_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["functions"] == 2

        # Verify async flag
        res = gs.query("MATCH (f:Function) WHERE f.async_func = true RETURN f.name ORDER BY f.name")
        results = res.result_set
        assert len(results) == 2
        assert results[0][0] == "fetch_data"
        assert results[1][0] == "process"

        # Check signature includes async
        res = gs.query("MATCH (f:Function {name: 'fetch_data'}) RETURN f.signature")
        results = res.result_set
        assert "async def" in results[0][0]

    def test_file_defines_relationship(self, graph_client, tmp_path: Path) -> None:
        """Test that File-DEFINES->Function/Class relationships exist."""
        test_file = tmp_path / "module.py"
        test_file.write_text("""
def my_function():
    pass

class MyClass:
    pass
""")

        gs = GraphStore(graph_client, "test_defines_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        builder.index_file(test_file)

        # Verify File -> Function relationship
        res = gs.query("MATCH (f:File)-[:DEFINES]->(fn:Function) RETURN f.path, fn.name")
        func_results = res.result_set
        assert len(func_results) == 1
        assert func_results[0][1] == "my_function"

        # Verify File -> Class relationship
        res = gs.query("MATCH (f:File)-[:DEFINES]->(c:Class) RETURN f.path, c.name")
        class_results = res.result_set
        assert len(class_results) == 1
        assert class_results[0][1] == "MyClass"


class TestPythonGraphBuilderEdgeCases:
    """Edge case tests for PythonGraphBuilder."""

    def test_empty_file(self, graph_client, tmp_path: Path) -> None:
        """Test handling of empty Python file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        gs = GraphStore(graph_client, "test_empty_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["functions"] == 0
        assert stats["classes"] == 0
        assert stats["imports"] == 0

    def test_nested_functions(self, graph_client, tmp_path: Path) -> None:
        """Test that nested functions are extracted."""
        test_file = tmp_path / "nested.py"
        test_file.write_text("""
def outer():
    def inner():
        pass
    return inner
""")

        gs = GraphStore(graph_client, "test_nested_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        # Both outer and inner should be counted
        assert stats["functions"] == 2

    def test_method_extraction(self, graph_client, tmp_path: Path) -> None:
        """Test that class methods are extracted as functions."""
        test_file = tmp_path / "class_methods.py"
        test_file.write_text("""
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
""")

        gs = GraphStore(graph_client, "test_method_graph")
        gs.connect()
        builder = PythonGraphBuilder(gs, "test-proj")

        stats = builder.index_file(test_file)

        assert stats["classes"] == 1
        assert stats["functions"] == 2  # add, subtract
