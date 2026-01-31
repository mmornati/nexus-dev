"""Tests for JavaScript/TypeScript code graph extraction."""

from unittest.mock import MagicMock, call

import pytest

from nexus_dev.code_graph_js import JSGraphBuilder


@pytest.fixture
def mock_graph():
    """Mock GraphStore."""
    return MagicMock()


@pytest.fixture
def builder(mock_graph):
    """Create JSGraphBuilder with mock graph."""
    return JSGraphBuilder(mock_graph, "test-proj")


def test_index_file_imports(builder, mock_graph, tmp_path):
    """Test extraction of imports."""
    content = """
    import React from 'react';
    import { useState } from 'react';
    import defaultExport, { named } from './components/Button';
    const axios = require('axios');
    require('./utils');
    """

    test_file = tmp_path / "imports.js"
    test_file.write_text(content, encoding="utf-8")

    stats = builder.index_file(test_file)

    assert stats["imports"] == 5

    # Verify import logic called correct graph queries
    # We can check specific calls or just count
    # 'react' appears twice, 'axios' once, local imports twice

    # Check if 'react' import hook was triggered
    # We look for calls to _add_import -> graph.query
    # Since _add_import calls graph.query twice (one for target file node, one for relationship),
    # we can verify arguments.

    # Check for node_modules import
    node_modules_call = call(
        """
            MERGE (f:File {path: $path})
            SET f.project_id = $project_id
            """,
        {"path": "node_modules/react", "project_id": "test-proj"},
    )
    assert node_modules_call in mock_graph.query.call_args_list


def test_index_file_functions(builder, mock_graph, tmp_path):
    """Test extraction of function definitions."""
    content = """
    function regularFunc(a, b) {
        return a + b;
    }

    const arrowFunc = (x) => x * 2;

    const expressionFunc = function() {
        return true;
    };

    async function asyncFunc() {
        await something();
    }
    """

    test_file = tmp_path / "funcs.js"
    test_file.write_text(content, encoding="utf-8")

    stats = builder.index_file(test_file)

    assert stats["functions"] == 4

    # Verify function nodes created
    # Check for regularFunc
    func_node_call = call(
        """
            MERGE (f:Function {id: $id})
            SET f.name = $name, f.file_path = $path
            """,
        {"id": f"{test_file}:regularFunc", "name": "regularFunc", "path": str(test_file)},
    )
    assert func_node_call in mock_graph.query.call_args_list


def test_index_file_classes(builder, mock_graph, tmp_path):
    """Test extraction of classes and inheritance."""
    content = """
    class BaseComponent {
        render() {}
    }

    class UserProfile extends BaseComponent {
        constructor() {
            super();
        }
    }

    class Standalone {}
    """

    test_file = tmp_path / "classes.ts"
    test_file.write_text(content, encoding="utf-8")

    stats = builder.index_file(test_file)

    assert stats["classes"] == 3

    # Verify inheritance relationship
    inheritance_call = call(
        """
                MATCH (child:Class {id: $child})
                MATCH (parent:Class) WHERE parent.name = $parent_name
                MERGE (child)-[:INHERITS]->(parent)
                """,
        {"child": f"{test_file}:UserProfile", "parent_name": "BaseComponent"},
    )
    # The SQL query formatting might vary slightly.
    # We use exact string matches in the mock, so we need to match exactly.
    assert inheritance_call in mock_graph.query.call_args_list


def test_language_detection(builder, mock_graph, tmp_path):
    """Test language property setting based on extension."""

    # JS file
    js_file = tmp_path / "test.js"
    js_file.write_text("console.log('hi');", encoding="utf-8")
    builder.index_file(js_file)

    js_call = call(
        """
            MERGE (f:File {path: $path})
            SET f.language = $language, f.project_id = $project_id
            """,
        {"path": str(js_file), "language": "javascript", "project_id": "test-proj"},
    )
    assert js_call in mock_graph.query.call_args_list

    # TS file
    ts_file = tmp_path / "test.tsx"
    ts_file.write_text("const x: number = 1;", encoding="utf-8")
    builder.index_file(ts_file)

    ts_call = call(
        """
            MERGE (f:File {path: $path})
            SET f.language = $language, f.project_id = $project_id
            """,
        {"path": str(ts_file), "language": "typescript", "project_id": "test-proj"},
    )
    assert ts_call in mock_graph.query.call_args_list
