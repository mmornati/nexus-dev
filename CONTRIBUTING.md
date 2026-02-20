# Contributing to Nexus-Dev

Thank you for your interest in contributing to Nexus-Dev!

## Adding a New Language Chunker

Nexus-Dev uses [tree-sitter](https://tree-sitter.github.io/) for parsing source code. Adding support for a new language is straightforward:

### 1. Create the Chunker File

Create `src/nexus_dev/chunkers/<language>_chunker.py`:

```python
"""Chunker for <Language> files using tree-sitter."""

from tree_sitter_language_pack import get_parser

from .base import BaseChunker, ChunkType, CodeChunk


class <Language>Chunker(BaseChunker):
    """Tree-sitter based chunker for <Language>."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".ext1", ".ext2"]  # File extensions

    def __init__(self) -> None:
        self.parser = get_parser("<language>")  # tree-sitter language name

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        tree = self.parser.parse(content.encode())
        chunks: list[CodeChunk] = []

        # Define tree-sitter queries for your language
        # See: https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries
        
        query = self.parser.language.query("""
            (function_definition name: (identifier) @fn.name) @fn
            (class_definition name: (identifier) @cls.name) @cls
        """)

        for node, name in query.captures(tree.root_node):
            chunk = self._node_to_chunk(node, content, file_path)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _node_to_chunk(
        self, node: Any, content: str, file_path: str
    ) -> CodeChunk | None:
        lines = content.split("\n")
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        
        chunk_content = "\n".join(lines[start_line:end_line + 1])
        
        return CodeChunk(
            content=chunk_content,
            chunk_type=ChunkType.FUNCTION,  # or CLASS, METHOD, etc.
            name=node.child_by_field_name("name").text.decode(),
            start_line=start_line + 1,
            end_line=end_line + 1,
            language="<language>",
            file_path=file_path,
        )
```

### 2. Register the Chunker

Add to `src/nexus_dev/chunkers/__init__.py`:

```python
from .<language>_chunker import <Language>Chunker

# In _register_default_chunkers():
ChunkerRegistry.register(<Language>Chunker())
```

### 3. Test Your Chunker

Create `tests/chunkers/test_<language>_chunker.py`:

```python
import pytest
from nexus_dev.chunkers.<language>_chunker import <Language>Chunker

def test_chunk_function():
    chunker = <Language>Chunker()
    content = '''// Your test code here'''
    
    chunks = chunker.chunk_content(content, "test.ext")
    
    assert len(chunks) == 1
    assert chunks[0].name == "expected_name"
    assert chunks[0].chunk_type == ChunkType.FUNCTION
```

### 4. Update Documentation

Add an entry to the README.md supported languages table.

## Finding Tree-Sitter Query Patterns

Use the [tree-sitter playground](https://tree-sitter.github.io/tree-sitter/playground) to explore your language's AST and build queries.

Common node types by language:

| Language | Functions | Classes | Methods |
|----------|-----------|---------|---------|
| Python | `function_definition` | `class_definition` | `function_definition` (nested) |
| JavaScript | `function_declaration`, `arrow_function` | `class_declaration` | `method_definition` |
| TypeScript | Same as JS + `function_signature` | Same as JS | Same as JS |
| Java | `method_declaration` | `class_declaration` | `method_declaration` |
| Go | `function_declaration` | `type_declaration` | `method_declaration` |
| Rust | `function_item` | `struct_item`, `impl_item` | `function_item` (in impl) |

## Development Setup

Since Nexus-Dev is not yet published to PyPI/Docker Hub, developers must build from source.

### Setup using Makefile (Recommended)

```bash
# Clone repository
git clone https://github.com/mmornati/nexus-dev.git
cd nexus-dev

# Setup full development environment (pyenv + venv + deps)
make setup

# Activate the virtual environment
source .venv/bin/activate
```

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Full dev environment setup (pyenv + venv + deps) |
| `make install-dev` | Install package with dev dependencies |
| `make lint` | Run ruff linter |
| `make format` | Format code + auto-fix lint issues |
| `make check` | Run all CI checks (lint + format + type-check) |
| `make test` | Run tests |
| `make test-cov` | Run tests with coverage report |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make docker-test` | Run tests inside Docker container |
| `make help` | Show all available commands |

### MCP Configuration (Development Mode)

Configure your AI agent to use the locally-built server. **This single configuration works for ALL your indexed projects!**

**For Claude Desktop / Cursor / Windsurf:**

```json
{
  "mcpServers": {
    "nexus-dev": {
      "command": "/path/to/nexus-dev/.venv/bin/python",
      "args": ["-m", "nexus_dev.server"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

> **How it works**: The server now defaults to searching **all indexed projects** when no specific project context is active. You don't need to configure `cwd` or create separate MCP entries for each project.

**Using Docker:**

The `/workspace` mount is the server's working directory. It looks for `nexus_config.json` and `.nexus/lessons/` there. Mount your project (or parent directory) to `/workspace`:

```json
{
  "mcpServers": {
    "nexus-dev": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/path/to/project:/workspace:ro",
        "-v", "nexus-dev-data:/data/nexus-dev",
        "-e", "OPENAI_API_KEY",
        "nexus-dev:latest"
      ]
    }
  }
}
```

## Adding New Features

If you are adding new features, such as new **Embedding Providers** or **Gateway MCP Tools**, make sure your PR includes:
1. Necessary tests and documentation updates.
2. Changes to `README.md` and related advanced docs (e.g., `docs/embedding-providers.md`).

## Code Style

- Format your code with `make format`
- Verify everything passes the linter and type checker with `make check`

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests and hooks pass (`make check` and `make test`, optionally `make docker-test`)
5. Submit a pull request

Thank you for contributing! 🎉
