# Contributing

Guide for contributing to Nexus-Dev development.

---

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) or pip
- Git

### Clone and Install

```bash
# Clone repository
git clone https://github.com/mmornati/nexus-dev.git
cd nexus-dev

# Option A: Use Makefile (recommended)
make setup
source .venv/bin/activate

# Option B: Manual
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Verify Installation

```bash
nexus-init --help
make check
```

---

## Development Workflow

### Branch Strategy

```
main          # Stable releases
├── feat/xxx  # New features
├── fix/xxx   # Bug fixes
└── docs/xxx  # Documentation
```

### Making Changes

1. **Create branch:**
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make changes**

3. **Run checks:**
   ```bash
   make check  # Lint + format + type check
   make test   # Run tests
   ```

4. **Commit:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feat/my-feature
   ```

---

## Code Style

### Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for formatting and linting:

```bash
make format   # Auto-format
make lint     # Check for issues
```

### Type Checking

We use [mypy](https://mypy-lang.org/):

```bash
make type-check
```

### Docstrings

Use Google-style docstrings:

```python
def search_code(query: str, limit: int = 5) -> list[Document]:
    """Search for code in the knowledge base.
    
    Args:
        query: Natural language search query.
        limit: Maximum number of results.
        
    Returns:
        List of matching documents.
        
    Raises:
        ValueError: If query is empty.
    """
```

---

## Testing

### Run Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/unit/test_database.py -v

# Specific test
pytest tests/unit/test_database.py::test_search -v
```

### Test Structure

```
tests/
├── unit/           # Unit tests (no external deps)
│   ├── test_chunkers.py
│   ├── test_database.py
│   └── test_embeddings.py
└── integration/    # Integration tests
    └── test_cli.py
```

### Writing Tests

```python
import pytest
from nexus_dev.chunkers import ChunkerRegistry

def test_python_chunker_extracts_functions():
    """Test that Python chunker finds function definitions."""
    content = '''
def hello():
    return "world"
'''
    chunks = ChunkerRegistry.chunk_file(Path("test.py"), content)
    
    assert len(chunks) == 1
    assert chunks[0].name == "hello"
    assert chunks[0].chunk_type == ChunkType.FUNCTION
```

---

## Adding Features

### Adding a Language Chunker

1. **Create chunker file:**
   ```python
   # src/nexus_dev/chunkers/rust.py
   from .base import CodeChunk, ChunkType
   
   def chunk_rust(file_path: Path, content: str) -> list[CodeChunk]:
       # Parse and extract chunks
       pass
   ```

2. **Register in registry:**
   ```python
   # src/nexus_dev/chunkers/__init__.py
   ChunkerRegistry.register([".rs"], chunk_rust)
   ```

3. **Add tests:**
   ```python
   # tests/unit/test_chunkers.py
   def test_rust_chunker():
       # Test implementation
   ```

### Adding an MCP Tool

1. **Add to server.py:**
   ```python
   @mcp.tool()
   async def my_new_tool(param: str) -> str:
       """Tool description.
       
       Args:
           param: Parameter description.
           
       Returns:
           Result description.
       """
       # Implementation
   ```

2. **Add documentation:**
   Update relevant docs in `docs/tools/`.

3. **Add tests:**
   ```python
   # tests/unit/test_server.py
   async def test_my_new_tool():
       result = await my_new_tool("test")
       assert "expected" in result
   ```

### Adding a CLI Command

1. **Add to cli.py:**
   ```python
   @cli.command("my-command")
   @click.option("--option", help="Option description")
   def my_command(option: str) -> None:
       """Command description."""
       # Implementation
   ```

2. **Register in pyproject.toml** (if new entry point).

3. **Add documentation:**
   Create `docs/cli/my-command.md`.

---

## Documentation

### Building Docs

```bash
# Install dependencies
pip install mkdocs-material mkdocs-git-revision-date-localized-plugin

# Serve locally
mkdocs serve

# Build
mkdocs build
```

### Documentation Structure

```
docs/
├── index.md              # Home page
├── quickstart.md         # Getting started
├── getting-started/      # Installation, config
├── cli/                  # CLI commands
├── tools/                # MCP tools
├── workflows/            # Usage guides
├── advanced/             # Deep dives
└── reference/            # Architecture, troubleshooting
```

---

## Release Process

### Version Bumping

Version is in `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

### Creating a Release

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create PR and merge to main
4. Tag release:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
5. GitHub Action handles PyPI publish

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/mmornati/nexus-dev/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mmornati/nexus-dev/discussions)

---

## Code of Conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](https://github.com/mmornati/nexus-dev/blob/main/CODE_OF_CONDUCT.md).
