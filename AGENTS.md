# AGENTS.md - Nexus-Dev Project Guidelines

> Guidelines for AI coding assistants working on the Nexus-Dev codebase.

## Project Overview

Nexus-Dev is a local RAG (Retrieval-Augmented Generation) MCP server that provides persistent memory for AI coding assistants. It indexes code, documentation, and lessons into a LanceDB vector database.

## Architecture

```
src/nexus_dev/
├── server.py         # FastMCP server with tool handlers
├── cli.py            # Click CLI commands (nexus-init, nexus-index, etc.)
├── database.py       # LanceDB wrapper (NexusDatabase, Document, SearchResult)
├── embeddings.py     # Embedding providers (OpenAI, Ollama)
├── config.py         # NexusConfig dataclass
└── chunkers/         # Language-specific code chunkers
    ├── base.py       # BaseChunker, ChunkerRegistry
    ├── python_chunker.py
    ├── java_chunker.py
    ├── javascript_chunker.py
    └── docs_chunker.py
```

## Development Commands

```bash
make dev          # Install in dev mode
make test         # Run tests
make test-cov     # Run tests with coverage
make lint         # Run ruff linter
make format       # Format code
make check        # Run all checks (lint + format + type-check)
```

## Before Pushing Code (CI Alignment)

**MANDATORY**: Always run these checks before committing/pushing to avoid CI failures:

```bash
make check   # Runs: lint + format-check + type-check (same as CI)
```

Or run them individually:

```bash
ruff check src/ tests/          # Linting (same as CI)
ruff format --check src/ tests/ # Format check (same as CI)
mypy src/                       # Type checking (same as CI)
```

### Auto-fix Lint Issues

To automatically fix linting and formatting issues:

```bash
make format  # Formats code + auto-fixes lint issues
```

> **Important**: The CI runs `ruff check src/ tests/`, `ruff format --check src/ tests/`, and `mypy src/`.  
> Local checks MUST pass before pushing or the CI build will fail.

## Coding Standards

- **Python 3.13+** with type hints
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pytest** with pytest-asyncio for async tests
- Follow PEP 8 naming conventions
- Use `async/await` for I/O operations

## Testing Guidelines

1. **Unit tests** go in `tests/unit/`
2. **Mock external dependencies** (LanceDB, HTTP clients) - don't make real API calls
3. Use `pytest.mark.asyncio` for async tests
4. Target 80%+ code coverage

## Key Patterns

### Adding a New MCP Tool

1. Add function in `server.py` with `@mcp.tool()` decorator
2. Add tests in `tests/unit/test_server.py`
3. Update `AGENTS.md` if user-facing

### Adding a New Chunker

1. Create `src/nexus_dev/chunkers/<language>_chunker.py`
2. Inherit from `BaseChunker`
3. Register in `ChunkerRegistry`
4. Add tests in `tests/chunkers/`

### Adding a CLI Command

1. Add function in `cli.py` with Click decorators
2. Add tests in `tests/unit/test_cli.py`

## Nexus-Dev Knowledge Base

This project uses **Nexus-Dev** for persistent AI memory. When working on this codebase:

### Available MCP Tools

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Search all indexed content (code, docs, lessons) |
| `search_code` | Find functions, classes, methods |
| `search_docs` | Search documentation/markdown files |
| `search_lessons` | Find past problem/solution pairs |
| `search_insights` | Find past mistakes, discoveries, optimizations |
| `search_implementations` | Find how features were built |
| `record_lesson` | Save a debugging lesson for future reference |
| `record_insight` | Capture LLM reasoning, mistakes, backtracking |
| `record_implementation` | Save implementation summaries with design decisions |
| `index_file` | Index a new file into the knowledge base |
| `get_project_context` | View project stats and recent lessons |

### Workflow Guidelines

**At Session Start:**
```
get_project_context()
```

**Before Implementing:**
```
search_code("<feature description>")
search_docs("<relevant topic>")
```

**When Debugging:**
```
search_lessons("<error message or problem description>")
```

**After Solving a Bug:**
```
record_lesson(
    problem="<what went wrong>",
    solution="<how you fixed it>",
    context="<optional: file path or additional info>"
)
```

### Best Practices

1. **Search first** - Always check for existing implementations before writing new code
2. **Record lessons** - After solving non-trivial bugs, save the solution for future reference
3. **Index important files** - When creating new modules, index them for searchability
4. **Check context** - Start sessions with `get_project_context()` to understand the codebase

## Automatic Knowledge Capture

> **IMPORTANT**: The tools below should be called **automatically** during development to build a knowledge base of insights and implementations.

### When to Record Insights (Real-Time)

Call `record_insight` **immediately** when any of the following happens:

**Mistakes** - You tried something that didn't work:
```
record_insight(
    category="mistake",
    description="<what went wrong>",
    reasoning="<why you thought it would work>",
    correction="<how you fixed it>"
)
```

Examples:
- Used a library version incompatible with Python version
- Chose an approach that had performance issues
- Misunderstood an API and had to rewrite

**Backtracking** - You changed direction on an approach:
```
record_insight(
    category="backtrack",
    description="<original approach>",
    reasoning="<why you're changing direction>",
    correction="<new approach>"
)
```

**Discoveries** - You found something non-obvious or useful:
```
record_insight(
    category="discovery",
    description="<what you discovered>",
    reasoning="<why it's useful/important>"
)
```

**Optimizations** - You found a better way to do something:
```
record_insight(
    category="optimization",
    description="<optimization made>",
    reasoning="<why it's better>",
    correction="<old approach>"
)
```

### When to Record Implementations (After Completion)

After finishing a feature, refactor, or significant work, call `record_implementation`:

```
record_implementation(
    title="<short title>",
    summary="<what was built - 1-3 sentences>",
    approach="<how it was built - technical approach>",
    design_decisions=[
        "Decision 1: rationale",
        "Decision 2: rationale"
    ],
    files_changed=["file1.py", "file2.py"]
)
```

**When to call**:
- After completing a user-requested feature
- After a significant refactor
- After fixing a complex bug
- When finishing implementation from an approved plan

### Search Before Recording

Before recording, **search first** to avoid duplicates:
- `search_insights("similar problem")` - Check if this was already encountered
- `search_implementations("similar feature")` - Check if this was already built
