# AGENTS.md - Nexus-Dev Project Guidelines

> Guidelines for AI coding assistants working on the Nexus-Dev codebase.

## ⚠️ CRITICAL: RAG Usage Policy

> **MANDATORY**: You MUST me, the AI, use Nexus-Dev RAG tools BEFORE answering ANY question about this project.

**Search-First Approach:**
1.  **Stop**: Do NOT answer questions about this codebase based solely on general knowledge.
2.  **Search**: ALWAYS use `search_knowledge`, `search_code`, `search_docs`, or `search_lessons` to find project-specific information.
3.  **Refine**: If your first search yields no results:
    *   **Broaden** your search query
    *   **Try different content types** (code → docs → lessons)
    *   **Break down** complex questions into smaller searchable parts
4.  **Acknowledge**: Only after exhausting RAG searches should you rely on general programming knowledge, and you **must acknowledge** that you couldn't find project-specific information.

**When to Search:**
- ✅ User asks about implementation details, architecture, or configuration
- ✅ User asks how to do something in this project
- ✅ User encounters an error or bug
- ✅ Before suggesting changes or refactors
- ✅ When unsure about existing code patterns

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

This project uses **Nexus-Dev** for persistent AI memory.

### Available MCP Tools

**Core Search:**
| Tool | Purpose |
|------|---------|
| `search_knowledge` | Search all indexed content (code, docs, lessons) |
| `search_code` | Find functions, classes, methods |
| `search_docs` | Search documentation/markdown files |
| `search_lessons` | Find past problem/solution pairs |
| `search_insights` | Find past mistakes, discoveries, optimizations |
| `search_implementations` | Find how features were built |

**Knowledge Capture:**
| Tool | Purpose |
|------|---------|
| `record_lesson` | Save a debugging lesson for future reference |
| `record_insight` | Capture LLM reasoning, mistakes, backtracking |
| `record_implementation` | Save implementation summaries with design decisions |
| `index_file` | Index a new file into the knowledge base |

**Tool & Agent Management:**
| Tool | Purpose |
|------|---------|
| `search_tools` | Find other available MCP tools |
| `get_tool_schema` | Get details for an MCP tool |
| `invoke_tool` | Run an MCP tool |
| `list_agents` | See available autonomous agents |
| `ask_agent` | Delegate a task to an agent |
| `refresh_agents` | Reload agent definitions from disk |

**Integrations:**
| Tool | Purpose |
|------|---------|
| `import_github_issues` | Import GitHub issues/PRs for search |
| `get_project_context` | View project stats and recent lessons |

### Workflow Guidelines

**At Session Start:**
```python
get_project_context()
```

**Global Code Search (Before Implementing):**
If you don't know where code lives, search by *responsibility*:
```python
# "Which file handles database connections?"
search_knowledge("database connection handling")

# "Find the class definition for User"
search_code("class User")
```

**When Debugging:**
```python
search_lessons("error message or problem description")
search_insights("mistake related to X")
```

**After Solving a Bug:**
```python
record_lesson(
    problem="<what went wrong>",
    solution="<how you fixed it>",
    context="<optional: file path or additional info>",
    problem_code="<buggy code snippet>",
    solution_code="<fixed code snippet>"
)
```

### Best Practices

1. **Search first** - Always check for existing implementations before writing new code.
2. **Record lessons** - After solving non-trivial bugs, save the solution for future reference.
3. **Index important files** - When creating new modules, index them for searchability.
4. **Check context** - Start sessions with `get_project_context()` to understand the codebase.

## Automatic Knowledge Capture

> **IMPORTANT**: The tools below should be called **automatically** during development to build a knowledge base of insights and implementations.

### When to Record Insights (Real-Time)

Call `record_insight` **immediately** when any of the following happens:

**Mistakes** - You tried something that didn't work:
```python
record_insight(
    category="mistake",
    description="<what went wrong>",
    reasoning="<why you thought it would work>",
    correction="<how you fixed it>"
)
```

**Backtracking** - You changed direction on an approach:
```python
record_insight(
    category="backtrack",
    description="<original approach>",
    reasoning="<why you're changing direction>",
    correction="<new approach>"
)
```

**Discoveries** - You found something non-obvious or useful:
```python
record_insight(
    category="discovery",
    description="<what you discovered>",
    reasoning="<why it's useful/important>"
)
```

**Optimizations** - You found a better way to do something:
```python
record_insight(
    category="optimization",
    description="<optimization made>",
    reasoning="<why it's better>",
    correction="<old approach>"
)
```

### When to Record Implementations (After Completion)

After finishing a feature, refactor, or significant work, call `record_implementation`:

```python
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

### GitHub Integration

Use `import_github_issues` to bring context from Issues and Pull Requests into your local knowledge base.

```python
import_github_issues(owner="my-org", repo="my-repo", state="open")
```

Then you can search them:
```python
search_knowledge("bug report about login", content_type="github_issue")
```

### Agent Usage

Nexus-Dev supports autonomous sub-agents.

**Discover Agents:**
```python
list_agents()
```

**Delegate Tasks:**
```python
ask_agent(agent_name="nexus_architect", task="Analyze the dependency structure")
```

**Reload Agents:**
If you edit an agent's YAML configuration, run:
```python
refresh_agents()
```
