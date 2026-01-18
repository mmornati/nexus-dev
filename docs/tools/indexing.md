# Indexing Tools

Tools for adding content to the knowledge base.

---

## index_file

Index a file into the knowledge base with language-aware chunking.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | ✅ | - | Path to file (relative or absolute) |
| `content` | string | | (read from disk) | Optional file content |
| `project_id` | string | | (current) | Project identifier |

### Supported File Types

| Extension | Language | Chunking Strategy |
|-----------|----------|-------------------|
| `.py`, `.pyw` | Python | AST-based (functions, classes) |
| `.js`, `.jsx`, `.mjs` | JavaScript | Tree-sitter |
| `.ts`, `.tsx`, `.mts` | TypeScript | Tree-sitter |
| `.java` | Java | Tree-sitter |
| `.md`, `.markdown` | Markdown | Section-based |
| `.rst` | reStructuredText | Section-based |
| `.txt` | Plain text | Paragraph-based |

### Example

```
index_file("src/new_module.py")
```

**Response:**

```
Indexed 12 chunks from src/new_module.py:
- 3 functions
- 2 classes
- 7 methods
```

---

## get_project_context

Get project statistics and recent lessons.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `project_id` | string | | (current) | Project identifier |
| `limit` | int | | `10` | Max recent items |

### Example

```
get_project_context()
```

**Response:**

```markdown
## Project Context: my-project

**Project ID:** 550e8400-e29b-41d4-a716-446655440000

### Statistics
| Type | Count |
|------|-------|
| Code chunks | 156 |
| Documentation | 42 |
| Lessons | 8 |
| Insights | 5 |
| Implementations | 3 |

### Recent Lessons
1. **TypeError with async context manager** (2024-01-15)
   - Fixed by adding `async with` instead of `with`

2. **Database connection pool exhaustion** (2024-01-14)
   - Increased pool size and added connection timeout
```

!!! tip "Session Start"
    Call `get_project_context()` at the start of each coding session to load relevant context.

---

## import_github_issues

Import GitHub issues and PRs into the knowledge base.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `repo` | string | ✅ | - | Repository name |
| `owner` | string | ✅ | - | Repository owner |
| `limit` | int | | `10` | Max items to import |
| `state` | string | | `"all"` | Filter: `open`, `closed`, `all` |

### Prerequisites

Requires the GitHub MCP server configured in `.nexus/mcp_config.json`.

### Example

```
import_github_issues(repo="nexus-dev", owner="mmornati", limit=50)
```

**Response:**

```
Imported 50 items from mmornati/nexus-dev:
- 32 issues
- 18 pull requests
```

---

## When to Index

| Scenario | Action |
|----------|--------|
| New file created | Index automatically with pre-commit hook |
| Major refactoring | Use `nexus-reindex` CLI |
| Ad-hoc file addition | Use `index_file` tool |
| After git pull | Consider re-indexing changed files |
