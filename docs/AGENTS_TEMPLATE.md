# AGENTS.md Template for Nexus-Dev Projects

> Copy this file to your project as `AGENTS.md` and customize for your codebase.

## Project Overview

<!-- Describe your project here -->

[Your project description]

## Architecture

<!-- Add your project's architecture -->

```
src/
├── ...
```

## Development Commands

<!-- Add your project's common commands -->

```bash
# Example commands
make dev
make test
```

---

## Nexus-Dev Knowledge Base

This project uses **Nexus-Dev** for persistent AI memory. When working on this codebase:

### Available MCP Tools

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Search all indexed content (code, docs, lessons) |
| `search_code` | Find functions, classes, methods |
| `search_docs` | Search documentation/markdown files |
| `search_lessons` | Find past problem/solution pairs |
| `record_lesson` | Save a debugging lesson for future reference |
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
