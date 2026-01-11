# Configuring AI Agents for Nexus-Dev

This guide explains how to configure AI coding assistants to use Nexus-Dev's MCP tools for persistent memory.

## Quick Setup

### 1. Add to Your Project's AGENTS.md

Copy this section to your project's `AGENTS.md` file:

```markdown
## Nexus-Dev Knowledge Base

This project uses Nexus-Dev for persistent AI memory. When working on this codebase:

### Available MCP Tools

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Search all indexed content |
| `search_code` | Find functions, classes, methods |
| `search_docs` | Search documentation files |
| `search_lessons` | Find past problem/solution pairs |
| `record_lesson` | Save a debugging lesson |
| `index_file` | Index a file into the knowledge base |
| `get_project_context` | View project stats and recent lessons |

### Workflow Guidelines

1. **Start sessions** with `get_project_context()` to see what's indexed
2. **Search first** before implementing - use `search_code()` or `search_docs()`
3. **Check lessons** when debugging - use `search_lessons("<error message>")`
4. **Record lessons** after solving bugs - use `record_lesson(problem, solution)`

### Example Usage

\`\`\`
# Before implementing authentication
search_code("user authentication")

# When debugging
search_lessons("TypeError None")

# After fixing a bug
record_lesson(
    problem="Database timeout on large queries",
    solution="Added pagination with limit=100"
)
\`\`\`
```

---

### 2. Add Workflow Files (Optional)

Copy the `.agent/workflows/` directory from this repository to your project:

```bash
cp -r path/to/nexus-dev/.agent/workflows your-project/.agent/
```

This adds slash commands:
- `/start-session` - Initialize context at session start
- `/search-first` - Search before implementing
- `/record-lesson` - Record debugging lessons
- `/index-code` - Index new files

---

### 3. Initialize Nexus-Dev in Your Project

```bash
cd your-project
nexus-init --project-name "Your Project" --embedding-provider ollama
nexus-reindex --yes
```

---

## Full AGENTS.md Template

For a complete template, see [AGENTS_TEMPLATE.md](./AGENTS_TEMPLATE.md).
