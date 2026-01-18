# nexus-init

Initialize Nexus-Dev in the current repository.

---

## Synopsis

```bash
nexus-init [OPTIONS]
```

---

## Description

Creates the configuration file, lessons directory, and optionally installs the pre-commit hook for automatic indexing.

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--project-name` | TEXT | (prompt) | Human-readable name for the project |
| `--embedding-provider` | CHOICE | `openai` | Embedding provider (`openai` or `ollama`) |
| `--install-hook / --no-hook` | FLAG | `False` | Install pre-commit hook for automatic indexing |
| `--link-hook` | FLAG | `False` | Install hook linked to parent project configuration (multi-repo) |
| `--discover-repos` | FLAG | `False` | Auto-discover git repositories and offer to install hooks |

---

## Examples

### Basic initialization

```bash
cd my-project
nexus-init --project-name "My Project"
```

**Output:**

```
✅ Created nexus_config.json
✅ Created .nexus/lessons/
✅ Created database directory at /Users/you/.local/share/nexus-dev/lancedb

Configure .gitignore for .nexus folder? [allow-lessons/ignore-all/skip] (allow-lessons): 
✅ Updated .gitignore (allow-lessons)

Project ID: 550e8400-e29b-41d4-a716-446655440000

⚠️  Using OpenAI embeddings. Ensure OPENAI_API_KEY is set.

----------------------------------------------------------------
🤖 COPY-PASTE THIS INTO YOUR AGENT'S SYSTEM PROMPT OR RULES:
----------------------------------------------------------------

## Nexus-Dev Knowledge Base

You have access to a local RAG system for this project.

**Project ID:** 550e8400-e29b-41d4-a716-446655440000

**MANDATORY**: You MUST use `nexus-dev` tools BEFORE answering questions about this code.
1. `search_knowledge("My Project <query>")` - Search code, docs, and lessons
2. `search_code("<class/function_name>")` - Find specific code definitions
3. `search_lessons("<error/problem>")` - Check for past solutions
4. `record_lesson(...)` - Save solutions after fixing non-trivial bugs

**Best Practice:**
- Start every session with `get_project_context()`
- Search before writing code
- Record insights with `record_insight()`

----------------------------------------------------------------
```

### With Ollama (local embeddings)

```bash
nexus-init --project-name "Private Project" --embedding-provider ollama
```

### With pre-commit hook

```bash
nexus-init --project-name "My Project" --install-hook
```

The pre-commit hook automatically indexes modified code files on each commit.

### Multi-repository projects

#### Install hook in sub-repository

```bash
cd sub-repo
nexus-init --link-hook
```

**Output:**
```
✅ Installed pre-commit hook (linked to parent-project/)
✅ Linked to parent project: My Project
   Project ID: 550e8400-e29b-41d4-a716-446655440000
   Project Root: /path/to/parent-project
```

#### Auto-discover all repositories

```bash
cd parent-project
nexus-init --discover-repos
```

**Output:**
```
Found 3 git repositories:
  📁 frontend
  📁 backend
  📁 shared

Install hooks in all repositories? [y/N]: y
  ✅ frontend
  ✅ backend
  ✅ shared

✅ Installed hooks in 3/3 repositories
   All repositories linked to project: My Project
```

See [Multi-Repository Projects](../advanced/multi-repo-projects.md) for detailed guide.

---

## Files Created

| Path | Purpose |
|------|---------|
| `nexus_config.json` | Project configuration |
| `.nexus/lessons/` | Directory for recorded lessons |
| `.nexus/lessons/.gitkeep` | Ensures directory is tracked |
| `~/.local/share/nexus-dev/lancedb/` | Vector database |

---

## .gitignore Configuration

During initialization, you're prompted to configure `.gitignore`:

| Choice | Effect |
|--------|--------|
| `allow-lessons` | Track lessons but ignore DB (recommended) |
| `ignore-all` | Ignore entire `.nexus/` directory |
| `skip` | Don't modify `.gitignore` |

---

## See Also

- [nexus-index](index-cmd.md) - Index files after initialization
- [nexus-status](status.md) - Check project setup
- [Multi-Repository Projects](../advanced/multi-repo-projects.md) - Multi-repo setup guide
- [Configuration Guide](../getting-started/configuration.md) - Detailed configuration options
