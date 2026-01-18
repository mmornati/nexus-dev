# Quick Start Guide

Get up and running with Nexus-Dev in 5 minutes.

!!! info "No Source Code Needed"
    This guide is for users who want to **use** Nexus-Dev. You don't need to clone the repository.

---

## Prerequisites

- **Python 3.13+**
- **MCP-compatible IDE**: Cursor, VS Code with Copilot, Claude Desktop
- **Embedding API**: OpenAI API key or Ollama running locally

---

## Step 1: Install Nexus-Dev

Install globally to avoid conflicts with project virtual environments:

=== "pipx (Recommended)"

    ```bash
    pipx install nexus-dev
    ```

=== "uv"

    ```bash
    uv tool install nexus-dev
    ```

=== "pip"

    ```bash
    pip install nexus-dev
    ```

Verify installation:

```bash
nexus-init --help
```

---

## Step 2: Set Your API Key

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

    !!! tip "Persist in Shell Profile"
        Add to `~/.zshrc` or `~/.bashrc` for persistence.

=== "Ollama (Local)"

    No API key needed. Ensure Ollama is running:

    ```bash
    ollama serve
    ollama pull nomic-embed-text
    ```

---

## Step 3: Initialize Your Project

Navigate to your project and run:

```bash
cd your-project
nexus-init --project-name "my-project" --embedding-provider openai
```

**Expected output:**

```
✅ Created nexus_config.json
✅ Created .nexus/lessons/
✅ Created database directory at ~/.local/share/nexus-dev/lancedb

Project ID: 550e8400-e29b-41d4-a716-446655440000
```

This creates:

| File/Directory | Purpose |
|---------------|---------|
| `nexus_config.json` | Project configuration |
| `.nexus/lessons/` | Recorded lessons (problem/solution pairs) |
| `~/.local/share/nexus-dev/lancedb/` | Vector database |

---

## Step 4: Index Your Codebase

```bash
# Index source code recursively
nexus-index src/ -r

# Index documentation
nexus-index docs/ README.md -r
```

**Expected output:**

```
  Found 42 files to index:

  📁 src/                               28 files
  📁 docs/                              14 files

Proceed with indexing? [y/N]: y
  ✅ main.py: 12 chunks
  ✅ utils.py: 8 chunks
  ...
✅ Indexed 156 chunks from 42 files
```

---

## Step 5: Configure Your IDE

Add Nexus-Dev to your MCP client configuration:

=== "Cursor"

    Edit `~/.cursor/mcp.json`:

    ```json
    {
      "mcpServers": {
        "nexus-dev": {
          "command": "nexus-dev",
          "args": [],
          "env": {
            "NEXUS_PROJECT_ROOT": "/path/to/your/project"
          }
        }
      }
    }
    ```

=== "Claude Desktop"

    Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

    ```json
    {
      "mcpServers": {
        "nexus-dev": {
          "command": "nexus-dev",
          "args": [],
          "env": {
            "NEXUS_PROJECT_ROOT": "/path/to/your/project"
          }
        }
      }
    }
    ```

=== "VS Code + Copilot"

    Add to your workspace settings or MCP configuration.

!!! warning "Important: Project Root"
    Set `NEXUS_PROJECT_ROOT` to your project directory. Without this, the server starts empty and you must run `refresh_agents` to load context.

---

## Step 6: Verify Setup

### Check Status via CLI

```bash
nexus-status
```

**Expected output:**

```
📊 Nexus-Dev Status

Project: my-project
Project ID: 550e8400-e29b-41d4-a716-446655440000
Embedding Provider: openai
Embedding Model: text-embedding-3-small
Database: /Users/you/.local/share/nexus-dev/lancedb

📈 Statistics:
   Total chunks: 156
   Code: 98
   Documentation: 58
   Lessons: 0
```

### Test in Your AI Agent

Paste this prompt in your IDE:

```
Search the Nexus-Dev knowledge base for functions related to "embeddings" 
and show me the project statistics.
```

If the AI uses `search_code` or `get_project_context` and returns results, **your setup is complete!** 🎉

---

## What's Next?

<div class="grid cards" markdown>

-   :material-book-open: **[Record Your First Lesson](tools/learning.md)**

    ---

    Learn how to capture problem/solution pairs for future reference.

-   :material-source-branch: **[Import GitHub Issues](cli/import-github.md)**
  
    ---

    Enrich your knowledge base with project history.

-   :material-robot: **[Create Custom Agents](workflows/new-project.md#optional-add-custom-agents)**

    ---

    Build specialized AI assistants for your project.

</div>
