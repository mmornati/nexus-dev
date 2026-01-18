# New Project Setup

A complete walkthrough for setting up Nexus-Dev in a brand new project.

---

## Overview

This guide covers:

1. Installing Nexus-Dev
2. Initializing your project
3. Indexing your codebase
4. Configuring your IDE
5. Verifying the setup
6. (Optional) Adding custom agents

**Time required:** ~10 minutes

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

Verify installation:

```bash
nexus-init --version
```

---

## Step 2: Set Up API Key

=== "OpenAI (Recommended for Beginners)"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
    
    Add to `~/.zshrc` or `~/.bashrc` for persistence.

=== "Ollama (Local/Private)"

    ```bash
    # Start Ollama
    ollama serve
    
    # Pull embedding model
    ollama pull nomic-embed-text
    ```

---

## Step 3: Initialize Project

Navigate to your project root:

```bash
cd /path/to/your-project
nexus-init --project-name "My Project" --embedding-provider openai
```

**Expected output:**

```
✅ Created nexus_config.json
✅ Created .nexus/lessons/
✅ Created database directory at ~/.local/share/nexus-dev/lancedb

Configure .gitignore for .nexus folder? [allow-lessons/ignore-all/skip]: allow-lessons
✅ Updated .gitignore (allow-lessons)

Project ID: 550e8400-e29b-41d4-a716-446655440000
```

!!! tip "Choosing `.gitignore` Option"
    - **allow-lessons**: Track lessons in git (recommended for team learning)
    - **ignore-all**: Keep everything local
    - **skip**: Don't modify `.gitignore`

---

## Step 4: Index Your Codebase

### Index Source Code

```bash
nexus-index src/ -r
```

### Index Documentation

```bash
nexus-index docs/ README.md -r
```

### Index Everything at Once

```bash
nexus-index src/ docs/ tests/ README.md -r
```

**Expected output:**

```
  Found 85 files to index:

  📁 src/                               42 files
  📁 docs/                              23 files
  📁 tests/                             19 files
  📁 Root                                1 files

Proceed with indexing? [y/N]: y
  ✅ main.py: 12 chunks
  ✅ config.py: 8 chunks
  ...

✅ Indexed 312 chunks from 85 files
```

---

## Step 5: Configure Your IDE

Add Nexus-Dev to your MCP client:

=== "Cursor"

    Edit `~/.cursor/mcp.json`:
    
    ```json
    {
      "mcpServers": {
        "nexus-dev": {
          "command": "nexus-dev",
          "args": [],
          "env": {
            "NEXUS_PROJECT_ROOT": "/path/to/your-project",
            "OPENAI_API_KEY": "sk-..."
          }
        }
      }
    }
    ```

=== "Claude Desktop"

    Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
    
    ```json
    {
      "mcpServers": {
        "nexus-dev": {
          "command": "nexus-dev",
          "args": [],
          "env": {
            "NEXUS_PROJECT_ROOT": "/path/to/your-project"
          }
        }
      }
    }
    ```

Restart your IDE after configuration.

---

## Step 6: Verify Setup

### Via CLI

```bash
nexus-status
```

Should show your project stats.

### Via AI Agent

Paste this in your IDE:

```
Search the knowledge base for "main function" and show project statistics.
```

If the AI uses Nexus-Dev tools and returns results, **you're done!** 🎉

---

## Step 7: Add AGENTS.md (Recommended)

Help your AI assistant use Nexus-Dev effectively by adding an `AGENTS.md` to your project:

```bash
# Copy the template
curl -o AGENTS.md https://raw.githubusercontent.com/mmornati/nexus-dev/main/docs/AGENTS_TEMPLATE.md
```

Or create manually:

```markdown
# AI Agent Instructions

## Nexus-Dev Knowledge Base

This project uses Nexus-Dev for persistent memory.

**MANDATORY**: Use these tools BEFORE writing code:

1. `search_knowledge("<query>")` - Find relevant code/docs
2. `search_lessons("<error>")` - Check for known solutions
3. `get_project_context()` - Get project overview

**After debugging**: Record lessons with `record_lesson()`

**Start each session** by calling `get_project_context()`
```

---

## Optional: Add Custom Agents

Create specialized AI personas for your project:

```bash
# List available templates
nexus-agent templates

# Create from template
nexus-agent init project_expert --from-template code_reviewer
```

Edit `agents/project_expert.yaml` to customize.

Then use in your IDE:

```
Use the ask_project_expert tool to review the authentication module.
```

---

## Optional: Install Pre-commit Hook

Automatically index modified files on commit:

```bash
nexus-init --project-name "My Project" --install-hook
```

Or add manually to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
MODIFIED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|js|ts)$')
if [ -n "$MODIFIED" ]; then
    nexus-index $MODIFIED -q
fi
```

---

## Next Steps

- [Daily Usage Patterns](daily-usage.md) - Best practices for everyday use
- [Record Your First Lesson](../tools/learning.md) - Capture knowledge
- [Import GitHub Issues](../cli/import-github.md) - Add project history
