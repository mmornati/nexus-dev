# Nexus-Dev Quick Start

Get up and running with Nexus-Dev in 5 minutes.

## Prerequisites

- Python 3.13+
- An MCP-compatible IDE (Cursor, Claude Desktop, VS Code with Copilot)
- OpenAI API key (for embeddings) or Ollama running locally

## Installation

```bash
pip install nexus-dev
```

## 1. Initialize Your Project

```bash
cd your-project
nexus-init --project-name "my-project" --embedding-provider openai
```

This creates:
- `nexus_config.json` - Project configuration
- `.nexus/lessons/` - Directory for recorded lessons
- `.nexus/lancedb/` - Vector database

## 2. Index Your Codebase

```bash
# Index source code
nexus-index src/ -r

# Index documentation
nexus-index docs/ README.md -r
```

## 3. Configure MCP Servers (Optional)

If you have other MCP servers (GitHub, Home Assistant, etc.):

```bash
# Initialize MCP configuration
nexus-mcp init --from-global

# Or add servers manually
nexus-mcp add github --command "npx" --args "-y" --args "@modelcontextprotocol/server-github"

# Index MCP tool documentation
nexus-index-mcp --all
```

## 4. Create a Custom Agent

### Option A: Use a Pre-Built Template (Recommended)

List available templates:
```bash
nexus-agent templates
```

Available templates:
- **code_reviewer** - Reviews code for bugs, security issues, and best practices
- **doc_writer** - Creates and updates technical documentation
- **debug_detective** - Analyzes errors and proposes fixes
- **refactor_architect** - Suggests code restructuring and design patterns
- **test_engineer** - Generates test cases and improves coverage
- **security_auditor** - Identifies vulnerabilities and recommends fixes
- **api_designer** - Reviews and designs REST/GraphQL APIs
- **performance_optimizer** - Finds performance bottlenecks and suggests optimizations

Create from template:
```bash
# Create with default model from template
nexus-agent init my_reviewer --from-template code_reviewer

# Override template model
nexus-agent init my_security --from-template security_auditor --model claude-opus-4.5
```

### Option B: Create Custom Agent Interactively

```bash
nexus-agent init my_custom_agent
# Follow the interactive prompts for role, goal, and backstory
```

### Customize Your Agent

Edit the generated `agents/your_agent.yaml`:

```yaml
name: "my_reviewer"
display_name: "My Reviewer"
description: "Delegate code review tasks to the My Reviewer agent."

profile:
  role: "Senior Code Reviewer"
  goal: "Identify bugs, security issues, and suggest improvements"
  backstory: "Expert developer with 10+ years of experience in code quality."
  tone: "Professional and constructive"

memory:
  enabled: true
  rag_limit: 5
  search_types: ["code", "documentation", "lesson"]

tools: []  # Empty = all tools available

llm_config:
  model_hint: "claude-sonnet-4.5"  # Primary model preference
  fallback_hints: ["auto"]          # Fallback strategy
  temperature: 0.5
  max_tokens: 4000
```

## 5. Configure Your IDE

Add Nexus-Dev to your IDE's MCP configuration:

### Cursor

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

### Claude Desktop

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

> **CRITICAL**: If you do not set `NEXUS_PROJECT_ROOT` (or ensure `cwd` is correct), the server will start empty. In that case, you **MUST** run the `refresh_agents` tool immediately after connecting to load your project's configuration and agents.

## 6. Use Your Agent

In your IDE, the agent appears as an MCP tool:

```
Use the ask_code_reviewer tool to review this function for security issues.
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   IDE       │────▶│  Nexus-Dev   │────▶│  LanceDB    │
│(Cursor/etc.)│◀────│  MCP Server  │◀────│  (RAG)      │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
  MCP Sampling       Custom Agents
  (model calls)      (ask_* tools)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **RAG Search** | Semantic search across code, docs, and lessons |
| **Custom Agents** | Define specialized personas in YAML |
| **MCP Gateway** | Access all your MCP servers through one endpoint |
| **IDE Models** | Uses your IDE's configured models (no extra API keys) |
| **Zero Config** | Agents auto-discovered from `./agents/` |

## CLI Reference

| Command | Description |
|---------|-------------|
| `nexus-init` | Initialize Nexus-Dev in a project |
| `nexus-index` | Index files or directories |
| `nexus-reindex` | Clear and rebuild the index |
| `nexus-status` | Show project statistics |
| `nexus-mcp init` | Initialize MCP configuration |
| `nexus-mcp add` | Add an MCP server |
| `nexus-mcp list` | List configured servers |
| `nexus-agent init` | Create a new custom agent |
| `nexus-agent list` | List configured agents |
| `nexus-agent templates` | Show available agent templates |

## Next Steps

- [Record lessons](./recording-lessons.md) to teach your agents
- [Configure MCP servers](./mcp-configuration.md) for extended capabilities
- [Customize agents](./custom-agents.md) for your workflow
