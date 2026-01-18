# MCP Tools Reference

Nexus-Dev exposes tools to AI coding assistants via the Model Context Protocol (MCP).

---

## Tool Categories

<div class="grid cards" markdown>

-   :material-magnify: **[Search Tools](search.md)**

    ---

    Find code, documentation, and lessons across your knowledge base.

-   :material-database-plus: **[Indexing Tools](indexing.md)**

    ---

    Add files and content to the knowledge base.

-   :material-school: **[Learning Tools](learning.md)**

    ---

    Record lessons, insights, and implementations.

-   :material-connection: **[Gateway Tools](gateway.md)**

    ---

    Access other MCP servers through Nexus-Dev.

-   :material-robot: **[Agent Tools](agents.md)**

    ---

    Invoke custom AI agents.

</div>

---

## Quick Reference

### Search

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Search all content types |
| `search_code` | Search only code |
| `search_docs` | Search only documentation |
| `search_lessons` | Search recorded lessons |
| `search_insights` | Search recorded insights |
| `search_implementations` | Search implementation records |

### Index & Learn

| Tool | Purpose |
|------|---------|
| `index_file` | Index a file into the knowledge base |
| `record_lesson` | Record a problem/solution pair |
| `record_insight` | Record a discovery or mistake |
| `record_implementation` | Record a completed implementation |
| `get_project_context` | Get project statistics |

### Gateway

| Tool | Purpose |
|------|---------|
| `search_tools` | Find tools across MCP servers |
| `get_tool_schema` | Get detailed tool parameters |
| `invoke_tool` | Execute a tool on a backend server |
| `list_servers` | List configured MCP servers |

### Agents

| Tool | Purpose |
|------|---------|
| `refresh_agents` | Reload custom agents from disk |
| `list_agents` | Show available agents |
| `ask_*` | Dynamic tools for each custom agent |

---

## Using Tools

Your AI assistant calls these tools automatically based on context. You can also request specific tool usage:

```
Search the Nexus-Dev knowledge base for functions related to "authentication"
```

Or be explicit:

```
Use the search_code tool to find the User class definition
```

---

## Tool Discovery

AI assistants discover available tools through MCP's tool listing protocol. When you configure Nexus-Dev as an MCP server, all tools become available to your IDE.
