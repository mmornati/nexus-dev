# CLI Reference

Nexus-Dev provides a comprehensive set of command-line tools for managing your knowledge base.

---

## Available Commands

| Command | Description |
|---------|-------------|
| [nexus-init](init.md) | Initialize a new Nexus-Dev project |
| [nexus-index](index-cmd.md) | Index files into the knowledge base |
| [nexus-status](status.md) | Show project status and statistics |
| [nexus-inspect](inspect.md) | Inspect database contents for debugging |
| [nexus-clean](clean.md) | Delete indexed data for a project |
| [nexus-search](search.md) | Search the knowledge base |
| [nexus-export](export.md) | Export knowledge base to markdown |
| [nexus-reindex](reindex.md) | Re-index entire project |
| [nexus-import-github](import-github.md) | Import GitHub issues/PRs |
| [nexus-mcp](mcp.md) | Start MCP server |
| [nexus-agent](agent.md) | Run agentic tasks |
| [nexus-index-mcp](index-mcp.md) | Index MCP server definitions |

---

## Quick Reference

### Initialize a new project

```bash
nexus-init --project-name "my-project" --embedding-provider openai
```

### Index code and documentation

```bash
nexus-index src/ docs/ -r
```

### Check project status

```bash
nexus-status
```

### Search the knowledge base

```bash
nexus-search "authentication function"
```

---

## Global Options

All commands support these standard options:

| Option | Description |
|--------|-------------|
| `--help` | Show command help and exit |
| `--version` | Show version and exit |

---

## Environment Variables

Commands respect these environment variables:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for OpenAI embeddings |
| `NEXUS_PROJECT_ROOT` | Override project root detection |
| `NEXUS_DB_PATH` | Custom database location |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (missing config, invalid arguments, etc.) |
