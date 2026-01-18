# CLI Reference

Nexus-Dev provides a comprehensive set of command-line tools for managing your knowledge base.

---

## Command Overview

| Command | Description |
|---------|-------------|
| [`nexus-init`](init.md) | Initialize Nexus-Dev in a project |
| [`nexus-index`](index-cmd.md) | Index files or directories |
| [`nexus-reindex`](reindex.md) | Clear and rebuild the entire index |
| [`nexus-status`](status.md) | Show project statistics |
| [`nexus-search`](search.md) | Search the knowledge base |
| [`nexus-export`](export.md) | Export knowledge to markdown files |
| [`nexus-import-github`](import-github.md) | Import GitHub issues and PRs |
| [`nexus-mcp`](mcp.md) | MCP server configuration |
| [`nexus-agent`](agent.md) | Custom agent management |
| [`nexus-index-mcp`](index-mcp.md) | Index MCP tool schemas |

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
