# nexus-import-github

Import GitHub issues and pull requests into the knowledge base.

---

## Synopsis

```bash
nexus-import-github [OPTIONS]
```

---

## Description

Fetches issues and PRs from a GitHub repository using the GitHub MCP server and indexes them for semantic search. This enriches your knowledge base with project history and discussions.

---

## Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--repo` | TEXT | ✅ | - | Repository name |
| `--owner` | TEXT | ✅ | - | Repository owner |
| `--limit` | INT | | `20` | Maximum items to import |
| `--state` | TEXT | | `all` | Issue state (`open`, `closed`, `all`) |

---

## Prerequisites

1. **GitHub MCP Server**: Configure the GitHub MCP server in `.nexus/mcp_config.json`:

```json
{
  "servers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      },
      "enabled": true
    }
  }
}
```

2. **Personal Access Token**: Create a GitHub token with `repo` scope.

---

## Examples

### Import from current repository

```bash
nexus-import-github --owner mmornati --repo nexus-dev
```

**Output:**

```
📥 Importing issues from mmornati/nexus-dev...
  ✅ Issue #42: Add custom agent support
  ✅ Issue #38: Fix embedding dimension mismatch
  ✅ PR #35: Implement MCP gateway mode
  ...
✅ Imported 20 issues/PRs
```

### Import only open issues

```bash
nexus-import-github --owner myorg --repo myrepo --state open
```

### Import more items

```bash
nexus-import-github --owner myorg --repo myrepo --limit 50
```

---

## Indexed Content

Each imported item includes:

| Field | Description |
|-------|-------------|
| Title | Issue/PR title |
| Body | Description content |
| State | Open/closed status |
| Labels | Applied labels |
| Author | Creator username |
| Created/Updated dates | Timestamps |

---

## Searching Imported Issues

After import, search with type filters:

```bash
nexus-search "authentication bug" --type github_issue
```

Or use MCP tools:

```
search_knowledge("authentication bug", content_type="all")
```

---

## See Also

- [import_github_issues tool](../tools/indexing.md) - MCP equivalent
- [MCP Configuration](../getting-started/configuration.md#mcp-configuration) - Gateway setup
