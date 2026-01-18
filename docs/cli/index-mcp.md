# nexus-index-mcp

Index MCP tool documentation into the knowledge base.

---

## Synopsis

```bash
nexus-index-mcp [OPTIONS]
```

---

## Description

Connects to configured MCP servers, retrieves their tool schemas, and indexes them for semantic search via the `search_tools` command.

---

## Options

| Option | Type | Description |
|--------|------|-------------|
| `-s, --server` | TEXT | Index a specific server |
| `-a, --all` | FLAG | Index all configured servers |
| `-c, --config` | PATH | Custom MCP config file |

---

## Examples

### Index all servers

```bash
nexus-index-mcp --all
```

**Output:**

```
Indexing MCP tools...

📡 Connecting to github...
  ✅ Indexed 15 tools
📡 Connecting to homeassistant...
  ✅ Indexed 23 tools
📡 Connecting to filesystem...
  ✅ Indexed 8 tools

✅ Indexed 46 tools from 3 servers
```

### Index specific server

```bash
nexus-index-mcp --server github
```

### Use custom config

```bash
nexus-index-mcp --config ~/my-mcp-config.json --all
```

---

## Indexed Content

For each tool, the following is indexed:

| Field | Description |
|-------|-------------|
| Name | Tool name (e.g., `create_issue`) |
| Server | Source server (e.g., `github`) |
| Description | Tool description |
| Parameters | JSON schema of input parameters |

---

## Using Indexed Tools

After indexing, use `search_tools` to find tools:

```bash
# Via CLI search
nexus-search "create github issue" --type tool
```

Or via MCP:

```
search_tools("create GitHub issue")
```

**Returns:**

```
## MCP Tools matching: 'create GitHub issue'

### 1. github.create_issue
**Description:** Create a new issue in a repository

**Parameters:**
{
  "owner": "string",
  "repo": "string",
  "title": "string",
  "body": "string"
}
```

---

## See Also

- [nexus-mcp](mcp.md) - Configure MCP servers
- [Gateway Tools](../tools/gateway.md) - search_tools and invoke_tool
