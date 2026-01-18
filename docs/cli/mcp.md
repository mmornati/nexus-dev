# nexus-mcp

Manage MCP server configuration for gateway mode.

---

## Synopsis

```bash
nexus-mcp COMMAND [OPTIONS]
```

---

## Description

Configure downstream MCP servers that Nexus-Dev can proxy to via gateway mode. This allows your AI agent to access multiple MCP servers through a single connection.

---

## Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize MCP configuration |
| `add` | Add a new MCP server |
| `remove` | Remove an MCP server |
| `list` | List configured servers |
| `enable` | Enable a server |
| `disable` | Disable a server |

---

## nexus-mcp init

Create initial MCP configuration.

```bash
# Create empty configuration
nexus-mcp init

# Import from global Claude/Cursor config
nexus-mcp init --from-global
```

**Output:**

```
✅ Created .nexus/mcp_config.json
```

---

## nexus-mcp add

Add a new downstream MCP server.

### Stdio Transport (Local Process)

```bash
nexus-mcp add github \
  --command "npx" \
  --args "-y" \
  --args "@modelcontextprotocol/server-github"
```

### SSE Transport (Remote Server)

```bash
nexus-mcp add homeassistant \
  --transport sse \
  --url "http://homeassistant.local:8123/mcp"
```

---

## nexus-mcp list

Show all configured servers.

```bash
nexus-mcp list
```

**Output:**

```
MCP Servers:

✅ github (stdio)
   Command: npx -y @modelcontextprotocol/server-github

✅ homeassistant (sse)  
   URL: http://homeassistant.local:8123/mcp

❌ slack (disabled)
   Command: npx -y @modelcontextprotocol/server-slack
```

---

## nexus-mcp enable/disable

Toggle server availability.

```bash
nexus-mcp disable slack
nexus-mcp enable slack
```

---

## Configuration File

Settings are stored in `.nexus/mcp_config.json`:

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

---

## See Also

- [Gateway Tools](../tools/gateway.md) - Using gateway mode
- [nexus-index-mcp](index-mcp.md) - Index tool schemas for search
