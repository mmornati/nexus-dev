# MCP Gateway Mode

Advanced guide for using Nexus-Dev as a gateway to other MCP servers.

---

## Why Gateway Mode?

Without gateway mode:
- Configure 10+ MCP servers in your IDE
- 100+ tools in context window
- High token usage
- Complex per-IDE configuration

With gateway mode:
- Configure only Nexus-Dev
- ~20 tools + semantic search
- Lower token usage
- One-time setup

```mermaid
flowchart TB
    subgraph Without["❌ Without Gateway"]
        IDE1["IDE"]
        GH1["GitHub MCP"]
        HA1["Home Assistant"]
        FS1["Filesystem"]
        DB1["Database"]
        
        IDE1 --> GH1
        IDE1 --> HA1
        IDE1 --> FS1
        IDE1 --> DB1
    end
    
    subgraph With["✅ With Gateway"]
        IDE2["IDE"]
        NX["Nexus-Dev"]
        GH2["GitHub"]
        HA2["Home Assistant"]
        FS2["Filesystem"]
        DB2["Database"]
        
        IDE2 --> NX
        NX --> GH2
        NX --> HA2
        NX --> FS2
        NX --> DB2
    end
```

---

## Setup

### 1. Initialize Configuration

```bash
# Create from global config
nexus-mcp init --from-global

# Or create empty
nexus-mcp init
```

### 2. Add Servers

#### Stdio Transport (Local Process)

```bash
nexus-mcp add github \
  --command "npx" \
  --args "-y" "@modelcontextprotocol/server-github"
```

Or edit `.nexus/mcp_config.json`:

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

#### SSE Transport (Remote HTTP)

```json
{
  "servers": {
    "homeassistant": {
      "transport": "sse",
      "url": "http://homeassistant.local:8123/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      },
      "enabled": true
    }
  }
}
```

### 3. Index Tool Schemas

```bash
nexus-index-mcp --all
```

This enables semantic search over tool descriptions.

---

## Gateway Workflow

### Step 1: Find the Tool

```
search_tools("create a GitHub issue")
```

Returns matching tools with parameter schemas.

### Step 2: Get Full Schema (Optional)

```
get_tool_schema("github", "create_issue")
```

Returns complete JSON schema for complex tools.

### Step 3: Invoke

```
invoke_tool(
    server="github",
    tool="create_issue",
    arguments={
        "owner": "mmornati",
        "repo": "nexus-dev",
        "title": "Add feature X",
        "body": "Description of the feature"
    }
)
```

---

## Configuration Reference

### Server Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `transport` | string | ✅ | `stdio` or `sse` |
| `command` | string | for stdio | Command to run |
| `args` | array | | Command arguments |
| `env` | object | | Environment variables |
| `url` | string | for sse | HTTP endpoint |
| `headers` | object | | HTTP headers |
| `timeout` | int | | Connection timeout (ms) |
| `enabled` | bool | | Enable/disable server |

### Example Configurations

#### GitHub

```json
{
  "github": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
    }
  }
}
```

#### Filesystem

```json
{
  "filesystem": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
  }
}
```

#### PostgreSQL

```json
{
  "postgres": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-postgres"],
    "env": {
      "DATABASE_URL": "postgresql://user:pass@localhost/db"
    }
  }
}
```

---

## Performance Considerations

### Connection Pooling

Nexus-Dev maintains persistent connections to avoid startup overhead:

- Stdio processes are kept running
- SSE connections are reused

### Token Efficiency

| Approach | Tools in Context | Approx. Tokens |
|----------|------------------|----------------|
| Direct (10 servers) | ~100 tools | ~50,000 |
| Gateway | ~20 tools | ~10,000 |
| Savings | - | **~80%** |

### Cold Start

First tool invocation may be slower due to:

- Process startup (stdio)
- Connection establishment (sse)

Subsequent calls are faster.

---

## Security

### API Keys

Store sensitive values in environment:

```json
{
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
  }
}
```

Then set in your shell:

```bash
export GITHUB_TOKEN="ghp_..."
```

### Network Isolation

For SSE servers, ensure network security:

- Use HTTPS in production
- Limit exposed endpoints
- Use authentication headers

---

## Troubleshooting

### Server Not Found

```
Server not found: github
```

Check `.nexus/mcp_config.json` exists and server is configured.

### Connection Failed

```
Error connecting to homeassistant: Connection refused
```

- Verify URL is correct
- Check server is running
- Test with curl: `curl http://homeassistant.local:8123/mcp`

### Tool Not Found

```
Tool not found: github.nonexistent_tool
```

- Run `nexus-index-mcp --server github` to refresh
- Check tool name spelling

---

## See Also

- [nexus-mcp CLI](../cli/mcp.md) - Server management
- [Gateway Tools](../tools/gateway.md) - Tool reference
- [nexus-index-mcp](../cli/index-mcp.md) - Indexing tools
