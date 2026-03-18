# Gateway Configuration

Complete reference for configuring Nexus-Dev as an MCP gateway.

---

## Overview

The MCP Gateway mode allows Nexus-Dev to act as a unified interface to multiple MCP servers, reducing tool clutter and enabling intelligent tool routing.

```mermaid
flowchart TB
    IDE["🤖 AI Agent"]
    NX["📡 Nexus-Dev Gateway"]
    GH["GitHub"]
    HA["Home Assistant"]
    FS["Filesystem"]
    DB["Database"]
    
    IDE -->|"search_tools()"| NX
    IDE -->|"invoke_tool()"| NX
    NX --> GH
    NX --> HA
    NX --> FS
    NX --> DB
```

---

## Configuration File

Create `.nexus/mcp_config.json` in your project root:

```json
{
  "version": "1.0",
  "gateway": {
    "default_timeout": 30.0,
    "max_concurrent_connections": 5,
    "shutdown_timeout": 5.0,
    "cache": {
      "enabled": true,
      "ttl_seconds": 300,
      "max_entries": 1000
    },
    "summarize": {
      "enabled": true,
      "max_list_items": 10,
      "max_output_chars": 500
    }
  },
  "servers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      },
      "enabled": true,
      "timeout": 30.0,
      "connect_timeout": 10.0
    }
  }
}
```

---

## Gateway Settings

### Top-Level Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `version` | string | required | Config format version (currently "1.0") |
| `gateway` | object | required | Global gateway configuration |
| `servers` | object | required | Server definitions |
| `profiles` | object | `{}` | Named server profiles |
| `active_profile` | string | `"default"` | Active server profile |

### Gateway Object

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_timeout` | float | 30.0 | Tool execution timeout (seconds) |
| `max_concurrent_connections` | int | 5 | Maximum parallel tool invocations |
| `shutdown_timeout` | float | 5.0 | Graceful shutdown timeout (seconds) |
| `cache` | object | (see below) | Global caching settings |
| `summarize` | object | (see below) | Global output summarization settings |

### Cache Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable result caching |
| `ttl_seconds` | float | 300 | Time-to-live for cached results |
| `max_entries` | int | 1000 | Maximum cached entries |

### Summarize Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable output summarization |
| `max_list_items` | int | 10 | Maximum items to show in lists |
| `max_output_chars` | int | 500 | Maximum characters in output |

---

## Server Configuration

### Server Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transport` | string | "stdio" | Connection type: `stdio`, `sse`, or `http` |
| `command` | string | null | Command to run (for stdio) |
| `args` | array | [] | Command arguments (for stdio) |
| `env` | object | {} | Environment variables |
| `url` | string | null | HTTP endpoint (for sse/http) |
| `headers` | object | {} | HTTP headers |
| `enabled` | bool | true | Enable/disable server |
| `timeout` | float | 30.0 | Tool execution timeout (overrides gateway) |
| `connect_timeout` | float | 10.0 | Connection timeout |
| `max_concurrent` | int | null | Per-server concurrent limit |
| `cache` | object | null | Per-server cache override |
| `summarize` | object | null | Per-server summarize override |

### Per-Server Overrides

Override gateway settings for specific servers:

```json
{
  "servers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "cache": {
        "enabled": true,
        "ttl_seconds": 600
      },
      "summarize": {
        "enabled": true,
        "max_output_chars": 1000
      }
    },
    "slow-server": {
      "timeout": 60.0,
      "max_concurrent": 1
    }
  }
}
```

---

## Server Profiles

Define named profiles to switch between server configurations:

```json
{
  "profiles": {
    "default": ["github", "filesystem"],
    "full": ["github", "filesystem", "postgres", "slack"],
    "minimal": ["github"]
  },
  "active_profile": "default"
}
```

Switch profiles using the CLI:

```bash
nexus-mcp profile set full
```

---

## Transport Types

### Stdio (Local Process)

Run MCP servers as local subprocesses:

```json
{
  "github": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
    }
  }
}
```

### SSE (Server-Sent Events)

Connect to HTTP servers with SSE:

```json
{
  "homeassistant": {
    "transport": "sse",
    "url": "http://homeassistant.local:8123/mcp",
    "headers": {
      "Authorization": "Bearer ${HA_TOKEN}"
    }
  }
}
```

### HTTP (Request-Response)

Connect to standard HTTP endpoints:

```json
{
  "custom-api": {
    "transport": "http",
    "url": "https://api.example.com/mcp",
    "headers": {
      "Authorization": "Bearer ${API_TOKEN}"
    }
  }
}
```

---

## Environment Variables

Reference environment variables in configuration:

```json
{
  "env": {
    "GITHUB_TOKEN": "${GITHUB_TOKEN}",
    "DATABASE_URL": "postgresql://user:pass@localhost/db"
  }
}
```

Set in your shell:

```bash
export GITHUB_TOKEN="ghp_..."
```

---

## Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Via CLI flag
nexus-dev --debug

# Logs written to /tmp/nexus-dev-debug.log
```

---

## Metrics & Monitoring

Use the `get_gateway_metrics` tool to monitor usage:

```
get_gateway_metrics()
```

Returns:

```markdown
Gateway Usage (last 24h):
- search_tools calls: 42
- invoke_tool calls: 156
- Cache hits: 89 (56.3%)
- Cache misses: 67

Tools by server:
- github: 98 calls
- filesystem: 35 calls
- homeassistant: 23 calls
```

---

## Common Configurations

### GitHub Server

```json
{
  "github": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
    },
    "enabled": true
  }
}
```

### Filesystem Server

```json
{
  "filesystem": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"],
    "enabled": true
  }
}
```

### PostgreSQL Server

```json
{
  "postgres": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-postgres"],
    "env": {
      "DATABASE_URL": "${DATABASE_URL}"
    },
    "enabled": true
  }
}
```

### Complete Example

```json
{
  "version": "1.0",
  "gateway": {
    "default_timeout": 30.0,
    "max_concurrent_connections": 5,
    "cache": {
      "enabled": true,
      "ttl_seconds": 300,
      "max_entries": 1000
    },
    "summarize": {
      "enabled": true,
      "max_list_items": 10,
      "max_output_chars": 500
    }
  },
  "profiles": {
    "default": ["github", "filesystem"],
    "development": ["github", "filesystem", "postgres"]
  },
  "active_profile": "default",
  "servers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      },
      "enabled": true,
      "cache": {
        "enabled": true,
        "ttl_seconds": 600
      }
    },
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"],
      "enabled": true
    },
    "postgres": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "${DATABASE_URL}"
      },
      "enabled": false,
      "timeout": 60.0
    }
  }
}
```

---

## CLI Commands

### Initialize Gateway

```bash
# Create from global config
nexus-mcp init --from-global

# Or create empty
nexus-mcp init
```

### Add Servers

```bash
nexus-mcp add github \
  --command "npx" \
  --args "-y" "@modelcontextprotocol/server-github"
```

### Index Tool Schemas

```bash
# Index all servers
nexus-index-mcp --all

# Index specific server
nexus-index-mcp --server github
```

### Manage Profiles

```bash
# List profiles
nexus-mcp profile list

# Switch profile
nexus-mcp profile set development
```

---

## See Also

- [Gateway Tools](../tools/gateway.md) - Tool reference
- [MCP CLI](../cli/mcp.md) - CLI commands
- [Troubleshooting](../reference/troubleshooting.md) - Debug tips
