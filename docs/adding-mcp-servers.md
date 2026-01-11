# Adding Custom MCP Servers

This guide explains how to configure additional MCP servers for use with Nexus-Dev's gateway mode.

## Quick Start

1. Initialize MCP configuration:
   ```bash
   nexus-mcp init
   ```

2. Add a server:
   ```bash
   nexus-mcp add my-server --command "my-mcp-command" --args "--port" --args "8080"
   ```

3. Index the server's tools:
   ```bash
   nexus-index-mcp --server my-server
   ```

## Configuration Format

The `.nexus/mcp_config.json` file:

```json
{
  "version": "1.0",
  "servers": {
    "my-server": {
      "enabled": true,
      "command": "my-mcp-command",
      "args": ["--port", "8080"],
      "env": {
        "API_KEY": "${MY_API_KEY}"
      }
    }
  },
  "profiles": {
    "default": ["my-server"]
  },
  "active_profile": "default"
}
```

## Environment Variables

Use `${VAR_NAME}` syntax for environment variable expansion:

```json
{
  "env": {
    "GITHUB_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}",
    "API_KEY": "${MY_API_KEY}"
  }
}
```

## Common MCP Servers

### GitHub
```bash
nexus-mcp add github \
  --command "npx" \
  --args "-y" \
  --args "@modelcontextprotocol/server-github" \
  --env "GITHUB_PERSONAL_ACCESS_TOKEN=\${GITHUB_TOKEN}"
```

### PostgreSQL
```bash
nexus-mcp add postgres \
  --command "npx" \
  --args "-y" \
  --args "@modelcontextprotocol/server-postgres" \
  --args "postgresql://user:pass@localhost:5432/db"
```

### Filesystem (for secure access to specific folders)
```bash
nexus-mcp add filesystem \
  --command "npx" \
  --args "-y" \
  --args "@modelcontextprotocol/server-filesystem" \
  --args "/path/to/allowed/folder"
```

## Troubleshooting

### Server Not Starting
Check the logs using:
```bash
nexus-mcp logs my-server
```

### Tools Not Showing Up
Re-index the server:
```bash
nexus-index-mcp --server my-server --force
```
