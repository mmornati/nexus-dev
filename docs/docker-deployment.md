# Docker Deployment Guide

Run Nexus-Dev MCP server in Docker for improved stability and isolation.

## Quick Start

```bash
# Build and run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f nexus-dev
```

The server will be available at `http://localhost:8080/mcp`.

## IDE Configuration

Update your MCP client configuration to use SSE transport:

```json
{
  "servers": {
    "nexus-dev": {
      "transport": "sse",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

## Manual Docker Run

```bash
# Build the image
docker build -t nexus-dev .

# Run with SSE transport (default)
docker run -d -p 8080:8080 \
  -v $(pwd):/workspace:ro \
  -v nexus-data:/data/nexus-dev \
  --name nexus-dev \
  nexus-dev

# Run with stdio transport (for piping)
docker run -i \
  -v $(pwd):/workspace:ro \
  nexus-dev --transport stdio
```

## Volumes

| Volume | Purpose |
|--------|---------|
| `/workspace` | Project files (read-only) |
| `/data/nexus-dev` | Persistent database storage |
| `/workspace/.nexus` | MCP configuration |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_DB_PATH` | `/data/nexus-dev/db` | Database location |
| `NEXUS_LOG_LEVEL` | `INFO` | Logging level |

## Troubleshooting

**Connection refused**: Ensure the container is running and port 8080 is exposed.

```bash
docker ps | grep nexus-dev
curl http://localhost:8080/health
```

**No MCP config found**: Mount your `.nexus` directory.

```bash
docker run -v $(pwd)/.nexus:/workspace/.nexus:ro ...
```
