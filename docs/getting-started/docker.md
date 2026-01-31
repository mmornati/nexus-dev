# Docker Usage
# Docker Usage

This guide explains how to run Nexus-Dev using Docker. This approach is recommended if you want to avoid installing Python dependencies locally or want to run the Nexus-Dev server in a containerized environment.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, but recommended)

---

## Quick Start

The easiest way to use Nexus-Dev with Docker is via the `nexus-docker` wrapper script provided in the repository.

1. **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/mmornati/nexus-dev.git
    cd nexus-dev
    ```

2. **Build the image**:
    ```bash
    docker-compose build
    ```

3. **Run a command**:
    ```bash
    # Check status
    ./nexus-docker nexus-status

    # List MCP servers
    ./nexus-docker nexus-mcp list
    ```

---

## Running the MCP Server

You can run the Nexus-Dev MCP server in two modes appropriate for Docker: **SSE** (Server-Sent Events) and **Stdio**.

### SSE Mode (Recommended)

SSE mode runs an HTTP server that your IDE or MCP client connects to. This is the default in `docker-compose.yml`.

1. **Start the server**:
    ```bash
    docker-compose up -d
    ```

2. **Configure your IDE**:

    **Cursor / Claude Desktop**:
    ```json
    {
      "mcpServers": {
        "nexus-dev": {
          "url": "http://localhost:8080/sse",
          "transport": "sse"
        }
      }
    }
    ```

    Note: Ensure port `8080` is exposed and not blocked by firewalls.

### Stdio Mode

If your IDE requires running a command directly (and supports Docker commands), you can use `docker run -i`.

**Cursor / Claude Desktop**:
```json
{
  "mcpServers": {
    "nexus-dev": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v", "/path/to/project:/workspace",
        "-v", "nexus-data:/data/nexus-dev",
        "nexus-dev:latest",
        "nexus-dev",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

*Note: This method is more complex to configure due to volume mounting paths matching exactly what the IDE expects.*

---

## Managing Gateway Configuration

When running via Docker, your MCP Gateway configuration (`mcp_config.json`) is stored in `.nexus/` inside the container. To make this persistent and editable:

1. **Volume Mounting**:
   The provided `docker-compose.yml` mounts your local `.nexus` directory to `/workspace/.nexus`.
   ```yaml
   volumes:
     - ./.nexus:/workspace/.nexus
   ```

2. **Managing Servers**:
   Use the `nexus-docker` script to add servers.

   ```bash
   # Initialize config
   ./nexus-docker nexus-mcp init

   # Add a GitHub server
   ./nexus-docker nexus-mcp add github \
       --command "npx" \
       --args "-y @modelcontextprotocol/server-github" \
       --env "GITHUB_PERSONAL_ACCESS_TOKEN=..."
   ```

### Connecting to Host Services

If you need your Dockerized Nexus-Dev to connect to services running on your host machine (like **Ollama** or **Local Home Assistant**), use `host.docker.internal`.

The `docker-compose.yml` is pre-configured with:
```yaml
environment:
  - OLLAMA_HOST=http://host.docker.internal:11434
extra_hosts:
  - "host.docker.internal:host-gateway"
```

This ensures that the `nexus-dev` agent can reach your local Ollama instance even from inside the container.

---

## CLI Usage via Docker

You can run any `nexus-*` CLI command using the wrapper or `docker run`.

### Indexing Code
```bash
./nexus-docker nexus-index src/ -r
```

### Searching Knowledge
```bash
./nexus-docker nexus-search "how to configure auth"
```

### Managing Agents
```bash
./nexus-docker nexus-agent list
```

---

## Troubleshooting

### "Project root not found"
Ensure you are running the docker command from the root of your project, and that the volume mount `-v $(pwd):/workspace` is correct. The container expects the project root to be at `/workspace`.

### "Connection Refused" (SSE)
- Check if the container is running: `docker ps`
- Verify port mapping: `0.0.0.0:8080->8080/tcp`
- Check logs: `docker-compose logs -f nexus-dev`
