# Configuration

Nexus-Dev is configured through `nexus_config.json` in your project root, supplemented by environment variables.

---

## Project Configuration

### nexus_config.json

Created by `nexus-init`, this file controls project-specific settings:

```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440000",
  "project_name": "my-project",
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "docs_folders": ["docs/", "README.md"],
  "include_patterns": ["**/*.py", "**/*.js", "**/*.ts", "**/*.java"],
  "exclude_patterns": ["**/node_modules/**", "**/__pycache__/**", "**/.venv/**"]
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `project_id` | string | auto-generated | Unique project identifier (UUID) |
| `project_name` | string | required | Human-readable project name |
| `embedding_provider` | string | `"openai"` | Provider for embeddings |
| `embedding_model` | string | varies | Model for embeddings |
| `docs_folders` | array | `["docs/", "README.md"]` | Paths to documentation |
| `include_patterns` | array | `["**/*.py", ...]` | Glob patterns for code files |
| `exclude_patterns` | array | `["**/node_modules/**", ...]` | Glob patterns to exclude |

---

## Embedding Providers

!!! warning "Provider Lock-in"
    Embeddings are NOT portable between providers. Changing providers requires re-indexing all documents with `nexus-reindex`.

### OpenAI (Default)

**Best for:** General purpose, ease of use.

```json
{
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small"
}
```

**Environment:**
```bash
export OPENAI_API_KEY="sk-..."
```

Available models:

| Model | Dimensions | Cost |
|-------|------------|------|
| `text-embedding-3-small` | 1536 | Low |
| `text-embedding-3-large` | 3072 | Higher |

---

### Ollama (Local/Privacy)

**Best for:** Privacy, local execution, cost savings.

```json
{
  "embedding_provider": "ollama",
  "embedding_model": "nomic-embed-text",
  "ollama_url": "http://localhost:11434"
}
```

**Setup:**
```bash
# Start Ollama
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
```

!!! tip "No API Key Needed"
    Ollama runs entirely locally. Your code never leaves your machine.

---

### Google Vertex AI (Enterprise)

**Best for:** Enterprise GCP users, high scalability.

**Install:**
```bash
pipx install nexus-dev[google]
```

```json
{
  "embedding_provider": "google",
  "embedding_model": "text-embedding-004",
  "google_project_id": "your-project-id",
  "google_location": "us-central1"
}
```

Uses standard Google Cloud Application Default Credentials (ADC).

---

### AWS Bedrock (Enterprise)

**Best for:** Enterprise AWS users.

**Install:**
```bash
pipx install nexus-dev[aws]
```

```json
{
  "embedding_provider": "aws",
  "embedding_model": "amazon.titan-embed-text-v1",
  "aws_region": "us-east-1"
}
```

Uses standard AWS credentials (`~/.aws/credentials` or environment variables).

---

### Voyage AI (High Performance)

**Best for:** State-of-the-art retrieval quality (RAG specialist).

**Install:**
```bash
pipx install nexus-dev[voyage]
```

```json
{
  "embedding_provider": "voyage",
  "embedding_model": "voyage-large-2",
  "voyage_api_key": "your-key"
}
```

---

### Cohere (Multilingual)

**Best for:** Multilingual search and reranking.

**Install:**
```bash
pipx install nexus-dev[cohere]
```

```json
{
  "embedding_provider": "cohere",
  "embedding_model": "embed-multilingual-v3.0",
  "cohere_api_key": "your-key"
}
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings |
| `NEXUS_PROJECT_ROOT` | Override project root detection |
| `NEXUS_DB_PATH` | Custom database location |

---

## MCP Configuration

For gateway mode, configure downstream MCP servers in `.nexus/mcp_config.json`:

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
    },
    "homeassistant": {
      "transport": "sse",
      "url": "http://homeassistant.local:8123/mcp",
      "headers": {
        "Authorization": "Bearer ..."
      },
      "enabled": true
    }
  }
}
```

See [MCP Gateway](../tools/gateway.md) for usage details.

---

## Docker Deployment

### Basic Configuration

```json
{
  "mcpServers": {
    "nexus-dev": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/path/to/project:/workspace:ro",
        "-v", "nexus-dev-data:/data/nexus-dev",
        "-e", "OPENAI_API_KEY",
        "ghcr.io/mmornati/nexus-dev:latest"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Volume Mounts

| Mount | Purpose |
|-------|---------|
| `/workspace` | Your project root (read-only) |
| `/data/nexus-dev` | Persistent database storage |

---

## Project Context & Startup

Nexus-Dev needs to know which project to load on startup:

1. **Automatic Detection**: If started with your project as the current working directory, it loads `nexus_config.json` automatically.

2. **Environment Variable**: Set `NEXUS_PROJECT_ROOT=/path/to/project` explicitly.

!!! tip "Pro Tip"
    Configure your MCP client to set `cwd` or `NEXUS_PROJECT_ROOT` to match your project. This avoids the need for manual `refresh_agents` calls.

---

## Next Steps

- [CLI Reference](../cli/index.md) - Learn all available commands
- [Workflows](../workflows/new-project.md) - Step-by-step usage guides
