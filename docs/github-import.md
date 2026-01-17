# GitHub Knowledge Import

Nexus-Dev can import **Issues** and **Pull Requests** from GitHub repositories into your knowledge base. This allows your AI agent to answer questions based on project history, bug reports, and decision-making discussions.

## Prerequisites

You must have the **GitHub MCP Server** configured in your `.nexus/mcp_config.json`.

### Local Setup (Stdio)
Requires `npx` (Node.js).

```json
{
  "servers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-pat-token"
      }
    }
  }
}
```

### Remote Setup (SSE)
If you are using a managed MCP provider.

```json
{
  "servers": {
    "github": {
      "transport": "sse",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "Bearer your-token"
      }
    }
  }
}
```

## Usage

### 1. CLI Command

You can manually trigger an import using the `nexus-import-github` command.

```bash
# Import everything (Open & Closed Issues + PRs)
nexus-import-github --owner <owner> --repo <repo>

# Example
nexus-import-github --owner mmornati --repo nexus-dev
```

**Options:**
- `--owner`: Repository owner (required).
- `--repo`: Repository name (required).
- `--limit`: Max items to import per page (default: 20).
- `--state`: Filter by state: `open`, `closed`, or `all` (default: `all`).

### 2. Using AI Agents

Once configured, your AI agent can use the `import_github_issues` tool directly.

**Example Prompts:**
> "Import the latest issues from the nexus-dev repository."
> "Check the knowledge base for any GitHub issues about 'connection timeout'."

### 3. Searching Imported Data

Imported items are indexed with specific types: `github_issue` and `github_pr`.

```bash
# Search specifically for issues
nexus-search "bug in login" --type github_issue

# Search broadly
nexus-search "feature request"
```

## How It Works

1. **Fetching**: The `GitHubImporter` queries the `github` MCP server for issues and pull requests.
2. **Processing**: It distinguishes between Issues and PRs.
3. **Indexing**: It creates `Document` objects with rich metadata (ID, title, state, author, body) and stores them in LanceDB.
4. **Retrieval**: These documents are embedded and available for semantic search alongside your code and documentation.
