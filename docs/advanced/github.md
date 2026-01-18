# GitHub Integration

Advanced guide for integrating GitHub data into your knowledge base.

---

## Overview

Nexus-Dev can import GitHub issues and pull requests to enrich your knowledge base with:

- **Bug reports** and their resolutions
- **Feature discussions** and design decisions
- **PR descriptions** and code change context
- **Community knowledge** from comments

---

## Prerequisites

### 1. GitHub MCP Server

Configure the GitHub MCP server in `.nexus/mcp_config.json`:

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

### 2. Personal Access Token

Create a token at [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens):

**Required scopes:**
- `repo` - Full repository access (for private repos)
- `public_repo` - Only public repositories

!!! tip "Fine-grained Tokens"
    For better security, use fine-grained tokens with repository-specific access.

---

## Importing Issues

### Basic Import

```bash
nexus-import-github --owner mmornati --repo nexus-dev
```

Imports 20 most recent issues/PRs (default).

### Import Options

```bash
# Import more items
nexus-import-github --owner mmornati --repo nexus-dev --limit 100

# Only open issues
nexus-import-github --owner mmornati --repo nexus-dev --state open

# Only closed (resolved) issues
nexus-import-github --owner mmornati --repo nexus-dev --state closed
```

### Via MCP Tool

```
import_github_issues(
    repo="nexus-dev",
    owner="mmornati",
    limit=50,
    state="all"
)
```

---

## What Gets Indexed

For each issue/PR:

| Field | Indexed | Searchable |
|-------|---------|------------|
| Title | ✅ | ✅ High weight |
| Body | ✅ | ✅ |
| State (open/closed) | ✅ | ✅ Filter |
| Labels | ✅ | ✅ |
| Author | ✅ | ✅ |
| Created/Updated dates | ✅ | ✅ |
| Issue/PR number | ✅ | ✅ |

---

## Searching GitHub Data

### By Content

```bash
nexus-search "authentication bug"
```

### By Type

```bash
nexus-search "rate limiting" --type github_issue
```

### Via MCP

```
search_knowledge("database connection error", content_type="all")
```

Issues and PRs are returned alongside code and documentation results.

---

## Use Cases

### 1. Bug Context

When encountering an error, search for related issues:

```
Search lessons and GitHub issues for "ConnectionRefusedError"
```

May find past discussions with workarounds or fixes.

### 2. Feature Research

Before implementing a feature:

```
Search GitHub issues for "webhooks" to see prior discussions
```

Understand past decisions and rejected approaches.

### 3. Code Archaeology

Understanding why code exists:

```
Search GitHub PRs for changes to "src/auth/middleware.py"
```

Find the PR that introduced the code with context.

---

## Best Practices

### Initial Import

For established projects, import historical data:

```bash
# Import all closed issues (solutions exist)
nexus-import-github --owner your-org --repo your-repo --state closed --limit 500

# Import recent open issues (current context)
nexus-import-github --owner your-org --repo your-repo --state open --limit 50
```

### Ongoing Sync

Periodically refresh with recent issues:

```bash
# Weekly sync of recent activity
nexus-import-github --owner your-org --repo your-repo --limit 50
```

!!! note "Duplicate Handling"
    Re-importing updates existing entries rather than creating duplicates.

### Organization-Wide

For monorepos or multi-repo projects:

```bash
# Import from multiple repos
nexus-import-github --owner your-org --repo frontend --limit 100
nexus-import-github --owner your-org --repo backend --limit 100
nexus-import-github --owner your-org --repo shared-lib --limit 50
```

---

## Advanced: Create Lessons from Issues

Convert resolved issues into lessons for better RAG:

### Manual Approach

1. Find resolved bug issues
2. Create lesson files:

```markdown
---
problem: "API returns 500 when user has no profile"
solution: "Added null check in UserService.getProfile()"
context: "GitHub Issue #42"
---

## Problem
API returns 500 error when fetching profile for users who haven't completed onboarding.

## Solution
Added null check in UserService.getProfile() to return empty profile object.

## Code
```python
def get_profile(self, user_id: str) -> Profile:
    profile = self.db.get_profile(user_id)
    if profile is None:
        return Profile.empty()  # Added this check
    return profile
```
```

3. Index the lessons:

```bash
nexus-index .nexus/lessons/ -r
```

### Automation Idea

Create a script that:
1. Queries closed issues with "bug" label
2. Extracts problem/solution from issue body
3. Generates lesson markdown files
4. Indexes them

---

## Troubleshooting

### Import Fails: "Server not found"

```bash
# Verify GitHub server is configured
nexus-mcp list
```

### Import Fails: "401 Unauthorized"

Check your token:
- Token hasn't expired
- Token has correct scopes
- Token is properly set in config

### Empty Results

- Repository might be private (need `repo` scope)
- No issues match the state filter
- Rate limiting (wait and retry)

---

## See Also

- [nexus-import-github CLI](../cli/import-github.md) - Command reference
- [import_github_issues tool](../tools/indexing.md) - MCP tool
- [Gateway Mode](gateway.md) - Using GitHub through gateway
