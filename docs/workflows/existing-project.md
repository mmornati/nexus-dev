# Existing Project Onboarding

A guide for adding Nexus-Dev to an existing project with documentation and history.

---

## Overview

This guide covers:

1. Installing and initializing Nexus-Dev
2. Bulk indexing existing code and documentation
3. Importing GitHub issues and PRs
4. Creating lessons from known bugs
5. Setting up for ongoing use

**Time required:** ~15-30 minutes (depending on project size)

---

## Step 1: Install and Initialize

```bash
# Install globally
pipx install nexus-dev

# Navigate to your project
cd /path/to/existing-project

# Initialize
nexus-init --project-name "My Existing Project" --embedding-provider openai
```

!!! warning "Existing .nexus Directory"
    If `.nexus/` already exists, you'll be prompted to overwrite. Backup existing lessons first if needed.

---

## Step 2: Bulk Index Codebase

### Recommended Approach

Index your main code directories:

```bash
# Index source code
nexus-index src/ lib/ app/ -r

# Index documentation
nexus-index docs/ README.md CONTRIBUTING.md -r

# Index configuration (often contains important patterns)
nexus-index config/ -r
```

### Large Projects

For very large projects, index incrementally:

```bash
# Core modules first
nexus-index src/core/ src/models/ -r

# Then supporting code
nexus-index src/utils/ src/api/ -r

# Finally tests (optional but useful)
nexus-index tests/ -r
```

### Monorepo Strategy

For monorepos, index packages separately:

```bash
nexus-index packages/auth/src/ -r
nexus-index packages/api/src/ -r
nexus-index packages/web/src/ -r
```

---

## Step 3: Import GitHub Issues (Optional)

If your project has GitHub issues/PRs, import them for context:

### Configure GitHub MCP Server

Create `.nexus/mcp_config.json`:

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

### Import Issues

```bash
# Import recent issues
nexus-import-github --owner your-org --repo your-repo --limit 100

# Import all issues (may take a while)
nexus-import-github --owner your-org --repo your-repo --limit 500 --state all
```

---

## Step 4: Seed Knowledge Base

### Create Lessons from Known Issues

If your team has documented known issues, convert them to lessons:

```bash
# Create lessons directory if not exists
mkdir -p .nexus/lessons
```

Create lessons manually in `.nexus/lessons/`:

```markdown
---
problem: "Database connection pool exhaustion under heavy load"
solution: "Increased pool size from 10 to 50 and added connection timeout of 30s"
context: "src/database/connection.py"
timestamp: "2024-01-15T10:30:00Z"
---

## Problem
Database connection pool exhaustion under heavy load

## Solution
Increased pool size from 10 to 50 and added connection timeout of 30s

## Code Changes
```python
# Before
pool = ConnectionPool(max_size=10)

# After
pool = ConnectionPool(max_size=50, timeout=30)
```
```

Then index the lessons:

```bash
nexus-index .nexus/lessons/ -r
```

### Import from Wiki/Docs

If you have a wiki or internal docs with troubleshooting info:

1. Export relevant pages as Markdown
2. Place in `docs/troubleshooting/` or similar
3. Index: `nexus-index docs/troubleshooting/ -r`

---

## Step 5: Capture Team Knowledge

### Record Insights from Team Members

Ask senior developers about common pitfalls and record them:

```python
# Via MCP tool
record_insight(
    category="discovery",
    description="The legacy API requires base64 encoding for auth tokens",
    reasoning="Not documented anywhere, discovered through trial and error"
)
```

### Document Architecture Decisions

Record major implementation decisions:

```python
record_implementation(
    title="Microservices Migration",
    summary="Migrated from monolith to microservices architecture",
    approach="Strangler fig pattern with API gateway",
    design_decisions=[
        "Used event sourcing for inter-service communication",
        "Kept shared database during transition for rollback capability"
    ],
    files_changed=["services/", "gateway/", "docker-compose.yml"]
)
```

---

## Step 6: Configure IDE

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "nexus-dev": {
      "command": "nexus-dev",
      "args": [],
      "env": {
        "NEXUS_PROJECT_ROOT": "/path/to/existing-project"
      }
    }
  }
}
```

---

## Step 7: Verify and Test

### Check Status

```bash
nexus-status
```

**Expected output:**

```
📊 Nexus-Dev Status

Project: My Existing Project
Project ID: 550e8400-e29b-41d4-a716-446655440000
...

📈 Statistics:
   Total chunks: 1,247
   Code: 892
   Documentation: 312
   Lessons: 15
   GitHub Issues: 28
```

### Test Search

```bash
nexus-search "authentication"
nexus-search "database connection error" --type lesson
```

---

## Step 8: Set Up Ongoing Maintenance

### Add AGENTS.md

```bash
curl -o AGENTS.md https://raw.githubusercontent.com/mmornati/nexus-dev/main/docs/AGENTS_TEMPLATE.md
```

### Install Pre-commit Hook

```bash
# Reinitialize with hook
nexus-init --project-name "My Existing Project" --install-hook
```

### Schedule Regular Re-indexing

For large/active projects, consider periodic full re-indexing:

```bash
# Add to crontab or CI pipeline
nexus-reindex
```

---

## Onboarding Checklist

- [ ] Installed Nexus-Dev globally
- [ ] Initialized project with `nexus-init`
- [ ] Indexed source code
- [ ] Indexed documentation
- [ ] Imported GitHub issues (if applicable)
- [ ] Created lessons from known issues
- [ ] Recorded team knowledge/insights
- [ ] Configured IDE
- [ ] Added AGENTS.md
- [ ] Installed pre-commit hook (optional)

---

## Next Steps

- [Daily Usage Patterns](daily-usage.md) - Integrate into your workflow
- [Custom Agents](../tools/agents.md) - Create project-specific assistants
- [Gateway Mode](../tools/gateway.md) - Connect other MCP servers
