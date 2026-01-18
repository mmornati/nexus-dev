# nexus-status

Show Nexus-Dev status and project statistics.

---

## Synopsis

```bash
nexus-status
```

---

## Description

Displays project configuration and knowledge base statistics including chunk counts by type.

---

## Example

```bash
nexus-status
```

**Output:**

```
📊 Nexus-Dev Status

Project: my-project
Project ID: 550e8400-e29b-41d4-a716-446655440000
Embedding Provider: openai
Embedding Model: text-embedding-3-small
Database: /Users/you/.local/share/nexus-dev/lancedb

📈 Statistics:
   Total chunks: 156
   Code: 98
   Documentation: 58
   Lessons: 0
```

---

## Not Initialized

If run in a directory without `nexus_config.json`:

```
❌ Nexus-Dev not initialized in this directory.
   Run 'nexus-init' to get started.
```

---

## See Also

- [nexus-init](init.md) - Initialize a project
- [get_project_context tool](../tools/indexing.md#get_project_context) - MCP equivalent
