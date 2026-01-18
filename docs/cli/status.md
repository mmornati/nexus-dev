# nexus-status

Show Nexus-Dev status and project statistics.

---

## Synopsis

```bash
nexus-status [OPTIONS]
```

---

## Description

Displays project configuration and knowledge base statistics including chunk counts by type.

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-v, --verbose` | FLAG | false | Show detailed debug information |

---

## Examples

### Basic usage

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
Database: /Users/you/.nexus-dev/db

📈 Statistics:
   Total chunks: 156
   Code: 98
   Documentation: 58
   Lessons: 0
```

### Verbose mode

```bash
nexus-status --verbose
```

**Output:**

```
📊 Nexus-Dev Status

Project: my-project
Project ID: 550e8400-e29b-41d4-a716-446655440000
Embedding Provider: openai
Embedding Model: text-embedding-3-small
Database: /Users/you/.nexus-dev/db

🔍 Debug Info:
   Database path exists: True
   Querying for project_id: 550e8400-e29b-41d4-a716-446655440000

📈 Statistics:
   Total chunks: 156
   Code: 98
   Documentation: 58
   Lessons: 0

   Document type breakdown:
     - code: 98
     - documentation: 58
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
- [nexus-inspect](inspect.md) - Detailed database inspection
- [get_project_context tool](../tools/indexing.md#get_project_context) - MCP equivalent
