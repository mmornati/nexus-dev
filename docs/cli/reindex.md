# nexus-reindex

Clear and rebuild the entire index.

---

## Synopsis

```bash
nexus-reindex
```

---

## Description

Clears the database and re-indexes all files matching the project's include patterns. Use after changing embedding providers or when the index becomes inconsistent.

---

## Example

```bash
nexus-reindex
```

**Output:**

```
🔍 Scanning files...
  Found 42 files to index:

  📁 src/                               28 files
  📁 docs/                              14 files

This will CLEAR the database and re-index the above files. Continue? [y/N]: y
🗑️  Clearing existing index...
   Index cleared and schema updated

📁 Re-indexing project...
  ✅ main.py: 12 chunks
  ✅ utils.py: 8 chunks
  ...

✅ Re-indexed 156 chunks from 42 files
```

---

## When to Use

| Scenario | Use `nexus-reindex` |
|----------|---------------------|
| Changed embedding provider | ✅ Required |
| Changed embedding model | ✅ Required |
| Database schema upgrade | ✅ Required |
| Index seems inconsistent | ✅ Recommended |
| Just want to add new files | ❌ Use `nexus-index` instead |

!!! warning "Destructive Operation"
    This command **deletes all indexed data** before rebuilding. Lessons and insights stored in the database will be preserved only if their source files exist in `.nexus/`.

---

## See Also

- [nexus-index](index-cmd.md) - Add files without clearing
- [Configuration](../getting-started/configuration.md) - Embedding provider options
