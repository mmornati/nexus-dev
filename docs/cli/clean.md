# nexus-clean

Safely delete indexed data for a project.

---

## Synopsis

```bash
nexus-clean [OPTIONS]
```

---

## Description

Remove all indexed chunks for a specific project from the knowledge base. Includes safety features like confirmation prompts and dry-run mode to prevent accidental data loss.

!!! warning "Data Deletion"
    This command permanently deletes indexed data. Use `--dry-run` first to preview what will be deleted.

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--project-id` | TEXT | current | Project ID to clean |
| `--all` | FLAG | false | Delete ALL projects (dangerous!) |
| `--dry-run` | FLAG | false | Show what would be deleted without deleting |

---

## Examples

### Preview deletion (dry-run)

```bash
nexus-clean --dry-run
```

**Output:**

```
Found 3 documents for project: verify_test

Document types:
  - lesson: 3

[DRY RUN] Would delete 3 documents for project verify_test
```

### Delete current project

```bash
nexus-clean
```

**Output:**

```
Found 156 documents for project: my-project

Document types:
  - code: 98
  - documentation: 58

Delete 156 documents? [y/N]: y
✅ Deleted 156 documents for project my-project
```

### Delete specific project

```bash
nexus-clean --project-id 550e8400-e29b-41d4-a716-446655440000
```

### Delete all projects (dangerous)

```bash
nexus-clean --all
```

**Output:**

```
⚠️  WARNING: This will delete ALL 10707 documents from the database!

Are you absolutely sure? [y/N]: y
✅ Deleted all 10707 documents
```

---

## Use Cases

- **Force re-indexing**: Clean project before running `nexus-reindex`
- **Remove old data**: Delete data from projects you no longer work on
- **Database cleanup**: Free up disk space by removing unused projects
- **Testing**: Reset knowledge base during development

---

## Safety Features

1. **Confirmation prompts**: Always asks before deleting
2. **Dry-run mode**: Preview deletions with `--dry-run`
3. **Detailed breakdown**: Shows what will be deleted by document type
4. **Current project default**: Requires explicit `--all` flag for full deletion

---

## See Also

- [nexus-reindex](reindex.md) - Re-index entire project
- [nexus-inspect](inspect.md) - View database contents
- [nexus-status](status.md) - Check project statistics
