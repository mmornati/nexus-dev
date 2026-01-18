# nexus-export

Export project knowledge to markdown files.

---

## Synopsis

```bash
nexus-export [OPTIONS]
```

---

## Description

Exports lessons, insights, and implementations from the knowledge base to portable markdown files. Useful for backup, sharing, or migrating to a different setup.

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--project-id` | TEXT | (current) | Project ID to export |
| `-o, --output` | PATH | `./nexus-export` | Output directory |

---

## Examples

### Export current project

```bash
nexus-export
```

**Output:**

```
Exporting knowledge for project: 550e8400-e29b-41d4-a716-446655440000
  - Found 15 lessons
  - Found 8 insights
  - Found 3 implementations

Successfully exported 26 files to ./nexus-export
```

### Custom output directory

```bash
nexus-export -o ./backup/knowledge
```

### Export specific project

```bash
nexus-export --project-id "other-project-id"
```

---

## Output Structure

```
nexus-export/
├── lessons/
│   ├── lesson_abc123.md
│   └── lesson_def456.md
├── insights/
│   ├── insight_ghi789.md
│   └── insight_jkl012.md
└── implementations/
    └── impl_mno345.md
```

Each file preserves the original metadata in YAML frontmatter, allowing re-import with `nexus-index`.

---

## Re-importing Exported Files

Exported files can be re-indexed using `nexus-index`:

```bash
nexus-index nexus-export/lessons/ nexus-export/insights/ -r
```

The indexer detects the document type from frontmatter and preserves timestamps.

---

## See Also

- [nexus-index](index-cmd.md) - Re-import exported files
- [record_lesson tool](../tools/learning.md) - Create new lessons
