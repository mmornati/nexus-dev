# nexus-inspect

Inspect database contents for debugging and troubleshooting.

---

## Synopsis

```bash
nexus-inspect [OPTIONS]
```

---

## Description

View detailed information about the Nexus-Dev database, including all projects, document counts, and sample content. Useful for debugging indexing issues and understanding what's stored in the knowledge base.

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--project-id` | TEXT | current | Project ID to inspect |
| `--limit` | INT | `5` | Number of sample documents to show |
| `--all-projects` | FLAG | false | Show all projects in database |

---

## Examples

### Inspect current project

```bash
nexus-inspect
```

**Output:**

```
🔍 Nexus-Dev Database Inspection

Database location: /Users/you/.nexus-dev/db
Database size: 983.04 MB

📊 Total documents across all projects: 10707

📁 Projects in database:
👉 356d304a-fb21-41d2-996c-2048da767ca7: 7366 chunks
   5265af9f-6001-41e0-8f43-eb31b0d97ad9: 2525 chunks

📈 Document types for project 356d304a-fb21-41d2-996c-2048da767ca7:
   code: 5877
   documentation: 1489

📄 Sample documents (limit: 5):
   - [code] authenticate_user
     File: /path/to/auth.py
     Lines: 15-32

   - [documentation] Installation Guide
     File: /path/to/README.md
     Lines: 1-50
```

### View all projects

```bash
nexus-inspect --all-projects
```

Shows a summary of all projects in the database without filtering to a specific one.

### More samples

```bash
nexus-inspect --limit 10
```

Display more sample documents for detailed inspection.

### Inspect specific project

```bash
nexus-inspect --project-id 550e8400-e29b-41d4-a716-446655440000
```

---

## Use Cases

- **Debugging indexing**: Verify files were indexed correctly
- **Cross-project review**: See what's stored across all your projects
- **Database audit**: Check database size and content distribution
- **Troubleshooting**: Identify why search isn't returning expected results

---

## See Also

- [nexus-status](status.md) - Quick project statistics
- [nexus-clean](clean.md) - Delete project data
- [nexus-search](search.md) - Search indexed content
