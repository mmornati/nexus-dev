---
description: Index new files after creating important modules
---

# Index New Code Workflow

Use this workflow after creating new important files to add them to the knowledge base.

## Steps

1. After creating a new file, index it:
   ```
   index_file("<path/to/new/file.py>")
   ```

2. Verify it was indexed:
   ```
   search_code("<main function or class name from new file>")
   ```

## When to Index

**DO index:**
- New modules with reusable functions or classes
- New API endpoints or handlers
- Configuration files with important settings
- Important utility functions

**DON'T index (automatically handled by reindex):**
- Test files
- Generated files
- Temporary or scratch files

## Bulk Reindex

To reindex the entire project:
// turbo
```bash
cd /Users/you/Projects/nexus-dev && nexus-reindex --yes
```

## When to Use

- After creating a new important source file
- After major refactoring that changes file structure
- When you notice search results are stale