# nexus-index

Manually index files or directories into the knowledge base.

---

## Synopsis

```bash
nexus-index [OPTIONS] PATHS...
```

---

## Description

Indexes code and documentation files into the vector database for semantic search. Supports Python, JavaScript/TypeScript, Java, Markdown, RST, and plain text.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `PATHS` | One or more files or directories to index |

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-r, --recursive` | FLAG | `False` | Index directories recursively |
| `-q, --quiet` | FLAG | `False` | Suppress output |

---

## Examples

### Index a single file

```bash
nexus-index src/main.py
```

### Index a directory recursively

```bash
nexus-index src/ -r
```

**Output:**

```
  Found 28 files to index:

  📁 src/                               28 files

Proceed with indexing? [y/N]: y
  ✅ main.py: 12 chunks
  ✅ utils.py: 8 chunks
  ✅ database.py: 15 chunks
  ...
✅ Indexed 156 chunks from 28 files
```

### Index multiple directories

```bash
nexus-index src/ docs/ tests/ -r
```

### Index specific files

```bash
nexus-index main.py utils.py config.py
```

### Silent mode (for scripts)

```bash
nexus-index src/ -r -q
```

---

## File Type Support

| Extension | Language | Chunker |
|-----------|----------|---------|
| `.py`, `.pyw` | Python | AST-based (functions, classes) |
| `.js`, `.jsx`, `.mjs`, `.cjs` | JavaScript | Tree-sitter |
| `.ts`, `.tsx`, `.mts`, `.cts` | TypeScript | Tree-sitter |
| `.java` | Java | Tree-sitter |
| `.md`, `.markdown` | Markdown | Section-based |
| `.rst` | reStructuredText | Section-based |
| `.txt` | Plain text | Paragraph-based |

---

## Include/Exclude Patterns

Files are indexed based on patterns in `nexus_config.json`:

```json
{
  "include_patterns": ["**/*.py", "**/*.js", "**/*.ts", "**/*.java"],
  "exclude_patterns": [
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/build/**",
    "**/dist/**"
  ]
}
```

!!! tip "Overriding Patterns"
    When indexing specific files directly (not directories), exclude patterns are ignored. This allows indexing files that would normally be excluded.

---

## How Chunking Works

Code files are parsed using language-aware chunkers that extract:

- **Functions**: Complete function definitions with signatures and bodies
- **Classes**: Class definitions with methods
- **Methods**: Individual methods within classes
- **Top-level code**: Import statements and module-level code

Documentation files are chunked by:

- **Sections**: Headers and their content
- **Paragraphs**: For plain text

---

## See Also

- [nexus-reindex](reindex.md) - Clear and rebuild the entire index
- [nexus-status](status.md) - View current index statistics
- [index_file tool](../tools/indexing.md) - Index files via MCP
