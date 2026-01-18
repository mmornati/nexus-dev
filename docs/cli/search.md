# nexus-search

Search the knowledge base from the command line.

---

## Synopsis

```bash
nexus-search [OPTIONS] QUERY
```

---

## Description

Performs semantic search across indexed content with optional type filtering.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `QUERY` | Search query (natural language) |

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--type` | TEXT | (all) | Filter by content type |
| `--limit` | INT | `5` | Number of results |

Valid `--type` values: `code`, `documentation`, `lesson`, `insight`, `implementation`

---

## Examples

### Basic search

```bash
nexus-search "authentication function"
```

**Output:**

```
🔍 Searching for 'authentication function'...

Found 3 results:

1. [CODE] authenticate_user (Score: 0.892)
   path: src/auth/user.py
   "def authenticate_user(username: str, password: str) -> User: ..."

2. [CODE] verify_token (Score: 0.784)
   path: src/auth/token.py
   "def verify_token(token: str) -> bool: Verify JWT token..."

3. [DOCUMENTATION] Authentication (Score: 0.723)
   path: docs/auth.md
   "# Authentication This module handles user authentication using..."
```

### Filter by type

```bash
nexus-search "database connection" --type code
```

### More results

```bash
nexus-search "error handling" --limit 10
```

---

## See Also

- [search_knowledge tool](../tools/search.md#search_knowledge) - MCP equivalent
- [search_code tool](../tools/search.md#search_code) - Code-specific search
