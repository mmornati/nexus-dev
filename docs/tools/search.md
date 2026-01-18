# Search Tools

Tools for finding code, documentation, and lessons in your knowledge base.

---

## search_knowledge

Search all indexed content with optional type filtering.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Natural language search query |
| `content_type` | string | | `"all"` | Filter: `all`, `code`, `documentation`, `lesson` |
| `project_id` | string | | (current) | Project to search (omit for all projects) |
| `limit` | int | | `5` | Maximum results (max: 20) |

### Example

```
search_knowledge("function that handles user authentication", content_type="code")
```

**Response:**

```markdown
## Search Results [CODE]: 'function that handles user authentication'

### Result 1: [CODE] authenticate_user
**File:** `src/auth/user.py`
**Type:** function (python)
**Lines:** 45-62

​```python
def authenticate_user(username: str, password: str) -> User:
    """Authenticate a user with username and password."""
    user = db.get_user(username)
    if not user or not verify_password(password, user.password_hash):
        raise AuthenticationError("Invalid credentials")
    return user
​```
```

---

## search_code

Search specifically in indexed code (functions, classes, methods).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Description of code to find |
| `project_id` | string | | (all) | Project to search |
| `limit` | int | | `5` | Maximum results |

### Example

```
search_code("class for database connections")
```

---

## search_docs

Search specifically in documentation (Markdown, RST, text).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query |
| `project_id` | string | | (all) | Project to search |
| `limit` | int | | `5` | Maximum results |

### Example

```
search_docs("how to configure the database")
```

---

## search_lessons

Search recorded lessons (problems and solutions).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Problem description or error message |
| `project_id` | string | | (all) | Project to search |
| `limit` | int | | `5` | Maximum results |

### Example

```
search_lessons("TypeError with None value")
```

!!! tip "Cross-Project Learning"
    Omit `project_id` to search lessons across all projects. Solutions from one project can help with similar issues in others.

---

## search_insights

Search recorded insights (discoveries, mistakes, optimizations).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | What you're looking for |
| `category` | string | | (all) | Filter: `mistake`, `discovery`, `backtrack`, `optimization` |
| `project_id` | string | | (all) | Project to search |
| `limit` | int | | `5` | Maximum results |

### Example

```
search_insights("httpx compatibility", category="mistake")
```

---

## search_implementations

Search recorded implementation summaries.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Feature or approach to find |
| `project_id` | string | | (all) | Project to search |
| `limit` | int | | `5` | Maximum results |

### Example

```
search_implementations("authentication flow")
```

---

## Best Practices

1. **Start broad, then narrow**: Use `search_knowledge` first, then switch to specific tools if needed.

2. **Use natural language**: "function that validates email addresses" works better than "email_validator".

3. **Search lessons early**: When encountering an error, search lessons before debugging from scratch.

4. **Cross-project search**: Omit `project_id` when looking for patterns that might exist in other projects.
