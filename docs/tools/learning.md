# Learning Tools

Tools for capturing knowledge: lessons, insights, and implementation records.

---

## record_lesson

Record a problem/solution pair for future reference.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem` | string | ✅ | - | Description of the problem |
| `solution` | string | ✅ | - | How the problem was resolved |
| `context` | string | | - | Additional context (file path, library, error) |
| `code_snippet` | string | | - | Code demonstrating problem or solution |
| `project_id` | string | | (current) | Project identifier |

### Example

```
record_lesson(
    problem="TypeError when passing None to user_service.get_user()",
    solution="Added null check before calling get_user() and return early if None",
    context="src/api/handlers.py, line 45",
    code_snippet="""
# Before (broken)
user = user_service.get_user(user_id)

# After (fixed)  
if user_id is None:
    return None
user = user_service.get_user(user_id)
"""
)
```

**Response:**

```
✅ Lesson recorded: lesson_abc123

Saved to: .nexus/lessons/lesson_abc123.md
```

### When to Record Lessons

| Scenario | Record? |
|----------|---------|
| Fixed a non-obvious bug | ✅ Yes |
| Solved a cryptic error message | ✅ Yes |
| Found a workaround for library issue | ✅ Yes |
| Typo fix | ❌ No |
| Trivial change | ❌ No |

---

## record_insight

Capture discoveries, mistakes, or optimizations during development.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `category` | string | ✅ | - | `discovery`, `mistake`, `backtrack`, `optimization` |
| `description` | string | ✅ | - | What happened |
| `reasoning` | string | ✅ | - | Why it happened / what you were thinking |
| `correction` | string | | - | How it was fixed (for mistakes/backtracks) |
| `files_affected` | array | | - | List of affected file paths |
| `project_id` | string | | (current) | Project identifier |

### Categories

| Category | Description | Example |
|----------|-------------|---------|
| `discovery` | Useful pattern or gotcha found | "This API requires URL encoding" |
| `mistake` | Wrong approach taken | "Used httpx 0.23 which is incompatible with Python 3.13" |
| `backtrack` | Had to undo and redo | "Should have used async from the start" |
| `optimization` | Performance improvement found | "Batching API calls reduced latency 10x" |

### Example

```
record_insight(
    category="mistake",
    description="Used httpx 0.23 which is incompatible with Python 3.13",
    reasoning="Assumed latest version would work, didn't check compatibility",
    correction="Upgraded to httpx 0.27 which supports Python 3.13"
)
```

---

## record_implementation

Document a completed feature or significant work.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `title` | string | ✅ | - | Short title |
| `summary` | string | ✅ | - | What was implemented (1-3 sentences) |
| `approach` | string | ✅ | - | Technical approach/architecture |
| `design_decisions` | array | ✅ | - | Key decisions with rationale |
| `files_changed` | array | ✅ | - | List of modified/created files |
| `related_plan` | string | | - | Path or URL to implementation plan |
| `project_id` | string | | (current) | Project identifier |

### Example

```
record_implementation(
    title="Add user authentication",
    summary="Implemented JWT-based authentication with refresh tokens",
    approach="Used PyJWT for token generation, Redis for refresh token storage",
    design_decisions=[
        "Used JWT over sessions for stateless auth",
        "Chose Redis for session cache due to speed requirements",
        "Added 15-min access token expiry with 7-day refresh tokens"
    ],
    files_changed=[
        "src/auth/jwt.py",
        "src/auth/middleware.py",
        "src/api/login.py"
    ]
)
```

---

## Storage and Retrieval

### File Storage

All recorded knowledge is saved to `.nexus/`:

```
.nexus/
├── lessons/
│   └── lesson_abc123.md
├── insights/
│   └── insight_def456.md
└── implementations/
    └── impl_ghi789.md
```

### Searching Recorded Knowledge

Use the corresponding search tools:

```
search_lessons("database connection error")
search_insights("performance", category="optimization")
search_implementations("authentication")
```

---

## Best Practices

1. **Record immediately**: Capture lessons right after solving problems while context is fresh.

2. **Be specific**: Include error messages, file paths, and code snippets.

3. **Explain the "why"**: Future you (or the AI) needs to understand the reasoning.

4. **Cross-reference**: Link related lessons and implementations.

5. **Review periodically**: Use `get_project_context()` to see recent recordings.
