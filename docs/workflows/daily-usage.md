# Daily Usage Patterns

Best practices for using Nexus-Dev in your everyday development workflow.

---

## Starting a Coding Session

Begin each session by loading project context:

```
Call get_project_context() to see project statistics and recent lessons.
```

This helps your AI assistant:

- Understand the project scope
- Recall recent lessons and insights
- Provide more contextual suggestions

---

## Before Writing New Code

### Search First

Before implementing a feature, check if similar code exists:

```
Search the knowledge base for "email validation" to see if we already have utilities for this.
```

Benefits:

- Avoid duplicate implementations
- Follow existing patterns
- Learn from past implementations

### Check for Lessons

When starting work on an area with known complexity:

```
Search lessons for any issues related to "payment processing"
```

---

## During Development

### Record Discoveries

When you learn something non-obvious:

```
Record an insight: I discovered that the third-party API requires URL encoding for special characters in query parameters.
```

### Index New Files

After creating significant new modules:

```
Index the new authentication module: src/auth/
```

---

## After Debugging

### Record the Solution

After solving a non-trivial bug:

```
Record a lesson:
- Problem: "TypeError: 'NoneType' object is not subscriptable in user_service.py"
- Solution: "Added null check for user profile before accessing nested fields"
```

### Include Code Examples

```
Record a lesson with code:
- Problem: Race condition in async cache update
- Solution: Added asyncio.Lock() around cache operations
- Code: [paste the before/after]
```

---

## Code Review

### Use Custom Agents

If you have the code_reviewer agent:

```
Ask code_reviewer to review src/api/endpoints.py for security and best practices.
```

### Search for Patterns

When reviewing unfamiliar code:

```
Search for similar implementations of "rate limiting" in the codebase.
```

---

## Documentation

### Keep Docs Updated

After significant changes, re-index documentation:

```bash
nexus-index docs/ -r
```

### Use doc_writer Agent

```
Ask doc_writer to create API documentation for the UserService class.
```

---

## End of Session

### Record Implementations

After completing a feature:

```
Record implementation:
- Title: "Add OAuth2 login"
- Summary: "Implemented OAuth2 authentication with Google and GitHub providers"
- Approach: "Used authlib for OAuth flow, JWT for session tokens"
- Decisions: ["Chose JWT over sessions for stateless scaling", "Added 30-day remember me option"]
```

### Commit Lessons

If you recorded lessons during the session, commit them:

```bash
git add .nexus/lessons/
git commit -m "Add lessons from debugging session"
```

---

## Weekly Maintenance

### Review Project Context

```
Show me recent lessons and insights from the past week.
```

### Re-index if Needed

After major refactoring:

```bash
nexus-reindex
```

### Check Database Stats

```bash
nexus-status
```

---

## Team Collaboration

### Share Lessons

Commit lessons to git so team members benefit:

```bash
git add .nexus/lessons/
git commit -m "Add lessons: database connection pooling"
git push
```

### Pull New Lessons

After pulling changes:

```bash
# Re-index to include new lessons
nexus-index .nexus/lessons/ -r
```

---

## Quick Reference

### Common Commands

```bash
# Check status
nexus-status

# Index new code
nexus-index src/new_module/ -r

# Search from CLI
nexus-search "authentication"

# Full re-index
nexus-reindex
```

### Common AI Prompts

```
# Start session
Call get_project_context()

# Before coding
Search for existing implementations of "caching"

# After debugging
Record a lesson about the TypeError I just fixed

# After feature
Record the implementation of the new API endpoint
```

---

## Anti-patterns to Avoid

| ❌ Don't | ✅ Do Instead |
|----------|---------------|
| Skip searching before coding | Always search first |
| Forget to record lessons | Record immediately after fixing bugs |
| Index rarely | Index after significant changes |
| Ignore project context | Call `get_project_context()` at session start |
| Keep lessons local only | Commit lessons to share with team |

---

## See Also

- [Search Tools](../tools/search.md) - All search capabilities
- [Learning Tools](../tools/learning.md) - Recording knowledge
- [CLI Reference](../cli/index.md) - All commands
