---
description: Search the knowledge base before implementing a feature
---

# Search Before Implementing Workflow

Use this workflow before writing new code to find existing implementations and patterns.

## Steps

1. Search for similar existing code:
   ```
   search_code("<feature or function description>")
   ```

2. Check documentation for guidance:
   ```
   search_docs("<topic or configuration>")
   ```

3. Look for relevant lessons:
   ```
   search_lessons("<potential issues in this area>")
   ```

4. Review results and identify:
   - Existing patterns to follow
   - Potential pitfalls to avoid
   - Related code that might need updates

## Example: Implementing User Authentication

```
# Find existing auth code
search_code("authentication login")

# Check for auth documentation
search_docs("authentication setup")

# Look for past auth issues
search_lessons("authentication error")
```

## When to Use

- Before implementing any new feature
- Before refactoring existing code
- When you're unsure how something is currently done
- When joining a new area of the codebase
