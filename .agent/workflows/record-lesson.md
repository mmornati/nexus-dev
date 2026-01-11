---
description: Record a debugging lesson after solving a problem
---

# Record Lesson Workflow

Use this workflow after solving a non-trivial bug or problem to save it for future reference.

## Steps

1. Identify the problem you just solved
   - What error or unexpected behavior occurred?
   - What file(s) were affected?

2. Document the solution:
   ```
   record_lesson(
       problem="<clear description of the problem>",
       solution="<how you fixed it>",
       context="<file path or additional context>"
   )
   ```

3. Verify the lesson was recorded:
   ```
   search_lessons("<keywords from your problem>")
   ```

## Example

```
record_lesson(
    problem="TypeError: Cannot read property 'id' of undefined when fetching user",
    solution="Added null check before accessing user.id, return early if user is undefined",
    context="src/services/user_service.py line 45"
)
```

## When to Use

- After fixing a bug that took more than a few minutes
- After solving a tricky edge case
- After debugging a configuration issue
- After resolving a dependency conflict
