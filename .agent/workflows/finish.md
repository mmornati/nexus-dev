---
description: Complete a task and record learnings
---

# Finish Task Workflow

Use this workflow after completing an implementation to ensure all knowledge is captured.

## Steps

// turbo-all

1. **Review the work done**
   - What feature/fix was implemented?
   - Were there any mistakes or backtracking during development?
   - What were the key design decisions?

2. **Check for uncaptured insights**
   ```
   search_insights("<topic related to this work>")
   ```
   
   If you made mistakes or backtracked but didn't record them yet:
   ```
   record_insight(
       category="mistake",  # or "backtrack", "discovery", "optimization"
       description="<what happened>",
       reasoning="<your thinking>",
       correction="<how you fixed it>"
   )
   ```

3. **Record the implementation**
   ```
   record_implementation(
       title="<feature/fix title>",
       summary="<what was built>",
       approach="<how it was built>",
       design_decisions=[
           "Chose X over Y because...",
           "Used pattern Z for..."
       ],
       files_changed=["list", "of", "modified", "files"]
   )
   ```

4. **Verify it was saved**
   ```
   search_implementations("<feature name>")
   ```

## When to Use

- After completing a user-requested feature
- After finishing a complex refactor
- After fixing a significant bug
- When you want to ensure nothing was missed

## Example

After implementing authentication:

```
# Check if already captured
search_insights("authentication")

# Record any mistakes made
record_insight(
    category="mistake",
    description="Used bcrypt 4.0 which has breaking API changes",
    reasoning="Assumed latest version would be compatible",
    correction="Downgraded to bcrypt 3.2 based on project dependencies"
)

# Record the implementation
record_implementation(
    title="Add JWT Authentication",
    summary="Implemented JWT-based authentication with refresh tokens",
    approach="Used PyJWT library with RSA256 signing, Redis for token blacklist",
    design_decisions=[
        "Chose JWT over sessions for stateless API",
        "Used Redis for blacklist due to TTL support and speed",
        "Implemented refresh token rotation for security"
    ],
    files_changed=[
        "src/auth/jwt_handler.py",
        "src/auth/middleware.py",
        "src/api/routes/auth.py"
    ]
)
```
