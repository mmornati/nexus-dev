# Multi-Repository Projects

Nexus-Dev supports multi-repository project setups where a parent folder contains the nexus configuration and multiple sub-folders contain independent git repositories.

## Overview

In a multi-repository (mono-repo style) project, you might have:

```
/artemis-coach/                      # Parent folder (no .git)
├── nexus_config.json                # Single configuration
├── .nexus/
│   └── lessons/                     # Centralized lessons
├── frontend/                        # Sub-repository
│   └── .git/
├── backend/                         # Sub-repository
│   └── .git/
└── shared/                          # Sub-repository
    └── .git/
```

All repositories share:
- **Single project ID** - One knowledge base for the entire project
- **Centralized lessons** - All lessons stored in parent `.nexus/lessons/`
- **Unified indexing** - Changes in any repo index to the same database

## Setup

### Step 1: Initialize Parent Project

From the parent directory:

```bash
cd /path/to/artemis-coach
nexus-init --project-name "Artemis Coach"
```

This creates:
- `nexus_config.json` - Project configuration
- `.nexus/lessons/` - Centralized lesson storage
- Database in `~/.nexus-dev/db/`

### Step 2: Install Hooks in Sub-Repositories

You have two options:

#### Option A: Manual Installation (Per Repository)

Install hooks one at a time:

```bash
cd frontend
nexus-init --link-hook

cd ../backend
nexus-init --link-hook

cd ../shared
nexus-init --link-hook
```

**Output:**
```
✅ Installed pre-commit hook (linked to artemis-coach/)
✅ Linked to parent project: Artemis Coach
   Project ID: artemis-coach_abc123
   Project Root: /path/to/artemis-coach
```

#### Option B: Auto-Discovery (All Repositories)

Install hooks in all sub-repositories at once:

```bash
cd /path/to/artemis-coach
nexus-init --discover-repos
```

**Output:**
```
Found 3 git repositories:
  📁 frontend
  📁 backend
  📁 shared

Install hooks in all repositories? [y/N]: y
  ✅ frontend
  ✅ backend
  ✅ shared

✅ Installed hooks in 3/3 repositories
   All repositories linked to project: Artemis Coach
```

## How It Works

### Project Root Discovery

The pre-commit hook automatically discovers the project root by walking up the directory tree to find `nexus_config.json`:

1. Hook executes in sub-repository (e.g., `frontend/`)
2. Walks up: `frontend/` → `artemis-coach/` → finds `nexus_config.json`
3. Changes to project root for indexing
4. All files indexed to parent project database

### Indexing Behavior

When you commit in a sub-repository:

```bash
cd frontend
git add src/App.tsx
git commit -m "Add new component"
```

The hook:
1. Finds parent `nexus_config.json`
2. Indexes `frontend/src/App.tsx` to parent project
3. Uses parent project ID for storage
4. All changes searchable via `search_code()` and `search_knowledge()`

## Best Practices

### Centralized Lessons

Store all lessons in the parent `.nexus/lessons/` folder:

```bash
/artemis-coach/.nexus/lessons/
├── lesson_001.md  # Frontend lesson
├── lesson_002.md  # Backend lesson
└── lesson_003.md  # Integration lesson
```

### Single Source of Truth

- Keep `nexus_config.json` only at the parent level
- Don't create separate configs in sub-repositories
- Sub-repositories automatically link to parent configuration

### Workflows

Use workflows from the parent directory:

```bash
cd /path/to/artemis-coach
# Use /search-first, /record-lesson, etc.
```

All nexus-dev MCP tools work across all repositories:
- `search_code()` - Searches all indexed code
- `record_lesson()` - Stores in parent `.nexus/lessons/`
- `get_project_context()` - Shows stats across all repos

## Troubleshooting

### Hook Not Finding Config

If the hook reports "No nexus_config.json found":

1. Verify parent config exists:
   ```bash
   ls /path/to/artemis-coach/nexus_config.json
   ```

2. Check hook was installed correctly:
   ```bash
   cd sub-repo
   cat .git/hooks/pre-commit | head -20
   ```
   
   Should contain project root discovery logic starting with `REPO_ROOT=$(git rev-parse --show-toplevel)`

3. Reinstall hook:
   ```bash
   cd sub-repo
   rm .git/hooks/pre-commit
   nexus-init --link-hook
   ```

### Files Not Indexing

Check that the hook is executable:

```bash
ls -la .git/hooks/pre-commit
# Should show: -rwxr-xr-x
```

If not:
```bash
chmod +x .git/hooks/pre-commit
```

### Verify Project ID

Ensure all repositories use the same project ID:

```bash
cd /path/to/artemis-coach
nexus-status  # Shows project ID

cd frontend
git commit --allow-empty -m "test"
# Hook output should show parent project root
```

## Migration from Separate Configs

If you previously had separate `nexus_config.json` files in each sub-repository:

1. **Backup data:**
   ```bash
   cd sub-repo
   nexus-export --output ../sub-repo-backup
   ```

2. **Remove sub-repo config:**
   ```bash
   rm nexus_config.json
   ```

3. **Link to parent:**
   ```bash
   rm .git/hooks/pre-commit
   nexus-init --link-hook
   ```

4. **Re-index from parent:**
   ```bash
   cd /path/to/parent
   nexus-index sub-repo/ -r
   ```

## Related Documentation

- [CLI Reference: nexus-init](../cli/init.md) - Full CLI options
- [Installation Guide](../getting-started/installation.md) - Initial setup
