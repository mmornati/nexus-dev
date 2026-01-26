# ADR-002: Multi-Editor Agent Configuration Command

| Status   | Proposed |
|----------|----------|
| Date     | 2026-01-26 |
| Authors  | @mmornati |

## Context

### Problem Statement

When projects are initialized with `nexus-init`, the command outputs agent instructions to the console. Users must manually copy these instructions into their editor's agent configuration file. This creates friction because:

1. **Different editors use different config files** - Cursor, Claude, Copilot, Windsurf, Gemini, Zed all have unique formats
2. **Manual copy-paste is error-prone** - Instructions may be incomplete or outdated
3. **Configuration isn't version-controlled** - Console output is ephemeral
4. **No update mechanism** - When nexus-dev evolves, existing configs become stale

### Research: Editor Configuration Files

| Editor/Tool | Config File(s) | Location | Format |
|-------------|---------------|----------|--------|
| **Cursor** | `.cursorrules` | Root | Markdown |
| | `.cursor/rules/*.md` | Rules folder | Markdown with frontmatter |
| | `AGENTS.md` | Root | Plain markdown |
| **Claude Code** | `CLAUDE.md` | Root or `.claude/` | Markdown |
| | `~/.claude/CLAUDE.md` | Global | Markdown |
| **GitHub Copilot** | `.github/copilot-instructions.md` | `.github/` | Markdown |
| **Windsurf** | `.windsurfrules` | Root | Markdown |
| | `.windsurf/rules/*.md` | Rules folder | Markdown |
| **Gemini** | `GEMINI.md` | Root | Markdown |
| | `AGENTS.md` | Root | Markdown (Android Studio) |
| **Zed** | `.rules` | Root | Markdown |
| | `CLAUDE.md`, `AGENT.md` | Root | Compatible aliases |

### Emerging Standard: AGENTS.md

`AGENTS.md` is emerging as an interoperability standard across tools. Both Cursor and Gemini explicitly support it. Creating symlinks from tool-specific files to `AGENTS.md` allows a single source of truth.

## Decision

**Add a `nexus-agent-config` CLI command** that:

1. Creates or updates agent configuration files for the current project
2. Detects the editor/tool in use or accepts explicit `--editor` flag
3. Uses `AGENTS.md` as the canonical source with optional symlinks
4. Includes nexus-dev tool instructions, project context, and session cache usage

## Technical Details

### New CLI Command

```bash
# Auto-detect editor and create appropriate config
nexus-agent-config

# Specify editor explicitly
nexus-agent-config --editor cursor
nexus-agent-config --editor claude
nexus-agent-config --editor copilot
nexus-agent-config --editor windsurf
nexus-agent-config --editor gemini
nexus-agent-config --editor zed

# Also create symlinks/copies for other editors
nexus-agent-config --all

# Update existing config (merge with new instructions)
nexus-agent-config --update
```

### Editor Detection Logic

```python
def detect_editor() -> str | None:
    """Detect which AI coding editor is likely in use."""
    cwd = Path.cwd()
    
    # Check for editor-specific markers
    if (cwd / ".cursor").exists() or (cwd / ".cursorrules").exists():
        return "cursor"
    if (cwd / ".claude").exists() or (cwd / "CLAUDE.md").exists():
        return "claude"
    if (cwd / ".github" / "copilot-instructions.md").exists():
        return "copilot"
    if (cwd / ".windsurfrules").exists() or (cwd / ".windsurf").exists():
        return "windsurf"
    if (cwd / "GEMINI.md").exists():
        return "gemini"
    if (cwd / ".rules").exists():
        return "zed"
    
    # Default to AGENTS.md (universal)
    return "agents"
```

### File Generation Strategy

| Editor | Primary File | Secondary (optional) |
|--------|-------------|---------------------|
| `agents` | `AGENTS.md` | - |
| `cursor` | `AGENTS.md` | `.cursorrules` (symlink) |
| `claude` | `AGENTS.md` | `CLAUDE.md` (symlink) |
| `copilot` | `AGENTS.md` | `.github/copilot-instructions.md` (copy) |
| `windsurf` | `AGENTS.md` | `.windsurfrules` (symlink) |
| `gemini` | `AGENTS.md` | `GEMINI.md` (symlink) |
| `zed` | `AGENTS.md` | `.rules` (symlink) |

### Template Content

The generated `AGENTS.md` includes:

```markdown
# <Project Name> - AI Agent Instructions

## Project Overview
<!-- Auto-filled from nexus_config.json or prompted -->

## Nexus-Dev Knowledge Base

This project uses nexus-dev for persistent AI memory.

**Project ID:** `<project_id>`

### Mandatory Search-First Workflow

BEFORE answering questions about this code:
1. `search_knowledge("<query>")` - Search code, docs, and lessons
2. `search_lessons("<error>")` - Check for past solutions
3. `search_session_context("<query>")` - Check this session's discoveries

### Recording Knowledge

After solving non-trivial problems:
- `record_lesson(problem="...", solution="...")` - Permanent lesson
- `cache_session_context(content="...", context_type="discovery")` - Session cache
- `record_insight(category="mistake|discovery", ...)` - Capture reasoning

### Session Best Practice

- Start each session: `get_project_context()`
- Search before implementing: `search_session_context("<intent>")`
- Cache discoveries: `cache_session_context(...)` 
- Promote valuable findings: `promote_to_lesson(...)`

## Additional Instructions
<!-- User can add custom content here -->
```

### Integration with nexus-init

The `nexus-init` command should prompt to run `nexus-agent-config`:

```python
# At end of init_command:
if click.confirm("Create agent configuration file (AGENTS.md)?", default=True):
    ctx.invoke(agent_config_command, editor=None)
```

## Consequences

### Positive

- **Zero-friction onboarding** - Configuration file created automatically
- **Version-controlled** - Agent instructions live with the code
- **Consistent** - All team members use same instructions
- **Updatable** - `--update` flag merges new capabilities

### Negative

- **File proliferation** - Multiple config files if using symlinks
- **Merge conflicts** - Generated content may conflict with user edits
- **Editor detection** - May mis-detect or fail to detect editor

### Mitigations

| Risk | Mitigation |
|------|------------|
| File proliferation | Use symlinks where supported, single `AGENTS.md` source |
| Merge conflicts | Use markers to separate generated vs. user content |
| Editor detection | Accept explicit `--editor` flag, default to universal format |

## Implementation Plan

### Phase 1: Core Command
- [ ] Add `agent_config_command` to cli.py
- [ ] Implement editor detection logic
- [ ] Create AGENTS.md template with nexus-dev instructions
- [ ] Integrate session cache instructions when Phase 1 complete

### Phase 2: Multi-Editor Support
- [ ] Add symlink/copy logic for each editor type
- [ ] Implement `--all` flag for multi-editor projects
- [ ] Add `--update` flag for safe merging

### Phase 3: Integration
- [ ] Prompt during `nexus-init` to create config
- [ ] Add `nexus-agent-config --check` for validation
- [ ] Document in MkDocs site

## References

- [Cursor Rules Documentation](https://cursor.com/docs)
- [Claude Code CLAUDE.md](https://docs.anthropic.com/en/docs/claude-code)
- [GitHub Copilot Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot)
- [Windsurf Rules](https://windsurf.com/docs)
- [Gemini Code Assist](https://cloud.google.com/products/gemini/code-assist)
- [Zed AI Rules](https://zed.dev/docs/assistant/agent)
