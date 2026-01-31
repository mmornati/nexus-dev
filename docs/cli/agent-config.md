# nexus-agent-config

Generate AI agent configuration files for your project (AGENTS.md and editor-specific rules).

## Usage

```bash
nexus-agent-config [OPTIONS]
```

## Description

This command creates a comprehensive `AGENTS.md` file in your project root, which serves as the primary source of truth for AI agents working in your repository. It explains the project architecture, available tools, and guidelines.

It also configures your specific editor (Cursor, Claude, VS Code, etc.) to use this documentation, creating necessary rule files (e.g., `.cursorrules`, `CLAUDE.md`, `.geminiignore`).

## Options

| Option | Description |
|--------|-------------|
| `--editor [editor]` | Explicitly specify the target editor (e.g., `cursor`, `claude`, `antigravity`). If not provided, it is auto-detected. |
| `--update` | Update existing configuration files (merges content where possible). Default is to skip existing files. |
| `--help` | Show help message and exit. |

## Supported Editors

- **Antigravity** (`.geminiignore`, `.antigravityignore`)
- **Cursor** (`.cursorrules`)
- **Claude** (`CLAUDE.md`)
- **GitHub Copilot** (`.github/copilot-instructions.md`)
- **Windsurf** (`.windsurfrules`)
- **Gemini Code Assist** (`GEMINI.md`)
- **Zed** (`.rules`)
- **VS Code** (Generic setup)

## Examples

Initialize configuration for the auto-detected editor:
```bash
nexus-agent-config
```

Force configuration for Cursor:
```bash
nexus-agent-config --editor cursor
```

Update existing configuration:
```bash
nexus-agent-config --update
```
