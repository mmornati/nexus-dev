"""Initialization commands."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import click

from nexus_dev.cli.utils import (
    _create_symlink,
    _find_project_root,
    _install_antigravity_files,
    _install_hook,
    _update_gitignore,
    detect_editor,
)
from nexus_dev.config import NexusConfig


@click.command("init")
@click.option(
    "--project-name",
    help="Human-readable name for the project",
)
@click.option(
    "--embedding-provider",
    type=click.Choice(["openai", "ollama"]),
    default="ollama",
    help="Embedding provider to use (default: ollama)",
)
@click.option(
    "--install-hook/--no-hook",
    default=False,
    help="Install pre-commit hook for automatic indexing",
)
@click.option(
    "--link-hook",
    is_flag=True,
    default=False,
    help="Install hook linked to parent project configuration (for multi-repo projects)",
)
@click.option(
    "--discover-repos",
    is_flag=True,
    default=False,
    help="Auto-discover git repositories and offer to install hooks",
)
def init_command(
    project_name: str | None,
    embedding_provider: Literal["openai", "ollama"],
    install_hook: bool,
    link_hook: bool,
    discover_repos: bool,
) -> None:
    """Initialize Nexus-Dev in the current repository.

    Creates configuration file, lessons directory, and optionally installs
    the pre-commit hook for automatic indexing.

    Multi-repository projects:
    - Use --link-hook to install a hook in a sub-repository that links to parent config
    - Use --discover-repos to auto-find all git repos and install hooks
    """
    cwd = Path.cwd()

    # Handle --link-hook: Install hook in sub-repo linked to parent config
    if link_hook:
        git_dir = cwd / ".git"
        if not git_dir.exists():
            click.echo("❌ Not a git repository. Cannot install hook.", err=True)
            return

        # Find parent project root
        project_root = _find_project_root(cwd.parent)
        if not project_root:
            click.echo("❌ No parent nexus_config.json found.", err=True)
            click.echo(
                "   Run 'nexus-init' in the parent directory first, "
                "or use 'nexus-init' without --link-hook to create a new project."
            )
            return

        # Load parent config to display project info
        parent_config = NexusConfig.load(project_root / "nexus_config.json")

        # Install hook
        _install_hook(cwd, project_root)

        click.echo("")
        click.echo(f"✅ Linked to parent project: {parent_config.project_name}")
        click.echo(f"   Project ID: {parent_config.project_id}")
        click.echo(f"   Project Root: {project_root}")
        return

    # Handle --discover-repos: Find and install hooks in all sub-repositories
    if discover_repos:
        # Ensure we have a config in current directory
        config_path = cwd / "nexus_config.json"
        if not config_path.exists():
            click.echo("❌ No nexus_config.json in current directory.", err=True)
            click.echo("   Run 'nexus-init' first to create project configuration.")
            return

        config = NexusConfig.load(config_path)

        # Find all .git directories
        git_repos = []
        for root, dirs, _ in os.walk(cwd):
            # Skip the root .git if there is one
            if ".git" in dirs:
                repo_path = Path(root)
                if repo_path != cwd:  # Don't include parent directory itself
                    git_repos.append(repo_path)
                # Don't traverse into .git directories
                dirs.remove(".git")

        if not git_repos:
            click.echo("No git repositories found in subdirectories.")
            return

        click.echo(f"Found {len(git_repos)} git repositor{'y' if len(git_repos) == 1 else 'ies'}:")
        for repo in git_repos:
            rel_path = repo.relative_to(cwd)
            click.echo(f"  📁 {rel_path}")

        click.echo("")
        if not click.confirm("Install hooks in all repositories?"):
            click.echo("Aborted.")
            return

        # Install hooks
        installed = 0
        for repo in git_repos:
            try:
                _install_hook(repo, cwd)
                installed += 1
                rel_path = repo.relative_to(cwd)
                click.echo(f"  ✅ {rel_path}")
            except Exception as e:
                rel_path = repo.relative_to(cwd)
                click.echo(f"  ❌ {rel_path}: {e}")

        click.echo("")
        click.echo(f"✅ Installed hooks in {installed}/{len(git_repos)} repositories")
        click.echo(f"   All repositories linked to project: {config.project_name}")
        return

    # Normal initialization flow
    config_path = cwd / "nexus_config.json"

    # Prompt for project name if not provided
    if not project_name:
        project_name = click.prompt("Project name")

    # Check if already initialized
    if config_path.exists():
        click.echo("⚠️  nexus_config.json already exists.")
        if not click.confirm("Overwrite existing configuration?"):
            click.echo("Aborted.")
            return

    # Create configuration
    config = NexusConfig.create_new(
        project_name=project_name,
        embedding_provider=embedding_provider,
    )
    config.save(config_path)
    click.echo("✅ Created nexus_config.json")

    # Create .nexus/lessons directory
    lessons_dir = cwd / ".nexus" / "lessons"
    lessons_dir.mkdir(parents=True, exist_ok=True)
    click.echo("✅ Created .nexus/lessons/")

    # Create .gitkeep so the directory is tracked
    gitkeep = lessons_dir / ".gitkeep"
    gitkeep.touch(exist_ok=True)

    # Create database directory
    db_path = config.get_db_path()
    db_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"✅ Created database directory at {db_path}")

    # Optionally install pre-commit hook
    if install_hook:
        _install_hook(cwd, cwd)

    # Configure .gitignore
    click.echo("")
    ignore_choice = click.prompt(
        "Configure .gitignore for .nexus folder?",
        type=click.Choice(["allow-lessons", "ignore-all", "skip"]),
        default="allow-lessons",
        show_default=True,
    )

    if ignore_choice != "skip":
        _update_gitignore(cwd, ignore_choice)

    click.echo("")
    click.echo(f"Project ID: {config.project_id}")

    if embedding_provider == "ollama":
        click.echo("")
        click.echo(
            "ℹ️  Using Ollama embeddings (local). Ensure Ollama is running at http://localhost:11434"
        )
    else:
        click.echo("")
        click.echo("⚠️  Using OpenAI embeddings. Ensure OPENAI_API_KEY is set.")

    click.echo("")
    click.echo("")
    click.echo("----------------------------------------------------------------")
    click.echo("🤖 Agent Configuration")
    click.echo("----------------------------------------------------------------")

    # Prompt to run agent configuration
    if click.confirm("Create AI agent configuration files (AGENTS.md)?", default=True):
        ctx = click.get_current_context()
        ctx.invoke(agent_config_command, editor=None)
    else:
        click.echo("Skipping agent configuration. Run 'nexus-agent-config' later to generate it.")

    click.echo("----------------------------------------------------------------")


@click.command("agent-config")
@click.option(
    "--editor",
    type=click.Choice(
        [
            "antigravity",
            "cursor",
            "claude",
            "copilot",
            "windsurf",
            "gemini",
            "zed",
            "vscode",
            "agents",
        ]
    ),
    help="Explicitly specify the editor to configure.",
)
@click.option(
    "--update/--no-update",
    default=False,
    help="Update existing configuration files (merges content where possible).",
)
def agent_config_command(editor: str | None, update: bool) -> None:
    """Create or update AI agent configuration files.

    Generates AGENTS.md and editor-specific configuration files (rules, ignores).
    Supports: Antigravity, Cursor, Claude, Copilot, Windsurf, Gemini, Zed.
    """
    cwd = Path.cwd()
    config_path = cwd / "nexus_config.json"

    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)

    # 1. Detect editor if not specified
    if not editor:
        editor = detect_editor()
        click.echo(f"🔍 Detected editor environment: {editor}")

    # 2. Generate AGENTS.md (Primary Source of Truth)
    agents_md_path = cwd / "AGENTS.md"

    # Path Resolution Fix:
    # __file__ is src/nexus_dev/cli/init.py
    # templates are in src/nexus_dev/templates
    template_dir = Path(__file__).parent.parent / "templates"

    agents_template_path = template_dir / "AGENTS.md"
    if not agents_template_path.exists():
        click.echo("❌ Template AGENTS.md not found in installation.", err=True)
        return

    agents_content = agents_template_path.read_text(encoding="utf-8")

    # Fill variables
    agents_content = agents_content.replace("{project_name}", config.project_name)
    agents_content = agents_content.replace("{project_id}", config.project_id)

    if agents_md_path.exists() and not update:
        click.echo("⚠️  AGENTS.md already exists. Use --update to overwrite/merge.")
    else:
        # For now, simple overwrite on update (could be smarter in future)
        agents_md_path.write_text(agents_content, encoding="utf-8")
        click.echo("✅ Created AGENTS.md")

    # 3. Handle Editor-Specific Files

    if editor == "antigravity":
        # Antigravity: .geminiignore, .antigravityignore, .aiexclude
        _install_antigravity_files(cwd, template_dir, update)

    elif editor == "cursor":
        # Cursor: Symlink .cursorrules -> AGENTS.md
        _create_symlink(cwd / ".cursorrules", agents_md_path, update)

    elif editor == "claude":
        _create_symlink(cwd / "CLAUDE.md", agents_md_path, update)

    elif editor == "windsurf":
        _create_symlink(cwd / ".windsurfrules", agents_md_path, update)

    elif editor == "gemini":
        _create_symlink(cwd / "GEMINI.md", agents_md_path, update)

    elif editor == "zed":
        _create_symlink(cwd / ".rules", agents_md_path, update)

    elif editor == "copilot":
        # Copilot doesn't support symlinks well in .github usually, copy it
        target = cwd / ".github" / "copilot-instructions.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or update:
            shutil.copy(agents_md_path, target)
            click.echo(f"✅ Created {target.relative_to(cwd)}")

    click.echo("")
    click.echo(f"🎉 Agent configuration complete for {editor}!")
