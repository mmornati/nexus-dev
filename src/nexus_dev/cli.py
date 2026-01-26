"""Nexus-Dev CLI commands.

Provides commands for:
- nexus-init: Initialize Nexus-Dev in a project
- nexus-index: Manually index files
- nexus-status: Show project statistics
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
from collections import defaultdict
from collections.abc import Coroutine
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

import click
import yaml

from .chunkers import ChunkerRegistry
from .config import NexusConfig
from .database import Document, DocumentType, NexusDatabase, generate_document_id
from .embeddings import create_embedder, validate_embedding_config
from .github_importer import GitHubImporter
from .mcp_client import MCPClientManager, MCPServerConnection
from .mcp_config import MCPConfig, MCPServerConfig


def _validate_embeddings_or_exit(config: NexusConfig) -> bool:
    """Validate embedding config, print error and return False if invalid.

    Args:
        config: Nexus-Dev configuration.

    Returns:
        True if valid, False if invalid (error message already printed).
    """
    is_valid, error_msg = validate_embedding_config(config)
    if not is_valid:
        click.echo(f"❌ Embedding configuration error: {error_msg}", err=True)
        click.echo(
            "   Configure embedding provider or set required environment variable.",
            err=True,
        )
        return False
    return True


def _find_project_root(start_path: Path | None = None) -> Path | None:
    """Find project root by walking up to find nexus_config.json.

    Args:
        start_path: Starting directory (defaults to cwd)

    Returns:
        Path to project root if found, None otherwise.
    """
    current = (start_path or Path.cwd()).resolve()

    for parent in [current] + list(current.parents):
        if (parent / "nexus_config.json").exists():
            return parent
        if parent == parent.parent:  # Reached filesystem root
            break

    return None


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run async function in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)


@click.group()
@click.version_option(version="0.1.0", prog_name="nexus-dev")
def cli() -> None:
    """Nexus-Dev CLI - Local RAG for AI coding agents.

    Nexus-Dev provides persistent memory for AI coding assistants by indexing
    your code and documentation into a local vector database.
    """


@cli.command("init")
@click.option(
    "--project-name",
    help="Human-readable name for the project",
)
@click.option(
    "--embedding-provider",
    type=click.Choice(["openai", "ollama"]),
    default="openai",
    help="Embedding provider to use (default: openai)",
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

    if embedding_provider == "openai":
        click.echo("")
        click.echo("⚠️  Using OpenAI embeddings. Ensure OPENAI_API_KEY is set.")

    click.echo("")
    click.echo("----------------------------------------------------------------")
    click.echo("🤖 COPY-PASTE THIS INTO YOUR AGENT'S SYSTEM PROMPT OR RULES:")
    click.echo("----------------------------------------------------------------")
    click.echo(f"""
## Nexus-Dev Knowledge Base

You have access to a local RAG system for this project.

**Project ID:** {config.project_id}

**MANDATORY**: You MUST use `nexus-dev` tools BEFORE answering questions about this code.
1. `search_knowledge("{config.project_name} <query>")` - Search code, docs, and lessons
2. `search_code("<class/function_name>")` - Find specific code definitions
3. `search_lessons("<error/problem>")` - Check for past solutions
4. `record_lesson(...)` - Save solutions after fixing non-trivial bugs

**Best Practice:**
- Start every session with `get_project_context()`
- Search before writing code
- Record insights with `record_insight()`
""")
    click.echo("----------------------------------------------------------------")


def _install_hook(git_dir_parent: Path, project_root: Path | None = None) -> None:
    """Install pre-commit hook.

    Args:
        git_dir_parent: Directory containing .git/
        project_root: Optional project root for multi-repo setups.
                     If None, assumes git_dir_parent is the project root.
    """
    git_dir = git_dir_parent / ".git"
    if not git_dir.exists():
        click.echo("⚠️  Not a git repository. Skipping hook installation.")
        return

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    hook_path = hooks_dir / "pre-commit"

    # Check if hook already exists
    if hook_path.exists():
        click.echo("⚠️  pre-commit hook already exists. Skipping.")
        return

    # Copy template
    template_path = Path(__file__).parent / "templates" / "pre-commit-hook"
    if template_path.exists():
        shutil.copy(template_path, hook_path)
    else:
        # Write inline
        hook_content = """#!/bin/bash
# Nexus-Dev Pre-commit Hook

set -e

echo "🧠 Nexus-Dev: Checking for files to index..."

MODIFIED_FILES=$(git diff --cached --name-only --diff-filter=ACM | \
  grep -E '\\.(py|js|jsx|ts|tsx|java)$' || true)

if [ -n "$MODIFIED_FILES" ]; then
    echo "📁 Indexing modified code files..."
    for file in $MODIFIED_FILES; do
        if [ -f "$file" ]; then
            python -m nexus_dev.cli index "$file" --quiet 2>/dev/null || true
        fi
    done
fi

LESSON_FILES=$(git diff --cached --name-only --diff-filter=A | \
  grep -E '^\\.nexus/lessons/.*\\.md$' || true)

if [ -n "$LESSON_FILES" ]; then
    echo "📚 Indexing new lessons..."
    for file in $LESSON_FILES; do
        if [ -f "$file" ]; then
            python -m nexus_dev.cli index-lesson "$file" --quiet 2>/dev/null || true
        fi
    done
fi

echo "✅ Nexus-Dev indexing complete"
"""
        hook_path.write_text(hook_content)

    # Make executable
    current_mode = hook_path.stat().st_mode
    hook_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if project_root and project_root != git_dir_parent:
        click.echo(f"✅ Installed pre-commit hook (linked to {project_root.name}/)")
    else:
        click.echo("✅ Installed pre-commit hook")


def _update_gitignore(cwd: Path, choice: str) -> None:
    """Update .gitignore based on user choice."""
    gitignore_path = cwd / ".gitignore"

    # define mapping for content
    content_map = {
        "allow-lessons": [
            "\n# Nexus-Dev",
            ".nexus_config.json",
            ".nexus/*",
            "!.nexus/lessons/",
            "",
        ],
        "ignore-all": ["\n# Nexus-Dev", ".nexus_config.json", ".nexus/", ""],
    }

    new_lines = content_map.get(choice, [])
    if not new_lines:
        return

    # Create if doesn't exist
    if not gitignore_path.exists():
        gitignore_path.write_text("\n".join(new_lines), encoding="utf-8")
        click.echo("✅ Created .gitignore")
        return

    # Append if exists
    current_content = gitignore_path.read_text(encoding="utf-8")

    # simple check to avoid duplication (imperfect but sufficient for init)
    if ".nexus" in current_content:
        click.echo("⚠️  .nexus already in .gitignore, skipping update.")
        return

    with gitignore_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    click.echo(f"✅ Updated .gitignore ({choice})")


@cli.command("index")
@click.argument("paths", nargs=-1, required=True)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Index directories recursively",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress output",
)
def index_command(paths: tuple[str, ...], recursive: bool, quiet: bool) -> None:
    """Manually index files or directories.

    PATHS can be files or directories. Use -r to recursively index directories.

    Examples:
        nexus-index src/
        nexus-index docs/ -r
        nexus-index main.py utils.py
    """
    # Load config
    config_path = Path.cwd() / "nexus_config.json"
    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)

    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    # Collect files to index
    files_to_index: list[Path] = []

    for path_str in paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            if not quiet:
                click.echo(f"⚠️  Path not found: {path_str}")
            continue

        if path.is_file():
            files_to_index.append(path)
        elif path.is_dir():
            if recursive:
                # Recursively find files using os.walk to prune ignored directories
                for root, dirs, files in os.walk(path):
                    root_path = Path(root)

                    # Compute relative path for pattern matching
                    rel_root = str(root_path.relative_to(Path.cwd()))
                    if rel_root == ".":
                        rel_root = ""

                    # Filter directories to prevent traversal into ignored paths
                    # We must modify dirs in-place to prune the walk
                    i = 0
                    while i < len(dirs):
                        d = dirs[i]
                        d_path = root_path / d
                        # We construct a mock path string for the directory check
                        # (relative path + directory name)
                        check_path = str(d_path.relative_to(Path.cwd()))

                        # Use a simpler check: if the directory ITSELF matches exclude pattern
                        # we should remove it.
                        should_exclude = False

                        # Check excludes for this directory
                        # We treat the directory string as match target for exclude patterns
                        # excluding the trailing slash for fnmatch
                        for pattern in config.exclude_patterns:
                            # Normalize pattern: remove trailing slash for directory matching
                            clean_pat = pattern.rstrip("/")
                            if clean_pat.endswith("/**"):
                                clean_pat = clean_pat[:-3]

                            # Simple fnmatch on the logic
                            if fnmatch(check_path, pattern) or fnmatch(check_path, clean_pat):
                                should_exclude = True
                                break

                            # Handle recursive wildcard start (e.g. **/node_modules)
                            if clean_pat.startswith("**/"):
                                suffix = clean_pat[3:]
                                if (
                                    check_path == suffix
                                    or check_path.endswith("/" + suffix)
                                    or fnmatch(check_path, suffix)
                                ):
                                    should_exclude = True
                                    break

                        if should_exclude:
                            if not quiet:
                                # Optional: debug output if needed, but keeping it clean for now
                                pass
                            del dirs[i]
                        else:
                            i += 1

                    # Add files
                    for file in files:
                        file_path = root_path / file
                        if _should_index(file_path, config):
                            files_to_index.append(file_path)
            else:
                # Only immediate children
                for file_path in path.iterdir():
                    if file_path.is_file():
                        # For explicit paths/directories, we check excludes but ignore
                        # include patterns to allow indexing "anything I point at"
                        # unless specifically excluded
                        is_excluded = _is_excluded(file_path, config)
                        if not is_excluded:
                            files_to_index.append(file_path)

    if not files_to_index:
        if not quiet:
            click.echo("No files to index.")
        return

    if not quiet:
        _print_file_summary(files_to_index)
        if not click.confirm("Proceed with indexing?"):
            click.echo("Aborted.")
            return

    # Index files
    total_chunks = 0
    errors = 0

    for file_path in files_to_index:
        try:
            # Read file
            content = file_path.read_text(encoding="utf-8")

            # Detect smart type from frontmatter
            detected_type, metadata = _detect_document_type_and_metadata(content)

            # Determine type
            if detected_type:
                doc_type = detected_type
            else:
                ext = file_path.suffix.lower()
                doc_type = (
                    DocumentType.DOCUMENTATION
                    if ext in (".md", ".markdown", ".rst", ".txt")
                    else DocumentType.CODE
                )

            # Delete existing
            _run_async(database.delete_by_file(str(file_path), config.project_id))

            # Chunk file
            chunks = ChunkerRegistry.chunk_file(file_path, content)

            if chunks:
                # Generate embeddings and store
                chunk_count = _run_async(
                    _index_chunks_sync(
                        chunks,
                        config.project_id,
                        doc_type,
                        embedder,
                        database,
                        metadata=metadata,
                    )
                )
                total_chunks += chunk_count

                if not quiet:
                    click.echo(f"  ✅ {file_path.name}: {chunk_count} chunks")

        except Exception as e:
            errors += 1
            if not quiet:
                click.echo(f"  ❌ {file_path.name}: {e!s}")

    if not quiet:
        click.echo("")
        click.echo(f"✅ Indexed {total_chunks} chunks from {len(files_to_index) - errors} files")
        if errors:
            click.echo(f"⚠️  {errors} file(s) failed")


async def _index_chunks_sync(
    chunks: list[Any],
    project_id: str,
    doc_type: DocumentType,
    embedder: Any,
    database: NexusDatabase,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Index chunks synchronously."""
    if not chunks:
        return 0

    texts = [chunk.get_searchable_text() for chunk in chunks]
    embeddings = await embedder.embed_batch(texts)

    documents = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        doc_id = generate_document_id(
            project_id,
            chunk.file_path,
            chunk.name,
            chunk.start_line,
        )

        # Prepare document kwargs
        doc_kwargs = {
            "id": doc_id,
            "text": chunk.get_searchable_text(),
            "vector": embedding,
            "project_id": project_id,
            "file_path": chunk.file_path,
            "doc_type": doc_type,
            "chunk_type": chunk.chunk_type.value,
            "language": chunk.language,
            "name": chunk.name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }

        # Add metadata if present
        if metadata and "timestamp" in metadata:
            try:
                # Handle ISO format from export
                if isinstance(metadata["timestamp"], str):
                    doc_kwargs["timestamp"] = datetime.fromisoformat(metadata["timestamp"])
                elif isinstance(metadata["timestamp"], datetime):
                    doc_kwargs["timestamp"] = metadata["timestamp"]
            except Exception:
                # Fallback to current time if parse fails
                pass

        doc = Document(**doc_kwargs)
        documents.append(doc)

    await database.upsert_documents(documents)
    return len(documents)


def _should_index(file_path: Path, config: NexusConfig) -> bool:
    """Check if file should be indexed based on config patterns."""
    rel_path = str(file_path.relative_to(Path.cwd()))

    # Check exclude patterns
    for pattern in config.exclude_patterns:
        if fnmatch(rel_path, pattern):
            return False

        # Also check without leading **/ if present (for root matches)
        if pattern.startswith("**/") and fnmatch(rel_path, pattern[3:]):
            return False

    # Check include patterns
    for pattern in config.include_patterns:
        if fnmatch(rel_path, pattern):
            return True

    # Also include docs folders
    for docs_folder in config.docs_folders:
        if rel_path.startswith(docs_folder) or rel_path == docs_folder.rstrip("/"):
            return True

    return False


def _is_excluded(file_path: Path, config: NexusConfig) -> bool:
    """Check if file is explicitly excluded by config patterns."""
    rel_path = str(file_path.relative_to(Path.cwd()))

    # Check exclude patterns
    for pattern in config.exclude_patterns:
        if fnmatch(rel_path, pattern):
            return True

        # Also check without leading **/ if present (for root matches)
        if pattern.startswith("**/") and fnmatch(rel_path, pattern[3:]):
            return True

    return False


def _detect_document_type_and_metadata(
    content: str,
) -> tuple[DocumentType | None, dict[str, Any]]:
    """Detect document type and metadata from frontmatter."""
    if not content.startswith("---\n"):
        return None, {}

    try:
        # Extract frontmatter
        _, frontmatter, _ = content.split("---", 2)
        data = yaml.safe_load(frontmatter)

        if not isinstance(data, dict):
            return None, {}

        # Detect type based on keys/values
        if data.get("category") in ["discovery", "mistake", "backtrack", "optimization"]:
            return DocumentType.INSIGHT, data

        if "problem" in data and "solution" in data:
            return DocumentType.LESSON, data

        if (
            "summary" in data
            and "approach" in data
            and ("files_changed" in data or "design_decisions" in data)
        ):
            return DocumentType.IMPLEMENTATION, data

        if data.get("type") == "github_issue":
            return DocumentType.GITHUB_ISSUE, data

        if data.get("type") == "github_pr":
            return DocumentType.GITHUB_PR, data

        return None, data

    except Exception:
        return None, {}


def _print_file_summary(files: list[Path]) -> None:
    """Print a summary of files to be indexed."""
    if not files:
        click.echo("No files to index.")
        return

    # Group by directory
    by_dir: dict[str, int] = defaultdict(int)
    for f in files:
        parent = str(f.parent.relative_to(Path.cwd()) if f.is_absolute() else f.parent)
        if parent == ".":
            parent = "Root"
        by_dir[parent] += 1

    click.echo(f"  Found {len(files)} files to index:")
    click.echo("")

    # Sort by directory name
    for directory, count in sorted(by_dir.items()):
        click.echo(f"  📁 {directory:<40} {count} files")

    click.echo("")


@cli.command("index-lesson")
@click.argument("lesson_file")
@click.option("-q", "--quiet", is_flag=True, help="Suppress output")
def index_lesson_command(lesson_file: str, quiet: bool) -> None:
    """Index a lesson file from .nexus/lessons/."""
    path = Path(lesson_file)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        if not quiet:
            click.echo(f"❌ Lesson file not found: {lesson_file}", err=True)
        return

    # Load config
    config_path = Path.cwd() / "nexus_config.json"
    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)

    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    try:
        content = path.read_text(encoding="utf-8")

        # Generate embedding
        embedding = _run_async(embedder.embed(content))

        # Create document
        doc_id = generate_document_id(
            config.project_id,
            str(path),
            path.stem,
            0,
        )

        doc = Document(
            id=doc_id,
            text=content,
            vector=embedding,
            project_id=config.project_id,
            file_path=str(path),
            doc_type=DocumentType.LESSON,
            chunk_type="lesson",
            language="markdown",
            name=path.stem,
            start_line=0,
            end_line=0,
        )

        _run_async(database.upsert_document(doc))

        if not quiet:
            click.echo(f"✅ Indexed lesson: {path.name}")

    except Exception as e:
        if not quiet:
            click.echo(f"❌ Failed to index lesson: {e!s}", err=True)


@cli.command("export")
@click.option("--project-id", help="Project ID to export (defaults to current config)")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: ./nexus-export)",
)
def export_command(project_id: str | None, output: Path | None) -> None:
    """Export project knowledge to markdown files."""
    from .config import NexusConfig
    from .database import DocumentType, NexusDatabase
    from .embeddings import create_embedder

    async def _export() -> None:
        # Load config
        config = None
        try:
            config_path = Path.cwd() / "nexus_config.json"
            if config_path.exists():
                config = NexusConfig.load(config_path)
        except Exception:
            pass

        effective_project_id = project_id
        if not effective_project_id and config:
            effective_project_id = config.project_id

        if not effective_project_id:
            click.secho("Error: No project-id provided and no nexus_config.json found.", fg="red")
            return

        # Initialize DB
        if not config:
            # Create temporary config for DB access
            config = NexusConfig.create_new("temp")

        try:
            # Validate embedding configuration before proceeding
            if not _validate_embeddings_or_exit(config):
                return

            embedder = create_embedder(config)
            db = NexusDatabase(config, embedder)
            db.connect()

            click.echo(f"Exporting knowledge for project: {effective_project_id}")

            # Get all documents for this project
            # searching with empty query returns all items for project/type
            # We fetch strictly structured data types: Lesson, Insight, Implementation
            types_to_export = [
                (DocumentType.LESSON, "lessons"),
                (DocumentType.INSIGHT, "insights"),
                (DocumentType.IMPLEMENTATION, "implementations"),
            ]

            base_dir = output or Path.cwd() / "nexus-export"
            base_dir.mkdir(parents=True, exist_ok=True)

            total_count = 0

            for doc_type, dirname in types_to_export:
                # We use a hack: search for " " (space) which usually matches everything
                # or rely on search implementation to support wildcards.
                # Since vector search always returns something, we use a high limit
                results = await db.search(
                    query="*",  # Some vector DBs verify query length
                    project_id=effective_project_id,
                    doc_type=doc_type,
                    limit=1000,
                )

                if not results:
                    continue

                type_dir = base_dir / dirname
                type_dir.mkdir(exist_ok=True)

                click.echo(f"  - Found {len(results)} {dirname}")

                for res in results:
                    # Use ID from metadata if available, else generate safe name
                    safe_name = "".join(c for c in res.name if c.isalnum() or c in "-_")
                    filename = f"{safe_name}.md"

                    file_path = type_dir / filename
                    file_path.write_text(res.text, encoding="utf-8")
                    total_count += 1

            click.secho(f"\nSuccessfully exported {total_count} files to {base_dir}", fg="green")

        except Exception as e:
            click.secho(f"Export failed: {e}", fg="red")

    _run_async(_export())


@cli.command("status")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed debug information")
def status_command(verbose: bool) -> None:
    """Show Nexus-Dev status and statistics."""
    config_path = Path.cwd() / "nexus_config.json"

    if not config_path.exists():
        click.echo("❌ Nexus-Dev not initialized in this directory.")
        click.echo("   Run 'nexus-init' to get started.")
        return

    config = NexusConfig.load(config_path)

    click.echo("📊 Nexus-Dev Status")
    click.echo("")
    click.echo(f"Project: {config.project_name}")
    click.echo(f"Project ID: {config.project_id}")
    click.echo(f"Embedding Provider: {config.embedding_provider}")
    click.echo(f"Embedding Model: {config.embedding_model}")
    click.echo(f"Database: {config.get_db_path()}")
    click.echo("")

    try:
        # Validate embedding configuration before proceeding
        if not _validate_embeddings_or_exit(config):
            return

        embedder = create_embedder(config)
        database = NexusDatabase(config, embedder)
        database.connect()

        if verbose:
            click.echo("🔍 Debug Info:")
            click.echo(f"   Database path exists: {config.get_db_path().exists()}")
            click.echo(f"   Querying for project_id: {config.project_id}")
            click.echo("")

        stats = _run_async(database.get_project_stats(config.project_id))

        click.echo("📈 Statistics:")
        click.echo(f"   Total chunks: {stats.get('total', 0)}")
        click.echo(f"   Code: {stats.get('code', 0)}")
        click.echo(f"   Documentation: {stats.get('documentation', 0)}")
        click.echo(f"   Lessons: {stats.get('lesson', 0)}")

        if verbose and stats.get("total", 0) > 0:
            click.echo("")
            click.echo("   Document type breakdown:")
            for doc_type, count in stats.items():
                if doc_type != "total":
                    click.echo(f"     - {doc_type}: {count}")

    except Exception as e:
        click.echo(f"⚠️  Could not connect to database: {e!s}")
        if verbose:
            import traceback

            click.echo("")
            click.echo("Full traceback:")
            click.echo(traceback.format_exc())


@cli.command("inspect")
@click.option("--project-id", help="Filter by project ID (default: current project)")
@click.option("--limit", default=5, help="Number of sample documents to show")
@click.option("--all-projects", is_flag=True, help="Show all projects in database")
def inspect_command(project_id: str | None, limit: int, all_projects: bool) -> None:
    """Inspect database contents for debugging."""
    # Load config for default project_id
    config = None
    if not all_projects and not project_id:
        config_path = Path.cwd() / "nexus_config.json"
        if config_path.exists():
            config = NexusConfig.load(config_path)
            project_id = config.project_id

    # Get database path from config or use default
    if config:
        # Validate embedding configuration before proceeding
        if not _validate_embeddings_or_exit(config):
            return

        embedder = create_embedder(config)
        database = NexusDatabase(config, embedder)
    else:
        # Use default config to access shared database
        default_config = NexusConfig.create_new("temp")
        # Validate embedding configuration before proceeding
        if not _validate_embeddings_or_exit(default_config):
            return

        embedder = create_embedder(default_config)
        database = NexusDatabase(default_config, embedder)

    database.connect()

    click.echo("🔍 Nexus-Dev Database Inspection")
    click.echo("")

    try:
        # Get database info
        db_path = database.config.get_db_path()
        click.echo(f"Database location: {db_path}")

        if db_path.exists():
            # Calculate database size
            total_size = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
            click.echo(f"Database size: {total_size / 1024 / 1024:.2f} MB")
        click.echo("")

        # Get all project statistics
        all_stats = _run_async(database.get_project_stats(None))
        click.echo(f"📊 Total documents across all projects: {all_stats.get('total', 0)}")
        click.echo("")

        # Get table and show project breakdown
        table = database._ensure_connected()
        df = table.to_pandas()

        if len(df) == 0:
            click.echo("⚠️  Database is empty")
            return

        # Group by project
        project_counts = df.groupby("project_id").size().sort_values(ascending=False)

        click.echo("📁 Projects in database:")
        for pid, count in project_counts.items():
            marker = "👉" if pid == project_id else "  "
            click.echo(f"{marker} {pid}: {count} chunks")
        click.echo("")

        # Show document type statistics for specific project or all
        if project_id:
            project_df = df[df["project_id"] == project_id]
            if len(project_df) == 0:
                click.echo(f"⚠️  No documents found for project: {project_id}")
                return

            click.echo(f"📈 Document types for project {project_id}:")
            type_counts = project_df.groupby("doc_type").size()
            for doc_type, count in type_counts.items():
                click.echo(f"   {doc_type}: {count}")
            click.echo("")

            # Show sample documents
            click.echo(f"📄 Sample documents (limit: {limit}):")
            samples = project_df.head(limit)
            for _idx, row in samples.iterrows():
                click.echo(f"   - [{row['doc_type']}] {row['name']}")
                click.echo(f"     File: {row['file_path']}")
                if row["start_line"] > 0:
                    click.echo(f"     Lines: {row['start_line']}-{row['end_line']}")
                click.echo("")
        else:
            # Show overall document type breakdown
            click.echo("📈 Document type breakdown (all projects):")
            type_counts = df.groupby("doc_type").size()
            for doc_type, count in type_counts.items():
                click.echo(f"   {doc_type}: {count}")

    except Exception as e:
        click.echo(f"❌ Error inspecting database: {e!s}", err=True)
        import traceback

        click.echo(traceback.format_exc(), err=True)


@cli.command("clean")
@click.option("--project-id", help="Project ID to clean (default: current project)")
@click.option("--all", "clean_all", is_flag=True, help="Delete ALL projects (dangerous!)")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
def clean_command(project_id: str | None, clean_all: bool, dry_run: bool) -> None:
    """Delete indexed data for a project."""
    # Validate options
    if clean_all and project_id:
        click.echo("❌ Cannot use both --all and --project-id", err=True)
        return

    if not clean_all and not project_id:
        # Try to get project_id from current directory
        config_path = Path.cwd() / "nexus_config.json"
        if config_path.exists():
            config = NexusConfig.load(config_path)
            project_id = config.project_id
        else:
            click.echo("❌ No project-id specified and no nexus_config.json found", err=True)
            click.echo("   Use --project-id or run from a project directory", err=True)
            return

    # Load database
    config_path = Path.cwd() / "nexus_config.json"
    if config_path.exists():
        config = NexusConfig.load(config_path)
    else:
        config = NexusConfig.create_new("temp")

    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    try:
        if clean_all:
            # Get total count
            stats = _run_async(database.get_project_stats(None))
            total = stats.get("total", 0)

            if total == 0:
                click.echo("⚠️  Database is already empty")
                return

            click.echo(f"⚠️  WARNING: This will delete ALL {total} documents from the database!")
            click.echo("")

            if dry_run:
                click.echo("[DRY RUN] Would delete entire database")
                return

            if not click.confirm("Are you absolutely sure?"):
                click.echo("Aborted.")
                return

            # Delete all by resetting
            database.reset()
            click.echo(f"✅ Deleted all {total} documents")

        else:
            # Delete specific project
            stats = _run_async(database.get_project_stats(project_id))
            count = stats.get("total", 0)

            if count == 0:
                click.echo(f"⚠️  No documents found for project: {project_id}")
                return

            click.echo(f"Found {count} documents for project: {project_id}")
            click.echo("")
            click.echo("Document types:")
            for doc_type, type_count in stats.items():
                if doc_type != "total":
                    click.echo(f"  - {doc_type}: {type_count}")
            click.echo("")

            if dry_run:
                click.echo(f"[DRY RUN] Would delete {count} documents for project {project_id}")
                return

            if not click.confirm(f"Delete {count} documents?"):
                click.echo("Aborted.")
                return

            # project_id is guaranteed to be set by validation logic above
            assert project_id is not None
            deleted = _run_async(database.delete_by_project(project_id))
            click.echo(f"✅ Deleted {deleted} documents for project {project_id}")

    except Exception as e:
        click.echo(f"❌ Error during cleanup: {e!s}", err=True)


@cli.command("reindex")
def reindex_command() -> None:
    """Re-index entire project (clear and rebuild)."""
    config_path = Path.cwd() / "nexus_config.json"

    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)

    # Collect files first to show summary
    click.echo("🔍 Scanning files...")

    cwd = Path.cwd()
    files_to_index: list[Path] = []

    for pattern in config.include_patterns:
        for file_path in cwd.glob(pattern):
            if file_path.is_file() and _should_index(file_path, config):
                files_to_index.append(file_path)

    # Also index docs folders
    for docs_folder in config.docs_folders:
        docs_path = cwd / docs_folder
        if docs_path.is_file():
            files_to_index.append(docs_path)
        elif docs_path.is_dir():
            for root, _, files in os.walk(docs_path):
                # Apply same pruning logic for docs if needed, though usually docs are safer
                # For consistency let's just collect files
                for file in files:
                    files_to_index.append(Path(root) / file)

    # Remove duplicates
    files_to_index = list(set(files_to_index))

    # Show summary and ask for confirmation
    _print_file_summary(files_to_index)

    if not click.confirm("This will CLEAR the database and re-index the above files. Continue?"):
        click.echo("Aborted.")
        return

    # Proceed with DB operations
    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    click.echo("🗑️  Clearing existing index...")
    # Reset database to handle schema changes
    database.reset()
    # Re-connect to create new table with updated schema
    database.connect()
    click.echo("   Index cleared and schema updated")

    click.echo("")
    click.echo("📁 Re-indexing project...")

    # Index all files
    total_chunks = 0
    for file_path in files_to_index:
        try:
            content = file_path.read_text(encoding="utf-8")
            ext = file_path.suffix.lower()
            doc_type = (
                DocumentType.DOCUMENTATION
                if ext in (".md", ".markdown", ".rst", ".txt")
                else DocumentType.CODE
            )

            chunks = ChunkerRegistry.chunk_file(file_path, content)
            if chunks:
                count = _run_async(
                    _index_chunks_sync(chunks, config.project_id, doc_type, embedder, database)
                )
                total_chunks += count
                click.echo(f"  ✅ {file_path.name}: {count} chunks")

        except Exception as e:
            click.echo(f"  ❌ Failed to index {file_path.name}: {e!s}", err=True)

    click.echo("")
    click.echo(f"✅ Re-indexed {total_chunks} chunks from {len(files_to_index)} files")


@cli.command("import-github")
@click.option("--repo", required=True, help="Repository name")
@click.option("--owner", required=True, help="Repository owner")
@click.option("--limit", default=20, help="Maximum number of issues to import")
@click.option("--state", default="all", help="Issue state (open, closed, all)")
def import_github_command(repo: str, owner: str, limit: int, state: str) -> None:
    """Import GitHub issues and PRs."""
    # Load config
    config_path = Path.cwd() / "nexus_config.json"
    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)

    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    database.connect()

    # Load MCP config handled by manager inside generally, but here we can rely on standard init
    client_manager = MCPClientManager()

    # Load MCP config
    mcp_config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    mcp_config = None
    if mcp_config_path.exists():
        try:
            mcp_config = MCPConfig.load(mcp_config_path)
        except Exception as e:
            click.echo(f"⚠️  Failed to load MCP config: {e}", err=True)

    if not mcp_config:
        click.echo("⚠️  No MCP config found. GitHub import may fail if server not found.")

    importer = GitHubImporter(database, config.project_id, client_manager, mcp_config)

    click.echo(f"📥 Importing issues from {owner}/{repo}...")

    try:
        count = _run_async(importer.import_issues(owner, repo, limit, state))
        click.echo(f"✅ Imported {count} issues/PRs")
    except Exception as e:
        click.echo(f"❌ Import failed: {e}", err=True)


@cli.command("search")
@click.argument("query")
@click.option("--type", "content_type", help="Content type to filter by")
@click.option("--limit", default=5, help="Number of results")
def search_command(query: str, content_type: str | None, limit: int) -> None:
    """Search the knowledge base."""
    # Load config
    config_path = Path.cwd() / "nexus_config.json"
    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)

    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    click.echo(f"🔍 Searching for '{query}'...")

    doc_type_enum = None
    if content_type:
        try:
            doc_type_enum = DocumentType(content_type)
        except ValueError:
            click.echo(f"⚠️  Invalid type '{content_type}'. Ignoring filter.")

    results = _run_async(database.search(query, limit=limit, doc_type=doc_type_enum))

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(results)} results:\n")

    for i, doc in enumerate(results, 1):
        click.echo(f"{i}. [{doc.doc_type.upper()}] {doc.name} (Score: {doc.score:.3f})")
        click.echo(f"   path: {doc.file_path}")
        # Preview text
        text = doc.text.replace("\n", " ").strip()
        if len(text) > 100:
            text = text[:97] + "..."
        click.echo(f'   "{text}"')
        click.echo("")


@cli.command("index-mcp")
@click.option("--server", "-s", help="Server name to index (from MCP config)")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to MCP config file (default: ~/.config/mcp/config.json)",
)
@click.option("--all", "-a", "index_all", is_flag=True, help="Index all configured servers")
def index_mcp_command(server: str | None, config: str | None, index_all: bool) -> None:
    """Index MCP tool documentation into the knowledge base.

    This command reads tool schemas from MCP servers and indexes them
    for semantic search via the search_tools command.

    Examples:
        nexus-index-mcp --server github
        nexus-index-mcp --all
        nexus-index-mcp --config ~/my-mcp-config.json --all
    """
    # Load MCP config
    mcp_config_data: dict[str, Any] | MCPConfig
    if config:
        config_path = Path(config)
    else:
        # Prioritize local project config
        local_config_path = Path.cwd() / ".nexus" / "mcp_config.json"
        if local_config_path.exists():
            config_path = local_config_path
        else:
            config_path = Path.home() / ".config" / "mcp" / "config.json"

    if not config_path.exists():
        click.echo(f"MCP config not found: {config_path}")
        click.echo("Specify --config or create ~/.config/mcp/config.json")
        return

    try:
        if config_path.name == "mcp_config.json" and config_path.parent.name == ".nexus":
            # Project-specific config
            mcp_config_data = MCPConfig.load(config_path)
        else:
            # Global config (or custom dict-based config)
            mcp_config_data = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON in MCP config: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"❌ Failed to load MCP config: {e}", err=True)
        return

    # Determine which servers to index
    servers_to_index = []
    if isinstance(mcp_config_data, MCPConfig):
        all_servers = list(mcp_config_data.servers.keys())
    else:
        all_servers = list(mcp_config_data.get("mcpServers", {}).keys())

    if index_all:
        servers_to_index = all_servers
    elif server:
        servers_to_index = [server]
    else:
        click.echo("Specify --server or --all")
        return

    # Index each server
    asyncio.run(_index_mcp_servers(mcp_config_data, servers_to_index))


async def _index_mcp_servers(
    mcp_config: dict[str, Any] | MCPConfig, server_names: list[str]
) -> None:
    """Index tools from specified MCP servers."""
    # Load config
    config_path = Path.cwd() / "nexus_config.json"
    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)
    client = MCPClientManager()

    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    for name in server_names:
        if isinstance(mcp_config, MCPConfig):
            server_config = mcp_config.servers.get(name)
            if not server_config:
                click.echo(f"Server not found: {name}")
                continue
            # Convert to internal connection format
            connection = MCPServerConnection(
                name=name,
                command=server_config.command or "",
                args=server_config.args,
                env=server_config.env,
                transport=server_config.transport,
                url=server_config.url,
                headers=server_config.headers,
                timeout=server_config.timeout,
            )
        else:
            server_dict = mcp_config.get("mcpServers", {}).get(name)
            if not server_dict:
                click.echo(f"Server not found: {name}")
                continue
            connection = MCPServerConnection(
                name=name,
                command=server_dict.get("command", ""),
                args=server_dict.get("args", []),
                env=server_dict.get("env"),
                transport=server_dict.get("transport", "stdio"),
                url=server_dict.get("url"),
                headers=server_dict.get("headers"),
                timeout=server_dict.get("timeout", 30.0),
            )

        # Connect and index

        click.echo(f"Indexing tools from: {name}")

        try:
            tools = await client.get_tools(connection)
            click.echo(f"  Found {len(tools)} tools")

            # Create documents and index
            for tool in tools:
                text = f"{name}.{tool.name}: {tool.description}"
                vector = await embedder.embed(text)

                doc = Document(
                    id=f"{name}:{tool.name}",
                    text=text,
                    vector=vector,
                    project_id=f"{config.project_id}_mcp_tools",
                    file_path=f"mcp://{name}/{tool.name}",
                    doc_type=DocumentType.TOOL,
                    chunk_type="tool",
                    language="mcp",
                    name=tool.name,
                    start_line=0,
                    end_line=0,
                    server_name=name,
                    parameters_schema=json.dumps(tool.input_schema),
                )

                await database.upsert_document(doc)

            click.echo(f"  ✅ Indexed {len(tools)} tools from {name}")

        except Exception as e:
            # Handle ExceptionGroup from anyio/TaskGroup
            if hasattr(e, "exceptions"):
                for sub_e in e.exceptions:
                    click.echo(f"  ❌ Failed to index {name}: {sub_e}")
            else:
                click.echo(f"  ❌ Failed to index {name}: {e}")

    click.echo("Done!")


@cli.group("mcp")
def mcp_group() -> None:
    """Manage MCP server configurations."""


@mcp_group.command("init")
@click.option(
    "--from-global",
    is_flag=True,
    help="Import servers from ~/.config/mcp/config.json",
)
def mcp_init_command(from_global: bool) -> None:
    """Initialize MCP configuration for this project.

    Creates .nexus/mcp_config.json with an empty configuration
    or imports from your global MCP config.

    Examples:
        nexus-mcp init
        nexus-mcp init --from-global
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"

    if config_path.exists() and not click.confirm("MCP config exists. Overwrite?"):
        click.echo("Aborted.")
        return

    # Ensure .nexus directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if from_global:
        # Import from global config
        global_path = Path.home() / ".config" / "mcp" / "config.json"
        if not global_path.exists():
            click.echo(f"Global config not found: {global_path}")
            return

        try:
            global_config = json.loads(global_path.read_text())
        except json.JSONDecodeError as e:
            click.echo(f"❌ Invalid JSON in global config: {e}")
            return

        servers = {}

        for name, cfg in global_config.get("mcpServers", {}).items():
            servers[name] = MCPServerConfig(
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                enabled=True,
            )

        mcp_config = MCPConfig(
            version="1.0",
            servers=servers,
            profiles={},
        )
        click.echo(f"Imported {len(servers)} servers from global config")
    else:
        # Create empty config
        mcp_config = MCPConfig(
            version="1.0",
            servers={},
            profiles={"default": []},
        )

    mcp_config.save(config_path)
    click.echo(f"✅ Created {config_path}")
    click.echo("")
    click.echo("Configuration initialized successfully!")
    click.echo("You can manually edit the config file to add MCP servers.")


@mcp_group.command("add")
@click.argument("name")
@click.option("--command", "-c", required=True, help="Command to run MCP server")
@click.option("--args", "-a", multiple=True, help="Arguments for the command")
@click.option("--env", "-e", multiple=True, help="Environment vars (KEY=value or KEY=${VAR})")
@click.option("--profile", "-p", default="default", help="Add to profile (default: default)")
def mcp_add_command(
    name: str, command: str, args: tuple[str, ...], env: tuple[str, ...], profile: str
) -> None:
    """Add an MCP server to the configuration.

    Examples:
        nexus-mcp add github --command "npx" --args "-y" \\
            --args "@modelcontextprotocol/server-github"
        nexus-mcp add myserver --command "my-mcp" --env "API_KEY=${MY_API_KEY}"
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    # Parse environment variables
    env_dict = {}
    for e in env:
        if "=" in e:
            k, v = e.split("=", 1)
            env_dict[k] = v

    # Add server
    mcp_config.servers[name] = MCPServerConfig(
        command=command,
        args=list(args),
        env=env_dict,
        enabled=True,
    )

    # Add to profile
    if profile not in mcp_config.profiles:
        mcp_config.profiles[profile] = []
    if name not in mcp_config.profiles[profile]:
        mcp_config.profiles[profile].append(name)

    mcp_config.save(config_path)
    click.echo(f"Added {name} to profile '{profile}'")


@mcp_group.command("list")
@click.option(
    "--all", "-a", "show_all", is_flag=True, help="Show all servers, not just active profile"
)
def mcp_list_command(show_all: bool) -> None:
    """List configured MCP servers.

    Examples:
        nexus-mcp list
        nexus-mcp list --all
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("No MCP config. Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    click.echo(f"Active profile: {mcp_config.active_profile}")
    click.echo("")

    if show_all:
        click.echo("All servers:")
        servers_to_show = list(mcp_config.servers.items())
    else:
        click.echo("Active servers:")
        # Get active profile server names
        if mcp_config.active_profile in mcp_config.profiles:
            active_server_names = mcp_config.profiles[mcp_config.active_profile]
            # Filter to only enabled servers
            servers_to_show = [
                (name, mcp_config.servers[name])
                for name in active_server_names
                if name in mcp_config.servers and mcp_config.servers[name].enabled
            ]
        else:
            # If no active profile, show all enabled servers
            servers_to_show = [
                (name, server) for name, server in mcp_config.servers.items() if server.enabled
            ]

    for name, server in servers_to_show:
        status = "✓" if server.enabled else "✗"
        click.echo(f"  {status} {name}")
        click.echo(f"    Command: {server.command} {' '.join(server.args)}")
        if server.env:
            click.echo(f"    Env: {', '.join(server.env.keys())}")

    click.echo("")
    click.echo(f"Profiles: {', '.join(mcp_config.profiles.keys())}")


@mcp_group.command("profile")
@click.argument("name", required=False)
@click.option("--add", "-a", multiple=True, help="Add server to profile")
@click.option("--remove", "-r", multiple=True, help="Remove server from profile")
@click.option("--create", is_flag=True, help="Create new profile")
def mcp_profile_command(
    name: str | None, add: tuple[str, ...], remove: tuple[str, ...], create: bool
) -> None:
    """Manage MCP profiles.

    Without arguments, shows current profile. With name, switches to that profile.

    Examples:
        nexus-mcp profile              # Show current
        nexus-mcp profile dev          # Switch to 'dev'
        nexus-mcp profile dev --create # Create new 'dev' profile
        nexus-mcp profile default --add homeassistant
        nexus-mcp profile default --remove github
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    if not name:
        # Show current profile
        click.echo(f"Active: {mcp_config.active_profile}")
        servers = mcp_config.profiles.get(mcp_config.active_profile, [])
        click.echo(f"Servers: {', '.join(servers) or '(none)'}")
        return

    if create:
        if name in mcp_config.profiles:
            click.echo(f"Profile '{name}' exists")
            return
        mcp_config.profiles[name] = []
        click.echo(f"Created profile: {name}")

    if name not in mcp_config.profiles:
        click.echo(f"Profile '{name}' not found")
        return

    # Add servers
    for server in add:
        if server not in mcp_config.profiles[name]:
            mcp_config.profiles[name].append(server)
            click.echo(f"Added {server} to {name}")
            # Warn if server not defined yet
            if server not in mcp_config.servers:
                click.echo(f"  ⚠️  Server '{server}' not defined. Add it with 'nexus-mcp add'")
        else:
            click.echo(f"Server {server} already in {name}")

    # Remove servers
    for server in remove:
        if server in mcp_config.profiles[name]:
            mcp_config.profiles[name].remove(server)
            click.echo(f"Removed {server} from {name}")

    # Switch profile
    if not add and not remove and not create:
        mcp_config.active_profile = name
        click.echo(f"Switched to profile: {name}")

    mcp_config.save(config_path)


def _set_server_enabled(name: str, enabled: bool) -> None:
    """Set server enabled status."""
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    if name not in mcp_config.servers:
        click.echo(f"Server not found: {name}")
        return

    mcp_config.servers[name].enabled = enabled
    mcp_config.save(config_path)

    status = "enabled" if enabled else "disabled"
    click.echo(f"{name}: {status}")


@mcp_group.command("enable")
@click.argument("name")
def mcp_enable_command(name: str) -> None:
    """Enable an MCP server."""
    _set_server_enabled(name, True)


@mcp_group.command("disable")
@click.argument("name")
def mcp_disable_command(name: str) -> None:
    """Disable an MCP server."""
    _set_server_enabled(name, False)


# Agent management commands
@cli.group("agent")
def agent_group() -> None:
    """Manage custom agents."""


@agent_group.command("templates")
def agent_templates_command() -> None:
    """List available agent templates."""
    from .agent_templates import list_templates

    templates = list_templates()

    if not templates:
        click.echo("No templates found.")
        return

    click.echo("📋 Available Agent Templates:")
    click.echo("")

    # Load and display each template
    import yaml

    from .agent_templates import get_template_path

    for template_name in sorted(templates):
        try:
            template_path = get_template_path(template_name)
            with open(template_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            display_name = data.get("display_name", template_name)
            role = data.get("profile", {}).get("role", "Unknown")
            model = data.get("llm_config", {}).get("model_hint", "auto")

            click.echo(f"  • {display_name} ({template_name})")
            click.echo(f"    Role: {role}")
            click.echo(f"    Model: {model}")
            click.echo("")
        except Exception as e:
            click.echo(f"  ⚠️  {template_name}: Failed to load ({e})")


@agent_group.command("init")
@click.argument("name")
@click.option("--from-template", "-t", "template_name", help="Create from template")
@click.option("--model", "-m", "custom_model", help="Override template model")
@click.option("--role", prompt=False, default=None)
@click.option("--goal", prompt=False, default=None)
@click.option(
    "--backstory",
    prompt=False,
    default=None,
)
def agent_init_command(
    name: str,
    template_name: str | None,
    custom_model: str | None,
    role: str | None,
    goal: str | None,
    backstory: str | None,
) -> None:
    """Create a new agent configuration.

    NAME is the agent identifier (lowercase with underscores).

    Examples:
        nexus-agent init my_reviewer --from-template code_reviewer
        nexus-agent init security_check -t security_auditor --model claude-opus-4.5
        nexus-agent init my_custom_agent
    """
    import re

    import yaml

    from .agent_templates import get_template_path, list_templates

    agents_dir = Path.cwd() / "agents"
    agents_dir.mkdir(exist_ok=True)

    # Normalize name
    agent_name = name.lower().replace(" ", "_").replace("-", "_")

    # Validate name format
    if not re.match(r"^[a-z][a-z0-9_]*$", agent_name):
        click.echo(f"❌ Invalid agent name: {agent_name}", err=True)
        click.echo(
            "   Name must start with a letter and contain only lowercase letters, "
            "numbers, and underscores."
        )
        return

    # Load from template if specified
    if template_name:
        available_templates = list_templates()
        if template_name not in available_templates:
            click.echo(f"❌ Template '{template_name}' not found.", err=True)
            click.echo(f"   Available templates: {', '.join(available_templates)}")
            return

        template_path = get_template_path(template_name)
        with open(template_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Customize the template
        config["name"] = agent_name
        config["display_name"] = name.replace("_", " ").title()
        config["description"] = f"Delegate tasks to the {name.replace('_', ' ').title()} agent."

        # Override model if specified
        if custom_model:
            config["llm_config"]["model_hint"] = custom_model

        click.echo(f"✅ Created agent from template: {template_name}")
    else:
        # Interactive mode
        if not role:
            role = click.prompt("Agent role (e.g., 'Code Reviewer')")
        if not goal:
            goal = click.prompt("Agent goal (e.g., 'Review code for best practices')")
        if not backstory:
            backstory = click.prompt(
                "Agent backstory", default="Expert developer with years of experience."
            )

        # Generate YAML content
        config = {
            "name": agent_name,
            "display_name": name.replace("_", " ").title(),
            "description": f"Delegate tasks to the {name.replace('_', ' ').title()} agent.",
            "profile": {
                "role": role,
                "goal": goal,
                "backstory": backstory,
                "tone": "Professional and helpful",
            },
            "memory": {
                "enabled": True,
                "rag_limit": 5,
                "search_types": ["code", "documentation", "lesson"],
            },
            "tools": [],
            "llm_config": {
                "model_hint": custom_model or "claude-sonnet-4.5",
                "fallback_hints": ["auto"],
                "temperature": 0.5,
                "max_tokens": 4000,
            },
        }

    output_file = agents_dir / f"{agent_name}.yaml"

    if output_file.exists() and not click.confirm(
        f"Agent {agent_name}.yaml already exists. Overwrite?"
    ):
        click.echo("Aborted.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.echo(f"✅ Created agent: {output_file}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. Edit {output_file} to customize your agent")
    click.echo("  2. Restart the MCP server to activate this agent")
    click.echo(f"  3. Use the 'ask_{agent_name}' tool in your IDE")


@agent_group.command("list")
def agent_list_command() -> None:
    """List all configured agents."""
    agents_dir = Path.cwd() / "agents"

    if not agents_dir.exists():
        click.echo("No agents directory found.")
        click.echo("Create an agent with: nexus-agent init <name>")
        return

    yaml_files = list(agents_dir.glob("*.yaml")) + list(agents_dir.glob("*.yml"))

    if not yaml_files:
        click.echo("No agents found.")
        click.echo("Create an agent with: nexus-agent init <name>")
        return

    click.echo("📋 Custom Agents:")
    click.echo("")

    import yaml

    for yaml_file in sorted(yaml_files):
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            name = data.get("name", yaml_file.stem)
            display_name = data.get("display_name", name)
            role = data.get("profile", {}).get("role", "Unknown")
            click.echo(f"  • {display_name} (ask_{name})")
            click.echo(f"    Role: {role}")
            click.echo("")
        except Exception as e:
            click.echo(f"  ⚠️  {yaml_file.name}: Failed to load ({e})")


# Entry points for pyproject.toml scripts
def init_command_entry() -> None:
    """Entry point for nexus-init."""
    cli(["init"])


def index_command_entry() -> None:
    """Entry point for nexus-index."""
    # Get args after the command name
    import sys

    cli(["index"] + sys.argv[1:])


def index_mcp_command_entry() -> None:
    """Entry point for nexus-index-mcp."""
    import sys

    cli(["index-mcp"] + sys.argv[1:])


def mcp_command_entry() -> None:
    """Entry point for nexus-mcp."""
    import sys

    cli(["mcp"] + sys.argv[1:])


def agent_command_entry() -> None:
    """Entry point for nexus-agent."""
    import sys

    cli(["agent"] + sys.argv[1:])


if __name__ == "__main__":
    cli()
