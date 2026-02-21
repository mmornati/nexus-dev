"""Shared utilities for Nexus-Dev CLI."""

from __future__ import annotations

import asyncio
import shutil
import stat
from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import datetime
from fnmatch import fnmatch
from functools import update_wrapper
from pathlib import Path
from typing import Any

import click
import yaml

from nexus_dev.config import NexusConfig
from nexus_dev.database import Document, DocumentType, NexusDatabase, generate_document_id
from nexus_dev.embeddings import validate_embedding_config


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
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If a loop is already running (e.g. in tests), use it
        return asyncio.get_event_loop().run_until_complete(coro)


def requires_config(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to load and validate configuration before running a command."""

    @click.pass_context
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        config_path = Path.cwd() / "nexus_config.json"
        if not config_path.exists():
            click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
            ctx.exit(1)

        try:
            config = NexusConfig.load(config_path)
        except Exception as e:
            click.echo(f"❌ Failed to load configuration: {e}", err=True)
            ctx.exit(1)

        if not _validate_embeddings_or_exit(config):
            ctx.exit(1)

        return ctx.invoke(f, *args, config=config, **kwargs)

    return update_wrapper(wrapper, f)


def detect_editor() -> str:
    """Detect which AI coding editor is likely in use."""
    cwd = Path.cwd()

    # Antigravity (Google)
    if (
        (cwd / ".antigravity").exists()
        or (cwd / ".geminiignore").exists()
        or (cwd / ".antigravityignore").exists()
    ):
        return "antigravity"

    # Cursor
    if (cwd / ".cursor").exists() or (cwd / ".cursorrules").exists():
        return "cursor"

    # Claude Code
    if (cwd / ".claude").exists() or (cwd / "CLAUDE.md").exists():
        return "claude"

    # GitHub Copilot
    if (cwd / ".github" / "copilot-instructions.md").exists():
        return "copilot"

    # Windsurf
    if (cwd / ".windsurfrules").exists() or (cwd / ".windsurf").exists():
        return "windsurf"

    # Gemini Code Assist
    if (cwd / "GEMINI.md").exists():
        return "gemini"

    # Zed
    if (cwd / ".rules").exists():
        return "zed"

    # VS Code (generic)
    if (cwd / ".vscode").exists():
        return "vscode"

    return "agents"


def _install_antigravity_files(cwd: Path, template_dir: Path, update: bool) -> None:
    """Install Antigravity-specific configuration files."""
    files = {
        "antigravity_geminiignore": ".geminiignore",
        "antigravity_antigravityignore": ".antigravityignore",
        "antigravity_aiexclude": ".aiexclude",
    }

    for template_name, target_name in files.items():
        target_path = cwd / target_name
        if target_path.exists() and not update:
            click.echo(f"   Skipping {target_name} (exists)")
            continue

        template_path = template_dir / template_name
        if template_path.exists():
            shutil.copy(template_path, target_path)
            click.echo(f"✅ Created {target_name}")
        else:
            click.echo(f"⚠️  Template {template_name} not found.")


def _create_symlink(target: Path, source: Path, update: bool) -> None:
    """Create a symlink from target to source."""
    if target.exists():
        if not update:
            click.echo(f"   Skipping {target.name} (exists)")
            return
        if target.is_symlink() or target.is_file():
            target.unlink()

    try:
        target.symlink_to(source.name)
        click.echo(f"✅ Linked {target.name} -> {source.name}")
    except OSError as e:
        # Fallback to copy if symlink fails (e.g. Windows without privileges)
        shutil.copy(source, target)
        click.echo(f"✅ Copied {source.name} to {target.name} (symlink failed: {e})")


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
    # Path(__file__) is src/nexus_dev/cli/utils.py
    # Templates are in src/nexus_dev/templates
    template_path = Path(__file__).parent.parent / "templates" / "pre-commit-hook"
    if template_path.exists():
        shutil.copy(template_path, hook_path)
    else:
        click.echo("❌ Pre-commit hook template not found. Hook installation failed.", err=True)
        return

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
