"""Nexus-Dev CLI commands.

Provides commands for:
- nexus-init: Initialize Nexus-Dev in a project
- nexus-index: Manually index files
- nexus-status: Show project statistics
"""

from __future__ import annotations

import asyncio
import json
import shutil
import stat
from collections.abc import Coroutine
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

import click

from .chunkers import ChunkerRegistry
from .config import NexusConfig
from .database import Document, DocumentType, NexusDatabase, generate_document_id
from .embeddings import create_embedder
from .mcp_client import MCPClientManager, MCPServerConnection
from .mcp_config import MCPConfig, MCPServerConfig


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
    prompt="Project name",
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
def init_command(
    project_name: str,
    embedding_provider: Literal["openai", "ollama"],
    install_hook: bool,
) -> None:
    """Initialize Nexus-Dev in the current repository.

    Creates configuration file, lessons directory, and optionally installs
    the pre-commit hook for automatic indexing.
    """
    cwd = Path.cwd()
    config_path = cwd / "nexus_config.json"

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
        _install_hook(cwd)

    # Print summary
    click.echo("")
    click.echo("🎉 Nexus-Dev initialized successfully!")
    click.echo("")
    click.echo("Next steps:")
    click.echo("  1. Index your code: nexus-index src/")
    click.echo("  2. Index documentation: nexus-index docs/ README.md")
    click.echo("  3. Configure your AI agent to use the nexus-dev MCP server")
    click.echo("")
    click.echo(f"Project ID: {config.project_id}")

    if embedding_provider == "openai":
        click.echo("")
        click.echo("⚠️  Using OpenAI embeddings. Ensure OPENAI_API_KEY is set.")


def _install_hook(cwd: Path) -> None:
    """Install pre-commit hook."""
    git_dir = cwd / ".git"
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
    template_path = Path(__file__).parent.parent.parent / "templates" / "pre-commit-hook"
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

    click.echo("✅ Installed pre-commit hook")


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
                # Recursively find files
                for file_path in path.rglob("*"):
                    if file_path.is_file() and _should_index(file_path, config):
                        files_to_index.append(file_path)
            else:
                # Only immediate children
                for file_path in path.iterdir():
                    if file_path.is_file() and _should_index(file_path, config):
                        files_to_index.append(file_path)

    if not files_to_index:
        if not quiet:
            click.echo("No files to index.")
        return

    if not quiet:
        click.echo(f"📁 Indexing {len(files_to_index)} file(s)...")

    # Index files
    total_chunks = 0
    errors = 0

    for file_path in files_to_index:
        try:
            # Read file
            content = file_path.read_text(encoding="utf-8")

            # Determine type
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
                    _index_chunks_sync(chunks, config.project_id, doc_type, embedder, database)
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

        doc = Document(
            id=doc_id,
            text=chunk.get_searchable_text(),
            vector=embedding,
            project_id=project_id,
            file_path=chunk.file_path,
            doc_type=doc_type,
            chunk_type=chunk.chunk_type.value,
            language=chunk.language,
            name=chunk.name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
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


@cli.command("status")
def status_command() -> None:
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
        embedder = create_embedder(config)
        database = NexusDatabase(config, embedder)
        database.connect()

        stats = _run_async(database.get_project_stats(config.project_id))

        click.echo("📈 Statistics:")
        click.echo(f"   Total chunks: {stats.get('total', 0)}")
        click.echo(f"   Code: {stats.get('code', 0)}")
        click.echo(f"   Documentation: {stats.get('documentation', 0)}")
        click.echo(f"   Lessons: {stats.get('lesson', 0)}")

    except Exception as e:
        click.echo(f"⚠️  Could not connect to database: {e!s}")


@cli.command("reindex")
@click.confirmation_option(prompt="This will delete and rebuild the entire index. Continue?")
def reindex_command() -> None:
    """Re-index entire project (clear and rebuild)."""
    config_path = Path.cwd() / "nexus_config.json"

    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)
    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    click.echo("🗑️  Clearing existing index...")
    click.echo("🗑️  Clearing existing index...")
    # Reset database to handle schema changes
    database.reset()
    # Re-connect to create new table with updated schema
    database.connect()
    click.echo("   Index cleared and schema updated")

    click.echo("")
    click.echo("📁 Re-indexing project...")

    # Index based on include patterns
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
            for file_path in docs_path.rglob("*"):
                if file_path.is_file():
                    files_to_index.append(file_path)

    # Remove duplicates
    files_to_index = list(set(files_to_index))

    click.echo(f"   Found {len(files_to_index)} files to index")

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

        except Exception as e:
            click.echo(f"  ❌ Failed to index {file_path.name}: {e!s}", err=True)

    click.echo("")
    click.echo(f"✅ Re-indexed {total_chunks} chunks from {len(files_to_index)} files")


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
    config_path = Path(config) if config else Path.home() / ".config" / "mcp" / "config.json"
    if not config_path.exists():
        click.echo(f"MCP config not found: {config_path}")
        click.echo("Specify --config or create ~/.config/mcp/config.json")
        return

    try:
        mcp_config = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON in MCP config: {e}", err=True)
        return

    # Determine which servers to index
    servers_to_index = []
    if index_all:
        servers_to_index = list(mcp_config.get("mcpServers", {}).keys())
    elif server:
        servers_to_index = [server]
    else:
        click.echo("Specify --server or --all")
        return

    # Index each server
    asyncio.run(_index_mcp_servers(mcp_config, servers_to_index))


async def _index_mcp_servers(mcp_config: dict[str, Any], server_names: list[str]) -> None:
    """Index tools from specified MCP servers."""
    # Load config
    config_path = Path.cwd() / "nexus_config.json"
    if not config_path.exists():
        click.echo("❌ nexus_config.json not found. Run 'nexus-init' first.", err=True)
        return

    config = NexusConfig.load(config_path)
    client = MCPClientManager()
    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    for name in server_names:
        server_config = mcp_config.get("mcpServers", {}).get(name)
        if not server_config:
            click.echo(f"Server not found: {name}")
            continue

        click.echo(f"Indexing tools from: {name}")

        try:
            connection = MCPServerConnection(
                name=name,
                command=server_config.get("command", ""),
                args=server_config.get("args", []),
                env=server_config.get("env"),
            )

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
            profiles={},
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


if __name__ == "__main__":
    cli()
