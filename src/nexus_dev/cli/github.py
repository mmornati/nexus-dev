"""GitHub integration commands."""

from __future__ import annotations

from pathlib import Path

import click

from nexus_dev.cli.utils import _run_async, requires_config
from nexus_dev.config import NexusConfig
from nexus_dev.database import NexusDatabase
from nexus_dev.embeddings import create_embedder
from nexus_dev.github_importer import GitHubImporter
from nexus_dev.mcp_client import MCPClientManager
from nexus_dev.mcp_config import MCPConfig


@click.command("import-github")
@click.option("--repo", required=True, help="Repository name")
@click.option("--owner", required=True, help="Repository owner")
@click.option("--limit", default=20, help="Maximum number of issues to import")
@click.option("--state", default="all", help="Issue state (open, closed, all)")
@requires_config
def import_github_command(
    config: NexusConfig, repo: str, owner: str, limit: int, state: str
) -> None:
    """Import GitHub issues and PRs."""
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
