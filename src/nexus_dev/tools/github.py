"""GitHub integration tools for Nexus-Dev."""

from __future__ import annotations

import logging

from nexus_dev.app_state import get_config, get_database, get_mcp_config
from nexus_dev.github_importer import GitHubImporter
from nexus_dev.mcp_client import MCPClientManager

logger = logging.getLogger(__name__)


async def import_github_issues(
    repo: str,
    owner: str,
    limit: int = 10,
    state: str = "all",
) -> str:
    """Import GitHub issues and pull requests into the knowledge base.

    Imports issues from the specified repository using the 'github' MCP server.
    Items are indexed for semantic search (search_knowledge) and can be filtered
    by 'github_issue' or 'github_pr' content types.

    Args:
        repo: Repository name (e.g., "nexus-dev").
        owner: Repository owner (e.g., "mmornati").
        limit: Maximum number of issues to import (default: 10).
        state: Issue state filter: "open" (default), "closed", or "all".

    Returns:
        Summary of imported items.
    """
    database = get_database()
    config = get_config()

    if not config:
        return "Error: No project configuration found. Run 'nexus-init' first."

    client_manager = MCPClientManager()
    mcp_config = get_mcp_config()

    importer = GitHubImporter(database, config.project_id, client_manager, mcp_config)

    try:
        count = await importer.import_issues(owner, repo, limit, state)
        return f"Successfully imported {count} issues/PRs from {owner}/{repo}."
    except Exception as e:
        return f"Failed to import issues: {e!s}"
