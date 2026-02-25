"""Nexus-Dev MCP Server State.

This module holds the global state and singleton accessors for the application.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context

from .agents import AgentManager
from .config import NexusConfig
from .database import NexusDatabase
from .embeddings import EmbeddingProvider, create_embedder
from .gateway.connection_manager import ConnectionManager
from .hybrid_db import HybridDatabase
from .mcp_config import MCPConfig

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("nexus-dev")

# Global state (initialized on startup or lazily)
_config: NexusConfig | None = None
_embedder: EmbeddingProvider | None = None
_database: NexusDatabase | None = None
_mcp_config: MCPConfig | None = None
_hybrid_db: HybridDatabase | None = None
_connection_manager: ConnectionManager | None = None
_agent_manager: AgentManager | None = None
_project_root: Path | None = None


def find_project_root() -> Path | None:
    """Find the project root by looking for nexus_config.json.

    Walks up from the current directory to find nexus_config.json.
    Also checks NEXUS_PROJECT_ROOT environment variable as a fallback.

    Returns:
        Path to project root if found, None otherwise.
    """
    global _project_root
    if _project_root:
        return _project_root

    import os

    # First check environment variable
    env_root = os.environ.get("NEXUS_PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root)
        if (env_path / "nexus_config.json").exists():
            logger.debug("Found project root from NEXUS_PROJECT_ROOT: %s", env_path)
            return env_path

    current = Path.cwd().resolve()
    logger.debug("Searching for project root from cwd: %s", current)

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / "nexus_config.json").exists():
            logger.debug("Found project root: %s", parent)
            _project_root = parent
            return parent
        # Stop at filesystem root
        if parent == parent.parent:
            logger.debug("Reached filesystem root without finding nexus_config.json")
            break

    logger.debug("No project root found (no nexus_config.json in directory tree)")
    return None


def get_config() -> NexusConfig | None:
    """Get or load configuration.

    Returns None if no nexus_config.json exists in cwd.
    This allows the MCP server to work without a project-specific config,
    enabling cross-project searches.
    """
    global _config
    if _config is None:
        root = find_project_root()
        config_path = (root if root else Path.cwd()) / "nexus_config.json"
        if config_path.exists():
            _config = NexusConfig.load(config_path)
        # Don't create default - None means "all projects"
    return _config


def get_mcp_config() -> MCPConfig | None:
    """Get or load MCP configuration.

    Loads from:
    1. Global: ~/.nexus/mcp_config.json
    2. Local: <project_root>/.nexus/mcp_config.json

    Returns merged configuration, or None if neither exists.
    """
    global _mcp_config
    if _mcp_config is None:
        root = find_project_root()
        local_path = (root if root else Path.cwd()) / ".nexus" / "mcp_config.json"
        global_path = Path.home() / ".nexus" / "mcp_config.json"

        _mcp_config = MCPConfig.load_hierarchical(global_path, local_path)

    return _mcp_config


def get_active_server_names() -> list[str]:
    """Get names of active MCP servers.

    Returns:
        List of active server names.
    """
    mcp_config = get_mcp_config()
    if not mcp_config:
        return []

    # Find the name for each active server config
    active_servers = mcp_config.get_active_servers()
    active_names = []
    for name, config in mcp_config.servers.items():
        if config in active_servers:
            active_names.append(name)
    return active_names


def get_connection_manager() -> ConnectionManager:
    """Get or create connection manager singleton.

    Returns:
        ConnectionManager instance for managing MCP server connections.
    """
    global _connection_manager
    if _connection_manager is None:
        mcp_config = get_mcp_config()
        if mcp_config is not None:
            _connection_manager = ConnectionManager(
                default_max_concurrent=mcp_config.gateway.max_concurrent_connections,
                shutdown_timeout=mcp_config.gateway.shutdown_timeout,
                cache_settings=mcp_config.gateway.cache,
            )
        else:
            _connection_manager = ConnectionManager()
    return _connection_manager


def get_embedder() -> EmbeddingProvider:
    """Get or create embedding provider."""
    global _embedder
    if _embedder is None:
        config = get_config()
        if config is None:
            # Create minimal config for embeddings only
            config = NexusConfig.create_new("default")
        _embedder = create_embedder(config)
    return _embedder


def get_database() -> NexusDatabase:
    """Get or create database connection."""
    global _database
    if _database is None:
        config = get_config()
        if config is None:
            # Create minimal config for database access
            config = NexusConfig.create_new("default")
        embedder = get_embedder()
        _database = NexusDatabase(config, embedder)
        _database.connect()
    return _database


def get_hybrid_db() -> HybridDatabase:
    """Get or create hybrid database connection."""
    global _hybrid_db
    if _hybrid_db is None:
        config = get_config()
        if config is None:
            # Create minimal config
            config = NexusConfig.create_new("default")
        _hybrid_db = HybridDatabase(config)
        # We don't verify connection here as it's opt-in via config
    return _hybrid_db


def get_hybrid_database() -> HybridDatabase:
    """Get hybrid database instance for graph queries.

    Returns:
        HybridDatabase instance

    Raises:
        RuntimeError: If hybrid mode is not enabled
    """
    hybrid_db = get_hybrid_db()

    if not hybrid_db.config.enable_hybrid_db:
        raise RuntimeError(
            "Hybrid database is not enabled. Set enable_hybrid_db=True in nexus_config.json"
        )

    # Ensure connection
    hybrid_db.connect()
    return hybrid_db


async def get_project_root_from_session(ctx: Context[Any, Any]) -> Path | None:
    """Get the project root from MCP session roots.

    Uses session.list_roots() to query the IDE for workspace folders.

    Args:
        ctx: FastMCP Context with session access.

    Returns:
        Path to the project root if found, None otherwise.
    """
    try:
        # Query the IDE for workspace roots
        roots_result = await ctx.session.list_roots()

        if not roots_result.roots:
            logger.debug("No roots returned from session.list_roots()")
            return None

        # Look for a root that contains nexus_config.json (indicates a nexus project)
        for root in roots_result.roots:
            uri = str(root.uri)
            # Handle file:// URIs
            path = Path(uri[7:]) if uri.startswith("file://") else Path(uri)

            if path.exists() and (path / "nexus_config.json").exists():
                logger.debug("Found nexus project root from session: %s", path)
                return path

        # Fall back to first root if none have nexus_config.json
        first_uri = str(roots_result.roots[0].uri)
        path = Path(first_uri[7:]) if first_uri.startswith("file://") else Path(first_uri)

        if path.exists():
            logger.debug("Using first root from session: %s", path)
            return path

    except Exception as e:
        logger.debug("Failed to get roots from session: %s", e)

    return None


def get_agent_manager() -> AgentManager | None:
    """Get the global agent manager instance."""
    return _agent_manager


def set_agent_manager(manager: AgentManager | None) -> None:
    """Set the global agent manager instance."""
    global _agent_manager
    _agent_manager = manager


def set_project_root(root: Path | None) -> None:
    """Set the global project root."""
    global _project_root
    _project_root = root


def reset_state() -> None:
    """Reset all global state variables. useful for testing or refreshing."""
    global _config, _embedder, _database, _mcp_config
    global _hybrid_db, _connection_manager, _agent_manager, _project_root

    _config = None
    _embedder = None
    _database = None
    _mcp_config = None
    _hybrid_db = None
    _connection_manager = None
    _agent_manager = None
    _project_root = None
