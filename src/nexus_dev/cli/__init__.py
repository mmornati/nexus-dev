"""Nexus-Dev CLI commands.

Provides commands for:
- nexus-init: Initialize Nexus-Dev in a project
- nexus-index: Manually index files
- nexus-status: Show project statistics
"""

from __future__ import annotations

import sys

import click

from nexus_dev.cli.agent import agent_group
from nexus_dev.cli.gateway import gateway_group
from nexus_dev.cli.github import import_github_command
from nexus_dev.cli.index import (
    clean_command,
    index_command,
    index_lesson_command,
    reindex_command,
)
from nexus_dev.cli.init import agent_config_command, init_command
from nexus_dev.cli.mcp import index_mcp_command, mcp_group
from nexus_dev.cli.search import (
    export_command,
    inspect_command,
    search_command,
    status_command,
)

# Re-export utilities if needed (optional)
# from nexus_dev.cli.utils import ...


@click.group()
@click.version_option(version="0.1.0", prog_name="nexus-dev")
def cli() -> None:
    """Nexus-Dev CLI - Local RAG for AI coding agents.

    Nexus-Dev provides persistent memory for AI coding assistants by indexing
    your code and documentation into a local vector database.
    """


# Register commands
cli.add_command(init_command)
cli.add_command(agent_config_command)
cli.add_command(index_command)
cli.add_command(index_lesson_command)
cli.add_command(reindex_command)
cli.add_command(clean_command)
cli.add_command(search_command)
cli.add_command(export_command)
cli.add_command(status_command)
cli.add_command(inspect_command)
cli.add_command(mcp_group)
cli.add_command(index_mcp_command)
cli.add_command(agent_group)
cli.add_command(import_github_command)
cli.add_command(gateway_group)


# Entry points for pyproject.toml scripts


def init_command_entry() -> None:
    """Entry point for nexus-init."""
    cli(["init"])


def index_command_entry() -> None:
    """Entry point for nexus-index."""
    # Get args after the command name
    cli(["index"] + sys.argv[1:])


def index_mcp_command_entry() -> None:
    """Entry point for nexus-index-mcp."""
    cli(["index-mcp"] + sys.argv[1:])


def mcp_command_entry() -> None:
    """Entry point for nexus-mcp."""
    cli(["mcp"] + sys.argv[1:])


def agent_command_entry() -> None:
    """Entry point for nexus-agent."""
    cli(["agent"] + sys.argv[1:])
