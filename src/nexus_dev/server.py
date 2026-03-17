"""Nexus-Dev MCP Server.

This module implements the MCP server using FastMCP, exposing tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from threading import Thread
from types import FrameType

from .agents import AgentManager
from .app_state import (
    find_project_root,
    get_config,
    get_database,
    get_mcp_config,
    mcp,
    set_agent_manager,
)
from .mcp_client import MCPClientManager, MCPServerConnection
from .tools.agents import ask_agent, list_agents, refresh_agents, register_agent_tools
from .tools.context import (
    get_recent_context,
    get_search_suggestions,
    get_session_context,
    set_session_context,
)
from .tools.github import import_github_issues
from .tools.graph import (
    find_callers,
    find_implementations,
    find_related_entities,
    search_dependencies,
)
from .tools.insight import search_implementations, search_insights
from .tools.knowledge import (
    get_project_context,
    index_file,
    record_implementation,
    record_insight,
    record_lesson,
)
from .tools.mcp_tools import (
    get_active_tools_resource,
    get_tool_schema,
    invoke_tool,
    list_servers,
    search_tools,
)
from .tools.search import (
    search_code,
    search_docs,
    search_knowledge,
    search_lessons,
    smart_search,
)
from .tools.system_prompts import get_gateway_prompt

logger = logging.getLogger(__name__)


async def _auto_index_mcp_tools() -> None:
    """Automatically index MCP tools from configured servers.

    This function runs on server startup to index tools from all active
    MCP servers defined in the MCP configuration file.
    """
    from .database import Document, DocumentType
    from .embeddings import create_embedder

    mcp_config = get_mcp_config()
    if not mcp_config:
        logger.debug("No MCP config found, skipping auto-index")
        return

    config = get_config()
    if config is None:
        from .config import NexusConfig

        config = NexusConfig.create_new("default")

    embedder = create_embedder(config)
    database = get_database()

    active_servers = mcp_config.get_active_servers()
    if not active_servers:
        logger.debug("No active MCP servers found, skipping auto-index")
        return

    server_names = [
        name for name, srv_cfg in mcp_config.servers.items() if srv_cfg in active_servers
    ]
    logger.info("Found MCP config: %s", mcp_config)
    logger.info("Auto-indexing MCP tools from %d servers...", len(server_names))

    async with MCPClientManager() as client:
        total_tools = 0
        servers_indexed = 0

        for name in server_names:
            server_config = mcp_config.servers.get(name)
            if not server_config or not server_config.enabled:
                continue

            logger.debug("Indexing tools from server: %s", name)

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

            try:
                tools = await client.get_tools(connection)
                logger.debug("Found %d tools from %s", len(tools), name)

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

                total_tools += len(tools)
                servers_indexed += 1
                logger.info("Indexed %d tools from %s", len(tools), name)

            except Exception as e:
                logger.warning("Failed to index tools from %s: %s", name, e)
                continue

    logger.info("Indexed %d tools from %d servers", total_tools, servers_indexed)


# Register Tools
mcp.tool()(search_knowledge)
mcp.tool()(search_docs)
mcp.tool()(search_lessons)
mcp.tool()(search_code)
mcp.tool()(search_tools)
mcp.tool()(get_recent_context)
mcp.tool()(set_session_context)
mcp.tool()(get_session_context)
mcp.tool()(get_search_suggestions)
mcp.tool()(smart_search)
mcp.tool()(search_dependencies)
mcp.tool()(find_callers)
mcp.tool()(find_implementations)
mcp.tool()(list_servers)
mcp.tool()(get_tool_schema)
mcp.tool()(invoke_tool)
mcp.tool()(index_file)
mcp.tool()(import_github_issues)
mcp.tool()(record_lesson)
mcp.tool()(record_insight)
mcp.tool()(search_insights)
mcp.tool()(record_implementation)
mcp.tool()(search_implementations)
mcp.tool()(find_related_entities)
mcp.tool()(get_project_context)
mcp.tool()(list_agents)
mcp.tool()(ask_agent)
mcp.tool()(refresh_agents)
mcp.tool()(get_gateway_prompt)

# Register Resources
mcp.resource("mcp://nexus-dev/active-tools")(get_active_tools_resource)


def main() -> None:
    """Run the MCP server."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Nexus-Dev MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default) or sse for Docker/network deployment",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--no-auto-index",
        action="store_true",
        help="Disable automatic MCP tools indexing on startup",
    )
    args = parser.parse_args()

    # Configure logging to always use stderr and a debug file
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Stderr handler
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(stderr_handler)

    # File handler for persistent debugging
    try:
        file_handler = logging.FileHandler("/tmp/nexus-dev-debug.log")
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
    except Exception:
        pass  # Fallback if /tmp is not writable

    root_logger.setLevel(logging.DEBUG)

    # Also ensure the module-specific logger is at INFO
    logger.setLevel(logging.DEBUG)

    def handle_signal(sig: int, frame: FrameType | None) -> None:
        logger.info("Received signal %s, shutting down...", sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Initialize on startup
    try:
        logger.info("Starting Nexus-Dev MCP server...")
        get_config()
        database = get_database()
        mcp_config = get_mcp_config()

        # Auto-index MCP tools in background if enabled and config exists
        if not args.no_auto_index and mcp_config:
            # Find config path for logging
            root = find_project_root()
            local_path = (root if root else Path.cwd()) / ".nexus" / "mcp_config.json"
            global_path = Path.home() / ".nexus" / "mcp_config.json"

            config_path = (
                local_path
                if local_path.exists()
                else (global_path if global_path.exists() else None)
            )
            logger.info("Found MCP config: %s", config_path or "unknown")
            logger.info("Auto-indexing MCP tools...")

            def run_index() -> None:
                asyncio.run(_auto_index_mcp_tools())

            index_thread = Thread(target=run_index, daemon=True)
            index_thread.start()

        # Load and register custom agents
        # Find project root and look for agents directory
        logger.debug("Current working directory: %s", Path.cwd())
        project_root = find_project_root()
        agents_dir = project_root / "agents" if project_root else None
        logger.debug("Project root: %s", project_root)
        logger.debug("Agents directory: %s", agents_dir)

        agent_manager = AgentManager(agents_dir=agents_dir)
        set_agent_manager(agent_manager)
        register_agent_tools(database, agent_manager)

        # Run server with selected transport
        if args.transport == "sse":
            logger.info(
                "Server initialization complete, running SSE transport on %s:%d",
                args.host,
                args.port,
            )
            mcp.run(transport="sse", host=args.host, port=args.port)  # type: ignore
        else:
            logger.info("Server initialization complete, running stdio transport")
            mcp.run(transport="stdio")
    except Exception as e:
        logger.critical("Fatal error in MCP server: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        logger.info("MCP server shutdown complete")


if __name__ == "__main__":
    main()
