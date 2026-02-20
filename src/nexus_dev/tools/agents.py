"""Agent management tools for Nexus-Dev."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp.server import Context

from nexus_dev.agents import AgentConfig, AgentExecutor, AgentManager
from nexus_dev.app_state import (
    find_project_root,
    get_config,
    get_database,
    get_project_root_from_session,
    mcp,
    reset_state,
    set_agent_manager,
    set_project_root,
)
from nexus_dev.database import NexusDatabase

logger = logging.getLogger(__name__)


def register_agent_tools(database: NexusDatabase, agent_manager: AgentManager | None) -> None:
    """Register dynamic tools for each loaded agent.

    Each agent becomes an MCP tool named `ask_<agent_name>`.
    """
    if agent_manager is None:
        return

    for agent_config in agent_manager:

        def create_agent_tool(cfg: AgentConfig) -> Any:
            """Create a closure to capture the agent config."""

            async def agent_tool(task: str) -> str:
                """Execute a task using the configured agent.

                Args:
                    task: The task description to execute.

                Returns:
                    Agent's response.
                """
                logger.info("Agent tool called: ask_%s for task: %s", cfg.name, task[:100])
                executor = AgentExecutor(cfg, database, mcp)
                config = get_config()
                project_id = config.project_id if config else None
                return await executor.execute(task, project_id)

            # Set the docstring dynamically
            agent_tool.__doc__ = cfg.description
            return agent_tool

        tool_name = f"ask_{agent_config.name}"
        tool_func = create_agent_tool(agent_config)

        # We use mcp.add_tool directly to allow dynamic registration at runtime
        # FastMCP.tool is a decorator, add_tool is the underlying method
        mcp.add_tool(fn=tool_func, name=tool_name, description=agent_config.description)
        logger.info("Registered agent tool: %s", tool_name)


async def list_agents(ctx: Context[Any, Any]) -> str:
    """List available agents in the current workspace.

    Discovers agents from the agents/ directory in the IDE's current workspace.
    Use ask_agent tool to execute tasks with a specific agent.

    Returns:
        List of available agents with names and descriptions.
    """
    # Try to get project root from session (MCP roots)
    project_root = await get_project_root_from_session(ctx)

    # Fall back to environment variable or cwd
    if not project_root:
        project_root = find_project_root()

    if not project_root:
        return "No project root found. Make sure you have a nexus_config.json in your workspace."

    agents_dir = project_root / "agents"
    if not agents_dir.exists():
        return f"No agents directory found at {agents_dir}. Create it and add agent YAML files."

    # Load agents from directory
    agent_manager = AgentManager(agents_dir=agents_dir)

    if len(agent_manager) == 0:
        return f"No agents found in {agents_dir}. Add YAML agent configuration files."

    lines = ["# Available Agents", ""]
    for agent in agent_manager:
        lines.append(f"## {agent.display_name or agent.name}")
        lines.append(f"- **Name:** `{agent.name}`")
        lines.append(f"- **Description:** {agent.description}")
        if agent.profile:
            lines.append(f"- **Role:** {agent.profile.role}")
        lines.append("")

    lines.append("Use `ask_agent` tool with the agent name to execute a task.")
    return "\n".join(lines)


async def ask_agent(agent_name: str, task: str, ctx: Context[Any, Any]) -> str:
    """Execute a task using a custom agent from the current workspace.

    Loads the specified agent from the workspace's agents/ directory and
    executes the given task.

    Args:
        agent_name: Name of the agent to use (e.g., 'nexus_architect').
        task: The task description to execute.

    Returns:
        Agent's response.
    """
    # Get database
    database = get_database()
    if database is None:
        return "Database not initialized. Run nexus-init first."

    # Try to get project root from session (MCP roots)
    project_root = await get_project_root_from_session(ctx)

    # Fall back to environment variable or cwd
    if not project_root:
        project_root = find_project_root()

    if not project_root:
        return "No project root found. Make sure you have a nexus_config.json in your workspace."

    agents_dir = project_root / "agents"
    if not agents_dir.exists():
        return f"No agents directory found at {agents_dir}."

    # Load agents from directory
    agent_manager = AgentManager(agents_dir=agents_dir)
    agent_config = agent_manager.get_agent(agent_name)

    if not agent_config:
        available = [a.name for a in agent_manager]
        return f"Agent '{agent_name}' not found. Available agents: {available}"

    # Execute the task
    try:
        executor = AgentExecutor(agent_config, database, mcp)
        config = get_config()
        project_id = config.project_id if config else None
        return await executor.execute(task, project_id)
    except Exception as e:
        logger.error("Agent execution failed: %s", e, exc_info=True)
        return f"Agent execution failed: {e!s}"


async def refresh_agents(ctx: Context[Any, Any]) -> str:
    """Discovers and registers individual agent tools from the current workspace.

    This tool:
    1. Queries the IDE for the current workspace root.
    2. Scans the 'agents/' directory for agent configurations.
    3. Dynamically registers 'ask_<agent_name>' tools for each agent found.
    4. Notifies the IDE that the tool list has changed.

    Returns:
        A report of registered agents or an error message.
    """
    project_root = await get_project_root_from_session(ctx)
    if not project_root:
        return "No nexus project root found in workspace (nexus_config.json missing)."

    # Persist the root globally so other tools find it
    set_project_root(project_root)

    # Reload other configs if they were initialized lazily from /
    # Reset specific globals as in original code
    # Original code:
    # global _config, _mcp_config, _database, _hybrid_db
    # _config = None
    # _mcp_config = None
    # _database = None
    # _hybrid_db = None

    # We use reset_state() but keep connection_manager if possible?
    # Original code didn't reset connection_manager.
    # But for safety and simplicity, reset_state() is cleaner.
    # However, if connection_manager holds persistent connections, we might want to keep it.
    # But config might change so maybe better to reset.

    # For now, I'll manually reset the ones that were reset in original code
    # relying on app_state internals? No, I should add specific reset or use reset_state.
    # Using reset_state clears everything. Let's assume that's acceptable or even better.
    # But wait, reset_state clears _project_root too!
    # So I must set project root AFTER reset.

    reset_state()
    set_project_root(project_root)

    database = get_database()
    if database is None:
        return "Database not initialized."

    agents_dir = project_root / "agents"
    if not agents_dir.exists():
        return f"No agents directory found at {agents_dir}."

    agent_manager = AgentManager(agents_dir=agents_dir)
    set_agent_manager(agent_manager)

    if len(agent_manager) == 0:
        return "No agents found in agents/ directory."

    # Register the tools
    register_agent_tools(database, agent_manager)

    # Notify the client that the tool list has changed
    try:
        await ctx.session.send_tool_list_changed()
    except Exception as e:
        logger.warning("Failed to send tool_list_changed notification: %s", e)

    agent_names = [a.name for a in agent_manager]
    return f"Successfully registered {len(agent_names)} agent tools: {', '.join(agent_names)}"
