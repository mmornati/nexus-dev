"""MCP management tools for Nexus-Dev."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from nexus_dev.app_state import (
    get_active_server_names,
    get_connection_manager,
    get_database,
    get_mcp_config,
)
from nexus_dev.database import DocumentType
from nexus_dev.gateway.metrics import get_gateway_metrics

logger = logging.getLogger(__name__)


class SearchToolsCache:
    """LRU cache with TTL for search_tools results."""

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 1000,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def _generate_key(self, query: str, server: str | None, limit: int) -> str:
        key_data = json.dumps(
            {"query": query, "server": server, "limit": limit},
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, query: str, server: str | None, limit: int) -> Any | None:
        key = self._generate_key(query, server, limit)
        if key not in self._cache:
            return None
        expires_at, value = self._cache[key]
        if time.monotonic() > expires_at:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, query: str, server: str | None, limit: int, value: Any) -> None:
        key = self._generate_key(query, server, limit)
        expires_at = time.monotonic() + self._ttl_seconds
        if key in self._cache:
            self._cache.move_to_end(key)
        while len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
        self._cache[key] = (expires_at, value)


_search_tools_cache: SearchToolsCache | None = None


def get_search_tools_cache() -> SearchToolsCache:
    global _search_tools_cache
    if _search_tools_cache is None:
        _search_tools_cache = SearchToolsCache(ttl_seconds=300.0, max_entries=1000)
    return _search_tools_cache


async def list_servers() -> str:
    """List all configured MCP servers and their status.

    Returns:
        List of MCP servers with connection status.
    """
    mcp_config = get_mcp_config()
    if not mcp_config:
        return "No MCP config. Run 'nexus-mcp init' first."

    output = ["## MCP Servers", ""]

    active = mcp_config.get_active_servers()
    active_names = {name for name, cfg in mcp_config.servers.items() if cfg in active}

    output.append("### Active")
    if active_names:
        for name in sorted(active_names):
            server = mcp_config.servers[name]
            details = ""
            if server.transport in ("sse", "http"):
                details = f"{server.transport.upper()}: {server.url}"
            else:
                details = f"Command: {server.command} {' '.join(server.args)}"
            output.append(f"- **{name}**: `{details}`")
    else:
        output.append("*No active servers*")

    output.append("")
    output.append("### Disabled")
    disabled = [name for name, server in mcp_config.servers.items() if name not in active_names]
    if disabled:
        for name in sorted(disabled):
            server = mcp_config.servers[name]
            status = "disabled" if not server.enabled else "not in profile"
            output.append(f"- {name} ({status})")
    else:
        output.append("*No disabled servers*")

    return "\n".join(output)


async def search_tools(
    query: str,
    server: str | None = None,
    limit: int = 5,
) -> str:
    """Find the RIGHT tool from other MCP servers to complete a task.

    This is your FIRST STEP when you need to do something outside Nexus-Dev:
    1. Use search_tools('<action>') to find available tools
    2. Use get_tool_schema(server, tool) to see parameters
    3. Use invoke_tool(server, tool, arguments) to execute

    Example workflow:
    1. search_tools('create GitHub issue') → Returns: github.create_issue with schema
    2. get_tool_schema('github', 'create_issue') → Returns required parameters
    3. invoke_tool(
         server='github',
         tool='create_issue',
         arguments={
             'owner': 'myorg',
             'repo': 'myrepo',
             'title': 'Bug fix',
             'body': 'Fixed the issue'
         }
       )

    CRITICAL: Never try to call external tools directly (e.g., github.create_issue)!
    The tool will not exist and will fail. Always use this gateway workflow.

    Args:
        query: Natural language description of what you want to do.
               Examples: "create a GitHub issue", "list files in directory",
               "send a notification to Home Assistant"
        server: Optional server name to filter results (e.g., "github").
        limit: Maximum results to return (default: 5, max: 10).

    Returns:
        Matching tools with server, name, description, and parameters.
    """
    cache = get_search_tools_cache()
    limit = min(max(1, limit), 10)

    # Check cache first
    cached_result: str | None = cache.get(query, server, limit)
    if cached_result is not None:
        logger.info("[Cache] search_tools HIT for query='%s', server='%s'", query, server)
        metrics = get_gateway_metrics()
        metrics.record_search_tools()
        return cached_result

    logger.info("[Cache] search_tools MISS for query='%s', server='%s'", query, server)
    database = get_database()

    # Search for tools
    results = await database.search(
        query=query,
        doc_type=DocumentType.TOOL,
        limit=limit,
    )
    logger.debug("[%s] Searching tools with query='%s'", "nexus-dev", query)
    try:
        logger.debug("[%s] DB Path in use: %s", "nexus-dev", database.config.get_db_path())
    except Exception as e:
        logger.debug("[%s] Could not print DB path: %s", "nexus-dev", e)

    logger.debug("[%s] Results found: %d", "nexus-dev", len(results))
    if results:
        logger.debug("[%s] First result: %s (%s)", "nexus-dev", results[0].name, results[0].score)

    # Filter by server if specified
    if server and results:
        results = [r for r in results if r.server_name == server]

    # Record the search_tools call for metrics
    metrics = get_gateway_metrics()
    metrics.record_search_tools()

    if not results:
        no_result_msg = (
            f"No tools found matching: '{query}' in server: '{server}'"
            if server
            else f"No tools found matching: '{query}'"
        )
        # Cache the no-result response
        cache.set(query, server, limit, no_result_msg)
        return no_result_msg

    # Format output
    output_parts = [f"## MCP Tools matching: '{query}'", ""]

    for i, result in enumerate(results, 1):
        # Parse parameters schema from stored JSON
        params = json.loads(result.parameters_schema) if result.parameters_schema else {}

        output_parts.append(f"### {i}. {result.server_name}.{result.name}")
        output_parts.append(f"**Description:** {result.text}")
        output_parts.append("")
        if params:
            output_parts.append("**Parameters:**")
            output_parts.append("```json")
            output_parts.append(json.dumps(params, indent=2))
            output_parts.append("```")
        output_parts.append("")

    formatted_output = "\n".join(output_parts)

    # Cache the result
    cache.set(query, server, limit, formatted_output)

    return formatted_output


async def get_tool_schema(server: str, tool: str) -> str:
    """Get the full JSON schema for a specific MCP tool.

    Use this after search_tools to get complete parameter details
    before calling invoke_tool.

    Args:
        server: Server name (e.g., "github")
        tool: Tool name (e.g., "create_pull_request")

    Returns:
        Full JSON schema with parameter types and descriptions.
    """
    mcp_config = get_mcp_config()
    if not mcp_config:
        return "No MCP config. Run 'nexus-mcp init' first."

    if server not in mcp_config.servers:
        available = ", ".join(sorted(mcp_config.servers.keys()))
        return f"Server not found: {server}. Available: {available}"

    server_config = mcp_config.servers[server]
    if not server_config.enabled:
        return f"Server is disabled: {server}"

    conn_manager = get_connection_manager()

    try:
        connection = await conn_manager.get_connection(server, server_config)
        tools_result = await connection.list_tools()

        for t in tools_result.tools:
            if t.name == tool:
                return json.dumps(
                    {
                        "server": server,
                        "tool": tool,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {},
                    },
                    indent=2,
                )

        available_tools = [t.name for t in tools_result.tools[:10]]
        hint = f" Available: {', '.join(available_tools)}..." if available_tools else ""
        return f"Tool not found: {server}.{tool}.{hint}"

    except Exception as e:
        return f"Error connecting to {server}: {e}"


async def invoke_tool(
    server: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Execute a tool on a backend MCP server through the Nexus-Dev gateway.

    This is STEP 3 of the gateway workflow:
    1. First: search_tools('<action>') to find the right tool
    2. Then: get_tool_schema(server, tool) to see required parameters
    3. Finally: invoke_tool(server, tool, arguments) to execute

    Example workflow:
    1. search_tools('create GitHub issue') → Returns: github.create_issue
    2. get_tool_schema('github', 'create_issue') → Returns parameters
    3. invoke_tool(
         server='github',           # NOT 'github.create_issue'!
         tool='create_issue',      # Tool name WITHOUT server prefix
         arguments={
             'owner': 'myorg',
             'repo': 'myrepo',
             'title': 'Bug fix',
             'body': 'Fixed the issue'
         }
       )

    CRITICAL: Pass server and tool as SEPARATE parameters!
    Do NOT combine them like invoke_tool('github.create_issue', {...})

    Args:
        server: MCP server name (e.g., "github", "homeassistant")
        tool: Tool name (e.g., "create_issue", "turn_on_light")
        arguments: Tool arguments as dictionary

    Returns:
        Tool execution result.
    """
    mcp_config = get_mcp_config()
    if not mcp_config:
        return "No MCP config. Run 'nexus-mcp init' first."

    if server not in mcp_config.servers:
        available = ", ".join(sorted(mcp_config.servers.keys()))
        return f"Server not found: {server}. Available: {available}"

    server_config = mcp_config.servers[server]

    if not server_config.enabled:
        return f"Server is disabled: {server}"

    conn_manager = get_connection_manager()

    # Record the invoke_tool call for metrics
    metrics = get_gateway_metrics()
    metrics.record_invoke_tool(server)

    try:
        result = await conn_manager.invoke_tool(
            server,
            server_config,
            tool,
            arguments or {},
        )

        # Format result for AI consumption
        if hasattr(result, "content"):
            # MCP CallToolResult object
            contents = []
            for item in result.content:
                if hasattr(item, "text"):
                    contents.append(item.text)
                else:
                    contents.append(str(item))
            return "\n".join(contents) if contents else "Tool executed successfully (no output)"

        return str(result)

    except Exception as e:
        return f"Tool invocation failed: {e}"


async def get_active_tools_resource() -> str:
    """List MCP tools from active profile servers.

    Returns a list of tools that are available based on the current
    profile configuration in .nexus/mcp_config.json.
    """
    mcp_config = get_mcp_config()
    if not mcp_config:
        return "No MCP config found. Run 'nexus-mcp init' first."

    database = get_database()
    active_servers = get_active_server_names()

    if not active_servers:
        return f"No active servers in profile: {mcp_config.active_profile}"

    # Query all tools once from the database
    all_tools = await database.search(
        query="",
        doc_type=DocumentType.TOOL,
        limit=1000,  # Get all tools
    )

    # Filter tools by active servers
    tools = [t for t in all_tools if t.server_name in active_servers]

    # Format output
    output = [f"# Active Tools (profile: {mcp_config.active_profile})", ""]

    for server in active_servers:
        server_tools = [t for t in tools if t.server_name == server]
        output.append(f"## {server}")
        if server_tools:
            for tool in server_tools:
                # Truncate description to 100 chars
                desc = tool.text[:100] + "..." if len(tool.text) > 100 else tool.text
                output.append(f"- {tool.name}: {desc}")
        else:
            output.append("*No tools found*")
        output.append("")

    return "\n".join(output)
