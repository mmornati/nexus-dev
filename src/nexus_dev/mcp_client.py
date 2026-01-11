"""MCP Client for connecting to backend MCP servers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@dataclass
class MCPToolSchema:
    """Schema for an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPServerConnection:
    """Connection to an MCP server."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None


class MCPClientManager:
    """Manages connections to multiple MCP servers."""

    async def get_tools(self, server: MCPServerConnection) -> list[MCPToolSchema]:
        """Get all tools from an MCP server.

        Args:
            server: Server connection config

        Returns:
            List of tool schemas
        """
        # Expand environment variables if needed
        env = expand_env_vars(server.env) if server.env else None

        # Create server parameters
        server_params = StdioServerParameters(command=server.command, args=server.args, env=env)

        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()

            # List tools
            tools_result = await session.list_tools()

            schemas = []
            for tool in tools_result.tools:
                schemas.append(
                    MCPToolSchema(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                    )
                )

            return schemas

    async def get_tool_schema(
        self, server: MCPServerConnection, tool_name: str
    ) -> MCPToolSchema | None:
        """Get schema for a specific tool.

        Note: The MCP protocol doesn't support fetching individual tool schemas,
        so this method fetches all tools and filters locally. For servers with
        many tools, consider calling get_tools() once and caching the results.

        Args:
            server: Server connection config
            tool_name: Name of the tool to get schema for

        Returns:
            Tool schema if found, None otherwise
        """
        tools = await self.get_tools(server)
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None


def expand_env_vars(env: dict[str, str]) -> dict[str, str]:
    """Expand ${VAR} patterns in environment dict.

    Args:
        env: Dictionary of environment variables with potential ${VAR} patterns

    Returns:
        Dictionary with expanded environment variables
    """
    result = {}
    pattern = re.compile(r"\$\{(\w+)\}")

    for key, value in env.items():

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        result[key] = pattern.sub(replacer, value)

    return result
