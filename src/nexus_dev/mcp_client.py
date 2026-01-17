"""MCP Client for connecting to backend MCP servers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
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
    transport: str = "stdio"
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout: float = 30.0


class MCPClientManager:
    """Manages connections to multiple MCP servers."""

    async def get_tools(self, server: MCPServerConnection) -> list[MCPToolSchema]:
        """Get all tools from an MCP server.

        Args:
            server: Server connection config

        Returns:
            List of tool schemas
        """
        if server.transport == "http":
            if not server.url:
                raise ValueError(f"URL required for HTTP transport: {server.name}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    server.url,
                    headers=server.headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {},
                    },
                    timeout=server.timeout,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except Exception:
                    # Check for SSE-wrapped JSON (Github quirk)
                    text = response.text
                    if "data: " in text:
                        # Extract JSON from data lines
                        json_lines = []
                        for line in text.splitlines():
                            if line.startswith("data: "):
                                json_lines.append(line.replace("data: ", "", 1))

                        try:
                            data = json.loads("".join(json_lines))
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to parse SSE-wrapped JSON from {server.url}. "
                                f"Status: {response.status_code}. Body: {response.text[:200]}..."
                            ) from e
                    else:
                        raise RuntimeError(
                            f"Failed to decode JSON response from {server.url}. "
                            f"Status: {response.status_code}. Body: {response.text[:200]}..."
                        ) from None

                if "error" in data:
                    raise RuntimeError(f"JSON-RPC error: {data['error']}")

                schemas = []
                # Result structure: {"result": {"tools": [...]}}
                tools_data = data.get("result", {}).get("tools", [])
                for tool in tools_data:
                    schemas.append(
                        MCPToolSchema(
                            name=tool.get("name"),
                            description=tool.get("description", ""),
                            input_schema=tool.get("inputSchema", {}),
                        )
                    )
                return schemas

        elif server.transport == "sse":
            if not server.url:
                raise ValueError(f"URL required for SSE transport: {server.name}")

            transport_cm = sse_client(
                url=server.url,
                headers=server.headers or {},
            )
        else:
            # Expand environment variables if needed
            env = expand_env_vars(server.env) if server.env else None

            # Create server parameters
            server_params = StdioServerParameters(command=server.command, args=server.args, env=env)
            transport_cm = stdio_client(server_params)

        async with (
            transport_cm as (read, write),
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

    async def call_tool(
        self,
        server: MCPServerConnection,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call a tool on an MCP server.

        Args:
            server: Server connection config
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if server.transport == "http":
            if not server.url:
                raise ValueError(f"URL required for HTTP transport: {server.name}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    server.url,
                    headers=server.headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments or {},
                        },
                    },
                    timeout=server.timeout,
                )
                response.raise_for_status()

                try:
                    data = response.json()
                except Exception:
                    # Check for SSE-wrapped JSON (Github quirk)
                    text = response.text
                    if "data: " in text:
                        # Extract JSON from data lines
                        json_lines = []
                        for line in text.splitlines():
                            if line.startswith("data: "):
                                json_lines.append(line.replace("data: ", "", 1))

                        try:
                            data = json.loads("".join(json_lines))
                        except Exception as e:
                            # print(f"ERROR: Failed to parse SSE JSON. Body: {response.text}")
                            raise RuntimeError(
                                f"Failed to parse SSE-wrapped JSON from {server.url}. "
                                f"Status: {response.status_code}. Body: {response.text[:200]}..."
                            ) from e
                    else:
                        # print(f"ERROR: Failed to parse JSON. Body: {response.text}")
                        raise RuntimeError(
                            f"Failed to decode JSON response from {server.url}. "
                            f"Status: {response.status_code}. Body: {response.text[:200]}..."
                        ) from None

                if "error" in data:
                    raise RuntimeError(f"JSON-RPC error: {data['error']}")

                return data.get("result", {})

        elif server.transport == "sse":
            if not server.url:
                raise ValueError(f"URL required for SSE transport: {server.name}")

            transport_cm = sse_client(
                url=server.url,
                headers=server.headers or {},
            )
        else:
            # Expand environment variables if needed
            env = expand_env_vars(server.env) if server.env else None

            # Create server parameters
            server_params = StdioServerParameters(command=server.command, args=server.args, env=env)
            transport_cm = stdio_client(server_params)

        async with (
            transport_cm as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await session.call_tool(tool_name, arguments or {})


def expand_env_vars(env: dict[str, str]) -> dict[str, str]:
    """Expand ${VAR} patterns in environment dict.

    Args:
        env: Dictionary of environment variables with potential ${VAR} patterns

    Returns:
        Dictionary with expanded environment variables
    """
    result = {}
    pattern = re.compile(r"\$\{(\w+)\}")

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    for key, value in env.items():
        result[key] = pattern.sub(replacer, value)

    return result
