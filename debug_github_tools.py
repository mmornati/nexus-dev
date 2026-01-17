import asyncio
import json
from nexus_dev.mcp_client import MCPClientManager, MCPServerConnection
from nexus_dev.mcp_config import MCPConfig
from pathlib import Path

async def list_github_tools():
    # Load config
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        print("Config not found")
        return

    mcp_config = MCPConfig.load(config_path)
    server_config = mcp_config.servers.get("github")
    
    if not server_config:
        print("GitHub server not found in config")
        return

    connection = MCPServerConnection(
        name="github",
        command=server_config.command or "",
        args=server_config.args,
        env=server_config.env,
        transport=server_config.transport,
        url=server_config.url,
        headers=server_config.headers,
        timeout=server_config.timeout,
    )

    client = MCPClientManager()
    try:
        tools = await client.get_tools(connection)
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            if tool.name == "list_pull_requests":
                print(f"Tool: {tool.name}")
                print(f"Schema: {tool.inputSchema}")
                break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(list_github_tools())
