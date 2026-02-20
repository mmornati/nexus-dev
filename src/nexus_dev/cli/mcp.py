"""MCP related commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import click

from nexus_dev.cli.utils import requires_config
from nexus_dev.config import NexusConfig
from nexus_dev.database import Document, DocumentType, NexusDatabase
from nexus_dev.embeddings import create_embedder
from nexus_dev.mcp_client import MCPClientManager, MCPServerConnection
from nexus_dev.mcp_config import MCPConfig, MCPServerConfig


async def _index_mcp_servers(
    mcp_config: dict[str, Any] | MCPConfig,
    server_names: list[str],
    config: NexusConfig,
) -> None:
    """Index tools from specified MCP servers."""
    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    async with MCPClientManager() as client:
        for name in server_names:
            if isinstance(mcp_config, MCPConfig):
                server_config = mcp_config.servers.get(name)
                if not server_config:
                    click.echo(f"Server not found: {name}")
                    continue
                # Convert to internal connection format
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
            else:
                server_dict = mcp_config.get("mcpServers", {}).get(name)
                if not server_dict:
                    click.echo(f"Server not found: {name}")
                    continue
                connection = MCPServerConnection(
                    name=name,
                    command=server_dict.get("command", ""),
                    args=server_dict.get("args", []),
                    env=server_dict.get("env"),
                    transport=server_dict.get("transport", "stdio"),
                    url=server_dict.get("url"),
                    headers=server_dict.get("headers"),
                    timeout=server_dict.get("timeout", 30.0),
                )

            # Connect and index

            click.echo(f"Indexing tools from: {name}")

            try:
                tools = await client.get_tools(connection)
                click.echo(f"  Found {len(tools)} tools")

                # Create documents and index
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

                click.echo(f"  ✅ Indexed {len(tools)} tools from {name}")

            except Exception as e:
                # Handle ExceptionGroup from anyio/TaskGroup
                if hasattr(e, "exceptions"):
                    for sub_e in e.exceptions:
                        click.echo(f"  ❌ Failed to index {name}: {sub_e}")
                else:
                    click.echo(f"  ❌ Failed to index {name}: {e}")

    click.echo("Done!")


@click.command("index-mcp")
@click.option("--server", "-s", help="Server name to index (from MCP config)")
@click.option(
    "--config",
    "-c",
    "mcp_config_path",
    type=click.Path(exists=True),
    help="Path to MCP config file (default: ~/.config/mcp/config.json)",
)
@click.option("--all", "-a", "index_all", is_flag=True, help="Index all configured servers")
@requires_config
def index_mcp_command(
    config: NexusConfig,
    server: str | None,
    mcp_config_path: str | None,
    index_all: bool,
) -> None:
    """Index MCP tool documentation into the knowledge base.

    This command reads tool schemas from MCP servers and indexes them
    for semantic search via the search_tools command.

    Examples:
        nexus-index-mcp --server github
        nexus-index-mcp --all
        nexus-index-mcp --config ~/my-mcp-config.json --all
    """
    # Load MCP config
    mcp_config_data: dict[str, Any] | MCPConfig
    if mcp_config_path:
        config_path = Path(mcp_config_path)
    else:
        # Prioritize local project config
        local_config_path = Path.cwd() / ".nexus" / "mcp_config.json"
        if local_config_path.exists():
            config_path = local_config_path
        else:
            config_path = Path.home() / ".config" / "mcp" / "config.json"

    if not config_path.exists():
        click.echo(f"MCP config not found: {config_path}")
        click.echo("Specify --config or create ~/.config/mcp/config.json")
        return

    try:
        if config_path.name == "mcp_config.json" and config_path.parent.name == ".nexus":
            # Project-specific config
            mcp_config_data = MCPConfig.load(config_path)
        else:
            # Global config (or custom dict-based config)
            mcp_config_data = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON in MCP config: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"❌ Failed to load MCP config: {e}", err=True)
        return

    # Determine which servers to index
    servers_to_index = []
    if isinstance(mcp_config_data, MCPConfig):
        all_servers = list(mcp_config_data.servers.keys())
    else:
        all_servers = list(mcp_config_data.get("mcpServers", {}).keys())

    if index_all:
        servers_to_index = all_servers
    elif server:
        servers_to_index = [server]
    else:
        click.echo("Specify --server or --all")
        return

    # Index each server
    asyncio.run(_index_mcp_servers(mcp_config_data, servers_to_index, config))


@click.group("mcp")
def mcp_group() -> None:
    """Manage MCP server configurations."""


@mcp_group.command("init")
@click.option(
    "--from-global",
    is_flag=True,
    help="Import servers from ~/.config/mcp/config.json",
)
def mcp_init_command(from_global: bool) -> None:
    """Initialize MCP configuration for this project.

    Creates .nexus/mcp_config.json with an empty configuration
    or imports from your global MCP config.

    Examples:
        nexus-mcp init
        nexus-mcp init --from-global
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"

    if config_path.exists() and not click.confirm("MCP config exists. Overwrite?"):
        click.echo("Aborted.")
        return

    # Ensure .nexus directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if from_global:
        # Import from global config
        global_path = Path.home() / ".config" / "mcp" / "config.json"
        if not global_path.exists():
            click.echo(f"Global config not found: {global_path}")
            return

        try:
            global_config = json.loads(global_path.read_text())
        except json.JSONDecodeError as e:
            click.echo(f"❌ Invalid JSON in global config: {e}")
            return

        servers = {}

        for name, cfg in global_config.get("mcpServers", {}).items():
            servers[name] = MCPServerConfig(
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                enabled=True,
            )

        mcp_config = MCPConfig(
            version="1.0",
            servers=servers,
            profiles={},
        )
        click.echo(f"Imported {len(servers)} servers from global config")
    else:
        # Create empty config
        mcp_config = MCPConfig(
            version="1.0",
            servers={},
            profiles={"default": []},
        )

    mcp_config.save(config_path)
    click.echo(f"✅ Created {config_path}")
    click.echo("")
    click.echo("Configuration initialized successfully!")
    click.echo("You can manually edit the config file to add MCP servers.")


@mcp_group.command("add")
@click.argument("name")
@click.option("--command", "-c", required=True, help="Command to run MCP server")
@click.option("--args", "-a", multiple=True, help="Arguments for the command")
@click.option("--env", "-e", multiple=True, help="Environment vars (KEY=value or KEY=${VAR})")
@click.option("--profile", "-p", default="default", help="Add to profile (default: default)")
def mcp_add_command(
    name: str, command: str, args: tuple[str, ...], env: tuple[str, ...], profile: str
) -> None:
    """Add an MCP server to the configuration.

    Examples:
        nexus-mcp add github --command "npx" --args "-y" \\
            --args "@modelcontextprotocol/server-github"
        nexus-mcp add myserver --command "my-mcp" --env "API_KEY=${MY_API_KEY}"
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    # Parse environment variables
    env_dict = {}
    for e in env:
        if "=" in e:
            k, v = e.split("=", 1)
            env_dict[k] = v

    # Add server
    mcp_config.servers[name] = MCPServerConfig(
        command=command,
        args=list(args),
        env=env_dict,
        enabled=True,
    )

    # Add to profile
    if profile not in mcp_config.profiles:
        mcp_config.profiles[profile] = []
    if name not in mcp_config.profiles[profile]:
        mcp_config.profiles[profile].append(name)

    mcp_config.save(config_path)
    click.echo(f"Added {name} to profile '{profile}'")


@mcp_group.command("list")
@click.option(
    "--all", "-a", "show_all", is_flag=True, help="Show all servers, not just active profile"
)
def mcp_list_command(show_all: bool) -> None:
    """List configured MCP servers.

    Examples:
        nexus-mcp list
        nexus-mcp list --all
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("No MCP config. Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    click.echo(f"Active profile: {mcp_config.active_profile}")
    click.echo("")

    if show_all:
        click.echo("All servers:")
        servers_to_show = list(mcp_config.servers.items())
    else:
        click.echo("Active servers:")
        # Get active profile server names
        if mcp_config.active_profile in mcp_config.profiles:
            active_server_names = mcp_config.profiles[mcp_config.active_profile]
            # Filter to only enabled servers
            servers_to_show = [
                (name, mcp_config.servers[name])
                for name in active_server_names
                if name in mcp_config.servers and mcp_config.servers[name].enabled
            ]
        else:
            # If no active profile, show all enabled servers
            servers_to_show = [
                (name, server) for name, server in mcp_config.servers.items() if server.enabled
            ]

    for name, server in servers_to_show:
        status = "✓" if server.enabled else "✗"
        click.echo(f"  {status} {name}")
        click.echo(f"    Command: {server.command} {' '.join(server.args)}")
        if server.env:
            click.echo(f"    Env: {', '.join(server.env.keys())}")

    click.echo("")
    click.echo(f"Profiles: {', '.join(mcp_config.profiles.keys())}")


@mcp_group.command("profile")
@click.argument("name", required=False)
@click.option("--add", "-a", multiple=True, help="Add server to profile")
@click.option("--remove", "-r", multiple=True, help="Remove server from profile")
@click.option("--create", is_flag=True, help="Create new profile")
def mcp_profile_command(
    name: str | None, add: tuple[str, ...], remove: tuple[str, ...], create: bool
) -> None:
    """Manage MCP profiles.

    Without arguments, shows current profile. With name, switches to that profile.

    Examples:
        nexus-mcp profile              # Show current
        nexus-mcp profile dev          # Switch to 'dev'
        nexus-mcp profile dev --create # Create new 'dev' profile
        nexus-mcp profile default --add homeassistant
        nexus-mcp profile default --remove github
    """
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    if not name:
        # Show current profile
        click.echo(f"Active: {mcp_config.active_profile}")
        servers = mcp_config.profiles.get(mcp_config.active_profile, [])
        click.echo(f"Servers: {', '.join(servers) or '(none)'}")
        return

    if create:
        if name in mcp_config.profiles:
            click.echo(f"Profile '{name}' exists")
            return
        mcp_config.profiles[name] = []
        click.echo(f"Created profile: {name}")

    if name not in mcp_config.profiles:
        click.echo(f"Profile '{name}' not found")
        return

    # Add servers
    for server in add:
        if server not in mcp_config.profiles[name]:
            mcp_config.profiles[name].append(server)
            click.echo(f"Added {server} to {name}")
            # Warn if server not defined yet
            if server not in mcp_config.servers:
                click.echo(f"  ⚠️  Server '{server}' not defined. Add it with 'nexus-mcp add'")
        else:
            click.echo(f"Server {server} already in {name}")

    # Remove servers
    for server in remove:
        if server in mcp_config.profiles[name]:
            mcp_config.profiles[name].remove(server)
            click.echo(f"Removed {server} from {name}")

    # Switch profile
    if not add and not remove and not create:
        mcp_config.active_profile = name
        click.echo(f"Switched to profile: {name}")

    mcp_config.save(config_path)


def _set_server_enabled(name: str, enabled: bool) -> None:
    """Set server enabled status."""
    config_path = Path.cwd() / ".nexus" / "mcp_config.json"
    if not config_path.exists():
        click.echo("Run 'nexus-mcp init' first")
        return

    mcp_config = MCPConfig.load(config_path)

    if name not in mcp_config.servers:
        click.echo(f"Server not found: {name}")
        return

    mcp_config.servers[name].enabled = enabled
    mcp_config.save(config_path)

    status = "enabled" if enabled else "disabled"
    click.echo(f"{name}: {status}")


@mcp_group.command("enable")
@click.argument("name")
def mcp_enable_command(name: str) -> None:
    """Enable an MCP server."""
    _set_server_enabled(name, True)


@mcp_group.command("disable")
@click.argument("name")
def mcp_disable_command(name: str) -> None:
    """Disable an MCP server."""
    _set_server_enabled(name, False)
