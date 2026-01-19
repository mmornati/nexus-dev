"""MCP Configuration management for Nexus-Dev."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jsonschema  # type: ignore[import-untyped]


@dataclass
class MCPServerConfig:
    """Individual MCP server configuration."""

    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    transport: Literal["stdio", "sse", "http"] = "stdio"
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0  # Tool execution timeout
    connect_timeout: float = 10.0  # Connection timeout
    max_concurrent: int | None = None  # Max concurrent invocations (None = use gateway default)


@dataclass
class GatewaySettings:
    """Global gateway configuration settings."""

    default_timeout: float = 30.0
    max_concurrent_connections: int = 5
    shutdown_timeout: float = 5.0  # Graceful shutdown timeout


@dataclass
class MCPConfig:
    """Nexus-Dev MCP project configuration."""

    version: str
    servers: dict[str, MCPServerConfig]
    profiles: dict[str, list[str]] = field(default_factory=dict)
    active_profile: str = "default"
    gateway: GatewaySettings = field(default_factory=GatewaySettings)

    @classmethod
    def load(cls, path: str | Path) -> MCPConfig:
        """Load and validate configuration from a JSON file.

        Args:
            path: Path to the configuration file.

        Returns:
            Validated MCPConfig instance.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            ValueError: If configuration is invalid against the schema.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MCP configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        cls.validate(data)

        servers = {
            name: MCPServerConfig(
                command=cfg.get("command"),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                enabled=cfg.get("enabled", True),
                transport=cfg.get("transport", "stdio"),
                url=cfg.get("url"),
                headers=cfg.get("headers", {}),
                timeout=cfg.get("timeout", 30.0),
                connect_timeout=cfg.get("connect_timeout", 10.0),
                max_concurrent=cfg.get("max_concurrent"),
            )
            for name, cfg in data["servers"].items()
        }

        profiles = data.get("profiles", {})
        active_profile = data.get("active_profile", "default")

        gateway_data = data.get("gateway", {})
        gateway = GatewaySettings(
            default_timeout=gateway_data.get("default_timeout", 30.0),
            max_concurrent_connections=gateway_data.get("max_concurrent_connections", 5),
            shutdown_timeout=gateway_data.get("shutdown_timeout", 5.0),
        )

        return cls(
            version=data["version"],
            servers=servers,
            profiles=profiles,
            active_profile=active_profile,
            gateway=gateway,
        )

    def get_active_servers(self) -> list[MCPServerConfig]:
        """Get a list of enabled MCP server configurations in the active profile.

        Returns:
            List of enabled MCPServerConfig instances from the active profile.
        """
        # If active profile doesn't exist or is empty, return all enabled servers
        if self.active_profile not in self.profiles:
            return [s for s in self.servers.values() if s.enabled]

        # Get servers in active profile
        profile_server_names = self.profiles[self.active_profile]
        active_servers = []

        for name in profile_server_names:
            if name in self.servers:
                server = self.servers[name]
                if server.enabled:
                    active_servers.append(server)

        return active_servers

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the configuration file.
        """
        path = Path(path)

        # Convert to dictionary format
        data = {
            "version": self.version,
            "servers": {
                name: {
                    k: v
                    for k, v in {
                        "command": server.command,
                        "args": server.args,
                        "env": server.env,
                        "enabled": server.enabled,
                        "transport": server.transport,
                        "url": server.url,
                        "headers": server.headers,
                        "timeout": server.timeout,
                        "connect_timeout": server.connect_timeout,
                        "max_concurrent": server.max_concurrent,
                    }.items()
                    if v is not None
                }
                for name, server in self.servers.items()
            },
            "profiles": self.profiles,
            "active_profile": self.active_profile,
            "gateway": {
                "default_timeout": self.gateway.default_timeout,
                "max_concurrent_connections": self.gateway.max_concurrent_connections,
                "shutdown_timeout": self.gateway.shutdown_timeout,
            },
        }

        # Validate before saving
        self.validate(data)

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def validate(data: dict[str, Any]) -> None:
        """Validate configuration data against the JSON schema.

        Args:
            data: Configuration data dictionary.

        Raises:
            ValueError: If configuration is invalid.
        """
        schema_path = Path(__file__).parent / "schemas" / "mcp_config_schema.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid MCP configuration: {e.message}") from e
