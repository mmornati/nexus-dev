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

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MCPConfig:
        """Create MCPConfig from a validated dictionary."""
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
            for name, cfg in data.get("servers", {}).items()
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
            version=data.get("version", "1.0"),
            servers=servers,
            profiles=profiles,
            active_profile=active_profile,
            gateway=gateway,
        )

    def merge(self, other: MCPConfig) -> MCPConfig:
        """Merge another configuration into this one (other overrides this).

        Args:
            other: The configuration to merge on top of this one.

        Returns:
            A new MCPConfig instance with merged settings.
        """
        # Servers: Merge dictionaries (other overrides this)
        merged_servers = self.servers.copy()
        merged_servers.update(other.servers)

        # Profiles: Merge dictionaries (other overrides this)
        merged_profiles = self.profiles.copy()
        merged_profiles.update(other.profiles)

        # Gateway: Local overrides global settings
        # We take the local gateway settings as primary
        merged_gateway = other.gateway

        return MCPConfig(
            version=other.version,
            servers=merged_servers,
            profiles=merged_profiles,
            active_profile=other.active_profile,
            gateway=merged_gateway,
        )

    @classmethod
    def load_hierarchical(
        cls, global_path: Path | None, local_path: Path | None
    ) -> MCPConfig | None:
        """Load configuration hierarchically (Global + Local).

        Args:
            global_path: Path to global config (e.g., ~/.nexus/mcp_config.json).
            local_path: Path to local config (e.g., project/.nexus/mcp_config.json).

        Returns:
            Merged MCPConfig, or None if neither exists.
        """
        import logging

        logger = logging.getLogger(__name__)

        global_dict: dict[str, Any] | None = None
        local_dict: dict[str, Any] | None = None

        if global_path and global_path.exists():
            try:
                with open(global_path, encoding="utf-8") as f:
                    global_dict = json.load(f)
            except Exception as e:
                logger.warning("Failed to load global MCP config from %s: %s", global_path, e)

        if local_path and local_path.exists():
            try:
                with open(local_path, encoding="utf-8") as f:
                    local_dict = json.load(f)
            except Exception as e:
                logger.warning("Failed to load local MCP config from %s: %s", local_path, e)

        if not global_dict and not local_dict:
            return None

        # Merge dictionaries explicitly, prefer local options
        if global_dict and local_dict:
            merged = global_dict.copy()

            # Merge servers
            merged_servers = merged.get("servers", {}).copy()
            merged_servers.update(local_dict.get("servers", {}))
            merged["servers"] = merged_servers

            # Merge profiles (combine lists using distinct order-preserving insertion)
            merged_profiles = merged.get("profiles", {}).copy()
            for profile, servers in local_dict.get("profiles", {}).items():
                existing = merged_profiles.get(profile, [])
                # Add local servers to global servers avoiding duplicates
                merged_profiles[profile] = existing + [s for s in servers if s not in existing]
            merged["profiles"] = merged_profiles

            # Gateway settings
            merged_gateway = merged.get("gateway", {}).copy()
            merged_gateway.update(local_dict.get("gateway", {}))
            merged["gateway"] = merged_gateway

            # Apply local active profile if specified
            if "active_profile" in local_dict:
                merged["active_profile"] = local_dict["active_profile"]

            try:
                return cls._from_dict(merged)
            except Exception as e:
                logger.warning("Failed to parse merged MCP config: %s", e)
                return None

        elif global_dict:
            try:
                return cls._from_dict(global_dict)
            except Exception as e:
                logger.warning("Failed to parse global MCP config: %s", e)
                return None
        elif local_dict:
            try:
                return cls._from_dict(local_dict)
            except Exception as e:
                logger.warning("Failed to parse local MCP config: %s", e)
                return None

        return None

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
