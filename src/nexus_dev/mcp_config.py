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
    cache_enabled: bool | None = None  # Override gateway cache setting (None = use gateway default)
    summarize: SummarizeSettings | None = None  # Override gateway summarize setting


@dataclass
class CacheSettings:
    """Cache configuration for tool invocation results."""

    enabled: bool = True
    ttl_seconds: float = 300.0  # 5 minutes
    max_entries: int = 1000


@dataclass
class SummarizeSettings:
    """Summarization configuration for tool outputs."""

    enabled: bool = True
    max_list_items: int = 10
    max_output_chars: int = 500


@dataclass
class GatewaySettings:
    """Global gateway configuration settings."""

    default_timeout: float = 30.0
    max_concurrent_connections: int = 5
    shutdown_timeout: float = 5.0  # Graceful shutdown timeout
    cache: CacheSettings = field(default_factory=CacheSettings)
    summarize: SummarizeSettings = field(default_factory=SummarizeSettings)


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

        def _load_summarize_settings(data: dict[str, Any] | None) -> SummarizeSettings | None:
            """Load summarize settings from dict."""
            if data is None:
                return None
            return SummarizeSettings(
                enabled=data.get("enabled", True),
                max_list_items=data.get("max_list_items", 10),
                max_output_chars=data.get("max_output_chars", 500),
            )

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
                cache_enabled=cfg.get("cache_enabled"),
                summarize=_load_summarize_settings(cfg.get("summarize")),
            )
            for name, cfg in data["servers"].items()
        }

        profiles = data.get("profiles", {})
        active_profile = data.get("active_profile", "default")

        gateway_data = data.get("gateway", {})
        gateway_cache_data = gateway_data.get("cache", {})
        gateway_cache = CacheSettings(
            enabled=gateway_cache_data.get("enabled", True),
            ttl_seconds=gateway_cache_data.get("ttl_seconds", 300.0),
            max_entries=gateway_cache_data.get("max_entries", 1000),
        )
        gateway_summarize_data = gateway_data.get("summarize", {})
        gateway_summarize = SummarizeSettings(
            enabled=gateway_summarize_data.get("enabled", True),
            max_list_items=gateway_summarize_data.get("max_list_items", 10),
            max_output_chars=gateway_summarize_data.get("max_output_chars", 500),
        )
        gateway = GatewaySettings(
            default_timeout=gateway_data.get("default_timeout", 30.0),
            max_concurrent_connections=gateway_data.get("max_concurrent_connections", 5),
            shutdown_timeout=gateway_data.get("shutdown_timeout", 5.0),
            cache=gateway_cache,
            summarize=gateway_summarize,
        )

        return cls(
            version=data["version"],
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
        global_config: MCPConfig | None = None
        local_config: MCPConfig | None = None

        if global_path and global_path.exists():
            try:
                global_config = cls.load(global_path)
            except Exception:
                # In a real CLI context we might want to warn, but here we proceed
                pass

        if local_path and local_path.exists():
            try:
                local_config = cls.load(local_path)
            except Exception:
                pass

        if global_config and local_config:
            return global_config.merge(local_config)
        elif global_config:
            return global_config
        elif local_config:
            return local_config

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

        def _serialize_summarize(s: SummarizeSettings | None) -> dict[str, Any] | None:
            if s is None:
                return None
            return {
                "enabled": s.enabled,
                "max_list_items": s.max_list_items,
                "max_output_chars": s.max_output_chars,
            }

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
                        "cache_enabled": server.cache_enabled,
                        "summarize": _serialize_summarize(server.summarize),
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
                "cache": {
                    "enabled": self.gateway.cache.enabled,
                    "ttl_seconds": self.gateway.cache.ttl_seconds,
                    "max_entries": self.gateway.cache.max_entries,
                },
                "summarize": {
                    "enabled": self.gateway.summarize.enabled,
                    "max_list_items": self.gateway.summarize.max_list_items,
                    "max_output_chars": self.gateway.summarize.max_output_chars,
                },
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
