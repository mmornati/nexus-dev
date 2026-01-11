"""MCP Configuration management for Nexus-Dev."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]


@dataclass
class MCPServerConfig:
    """Individual MCP server configuration."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPConfig:
    """Nexus-Dev MCP project configuration."""

    version: str
    servers: dict[str, MCPServerConfig]

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
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                enabled=cfg.get("enabled", True),
            )
            for name, cfg in data["servers"].items()
        }

        return cls(version=data["version"], servers=servers)

    def get_active_servers(self) -> list[MCPServerConfig]:
        """Get a list of enabled MCP server configurations.

        Returns:
            List of enabled MCPServerConfig instances.
        """
        return [s for s in self.servers.values() if s.enabled]

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
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                    "enabled": server.enabled,
                }
                for name, server in self.servers.items()
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
