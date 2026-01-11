"""Unit tests for MCP configuration."""

import json
from pathlib import Path

import pytest
from nexus_dev.mcp_config import MCPConfig, MCPServerConfig


@pytest.fixture
def valid_config_data():
    return {
        "version": "1.0",
        "servers": {
            "test-server": {
                "command": "python",
                "args": ["-m", "test_server"],
                "env": {"DEBUG": "true"},
                "enabled": True,
            }
        },
    }


def test_mcp_config_load_valid(tmp_path, valid_config_data):
    config_path = tmp_path / "mcp_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(valid_config_data, f)

    config = MCPConfig.load(config_path)
    assert config.version == "1.0"
    assert "test-server" in config.servers
    server = config.servers["test-server"]
    assert server.command == "python"
    assert server.args == ["-m", "test_server"]
    assert server.env == {"DEBUG": "true"}
    assert server.enabled is True


def test_mcp_config_invalid_version(tmp_path, valid_config_data):
    valid_config_data["version"] = "2.0"  # Invalid according to schema (const: "1.0")
    config_path = tmp_path / "mcp_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(valid_config_data, f)

    with pytest.raises(ValueError, match="Invalid MCP configuration"):
        MCPConfig.load(config_path)


def test_mcp_config_missing_command(tmp_path, valid_config_data):
    del valid_config_data["servers"]["test-server"]["command"]
    config_path = tmp_path / "mcp_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(valid_config_data, f)

    with pytest.raises(ValueError, match="Invalid MCP configuration"):
        MCPConfig.load(config_path)


def test_mcp_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        MCPConfig.load("non_existent_config.json")


def test_mcp_server_config_defaults():
    server = MCPServerConfig(command="ls")
    assert server.args == []
    assert server.env == {}
    assert server.enabled is True
