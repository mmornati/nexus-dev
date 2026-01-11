"""Unit tests for MCP configuration."""

import json

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


def test_mcp_config_get_active_servers(valid_config_data):
    # Add a disabled server
    valid_config_data["servers"]["disabled-server"] = {
        "command": "echo",
        "enabled": False,
    }
    config = MCPConfig(
        version=valid_config_data["version"],
        servers={
            name: MCPServerConfig(**cfg) for name, cfg in valid_config_data["servers"].items()
        },
    )

    active_servers = config.get_active_servers()
    assert len(active_servers) == 1
    assert active_servers[0].command == "python"


def test_mcp_config_save(tmp_path, valid_config_data):
    """Test saving MCP configuration to a file."""
    config = MCPConfig(
        version=valid_config_data["version"],
        servers={
            name: MCPServerConfig(**cfg) for name, cfg in valid_config_data["servers"].items()
        },
    )

    config_path = tmp_path / "saved_config.json"
    config.save(config_path)

    assert config_path.exists()

    # Verify saved content
    with open(config_path, encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["version"] == "1.0"
    assert "test-server" in saved_data["servers"]
    assert saved_data["servers"]["test-server"]["command"] == "python"
    assert saved_data["servers"]["test-server"]["args"] == ["-m", "test_server"]
    assert saved_data["servers"]["test-server"]["env"] == {"DEBUG": "true"}
    assert saved_data["servers"]["test-server"]["enabled"] is True


def test_mcp_config_save_empty(tmp_path):
    """Test saving an empty MCP configuration."""
    config = MCPConfig(version="1.0", servers={})

    config_path = tmp_path / "empty_config.json"
    config.save(config_path)

    assert config_path.exists()

    # Verify saved content
    with open(config_path, encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["version"] == "1.0"
    assert saved_data["servers"] == {}


def test_mcp_config_save_and_load_roundtrip(tmp_path, valid_config_data):
    """Test that save and load are consistent (roundtrip)."""
    config = MCPConfig(
        version=valid_config_data["version"],
        servers={
            name: MCPServerConfig(**cfg) for name, cfg in valid_config_data["servers"].items()
        },
    )

    config_path = tmp_path / "roundtrip_config.json"
    config.save(config_path)

    # Load it back
    loaded_config = MCPConfig.load(config_path)

    assert loaded_config.version == config.version
    assert len(loaded_config.servers) == len(config.servers)
    for name, server in config.servers.items():
        assert name in loaded_config.servers
        loaded_server = loaded_config.servers[name]
        assert loaded_server.command == server.command
        assert loaded_server.args == server.args
        assert loaded_server.env == server.env
        assert loaded_server.enabled == server.enabled


def test_mcp_config_profiles():
    """Test that profiles can be added and retrieved."""
    config = MCPConfig(
        version="1.0",
        servers={
            "server1": MCPServerConfig(command="cmd1"),
            "server2": MCPServerConfig(command="cmd2"),
        },
        profiles={
            "default": ["server1", "server2"],
            "dev": ["server1"],
        },
    )

    assert "default" in config.profiles
    assert "dev" in config.profiles
    assert config.profiles["default"] == ["server1", "server2"]
    assert config.profiles["dev"] == ["server1"]


def test_mcp_config_save_and_load_with_profiles(tmp_path):
    """Test that profiles are saved and loaded correctly."""
    config = MCPConfig(
        version="1.0",
        servers={
            "github": MCPServerConfig(command="npx", args=["-y", "github-server"]),
            "gitlab": MCPServerConfig(command="npx", args=["-y", "gitlab-server"]),
        },
        profiles={
            "default": ["github"],
            "all": ["github", "gitlab"],
        },
    )

    config_path = tmp_path / "config_with_profiles.json"
    config.save(config_path)

    # Load it back
    loaded_config = MCPConfig.load(config_path)

    assert loaded_config.profiles == config.profiles
    assert loaded_config.profiles["default"] == ["github"]
    assert loaded_config.profiles["all"] == ["github", "gitlab"]


def test_mcp_config_active_profile_default():
    """Test that active_profile defaults to 'default'."""
    config = MCPConfig(
        version="1.0",
        servers={"server1": MCPServerConfig(command="cmd1")},
    )
    assert config.active_profile == "default"


def test_mcp_config_active_profile_custom():
    """Test setting custom active_profile."""
    config = MCPConfig(
        version="1.0",
        servers={"server1": MCPServerConfig(command="cmd1")},
        profiles={"dev": ["server1"]},
        active_profile="dev",
    )
    assert config.active_profile == "dev"


def test_mcp_config_save_and_load_with_active_profile(tmp_path):
    """Test that active_profile is saved and loaded correctly."""
    config = MCPConfig(
        version="1.0",
        servers={
            "github": MCPServerConfig(command="npx", args=["-y", "github-server"]),
            "gitlab": MCPServerConfig(command="npx", args=["-y", "gitlab-server"]),
        },
        profiles={
            "default": ["github"],
            "all": ["github", "gitlab"],
        },
        active_profile="all",
    )

    config_path = tmp_path / "config_with_active_profile.json"
    config.save(config_path)

    # Load it back
    loaded_config = MCPConfig.load(config_path)

    assert loaded_config.active_profile == "all"


def test_mcp_config_get_active_servers_with_profile():
    """Test get_active_servers returns servers from active profile."""
    config = MCPConfig(
        version="1.0",
        servers={
            "github": MCPServerConfig(command="npx", enabled=True),
            "gitlab": MCPServerConfig(command="npx", enabled=True),
            "disabled": MCPServerConfig(command="echo", enabled=False),
        },
        profiles={
            "default": ["github"],
            "all": ["github", "gitlab", "disabled"],
        },
        active_profile="default",
    )

    active_servers = config.get_active_servers()
    assert len(active_servers) == 1
    assert active_servers[0].command == "npx"


def test_mcp_config_get_active_servers_filters_disabled_in_profile():
    """Test get_active_servers filters out disabled servers even in profile."""
    config = MCPConfig(
        version="1.0",
        servers={
            "github": MCPServerConfig(command="npx", enabled=True),
            "gitlab": MCPServerConfig(command="npx", enabled=False),
        },
        profiles={
            "all": ["github", "gitlab"],
        },
        active_profile="all",
    )

    active_servers = config.get_active_servers()
    assert len(active_servers) == 1
    assert active_servers[0].command == "npx"


def test_mcp_config_get_active_servers_no_profile():
    """Test get_active_servers returns all enabled when profile doesn't exist."""
    config = MCPConfig(
        version="1.0",
        servers={
            "github": MCPServerConfig(command="npx", enabled=True),
            "gitlab": MCPServerConfig(command="npx", enabled=True),
            "disabled": MCPServerConfig(command="echo", enabled=False),
        },
        profiles={},
        active_profile="nonexistent",
    )

    active_servers = config.get_active_servers()
    assert len(active_servers) == 2  # Should return all enabled servers
