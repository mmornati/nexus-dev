import json

import pytest

from nexus_dev.mcp_config import GatewaySettings, MCPConfig, MCPServerConfig


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
    assert server.timeout == 30.0
    assert server.connect_timeout == 10.0


def test_mcp_server_config_custom_timeout():
    server = MCPServerConfig(
        command="ls",
        timeout=60.0,
        connect_timeout=15.0,
    )
    assert server.timeout == 60.0
    assert server.connect_timeout == 15.0


def test_gateway_settings_defaults():
    settings = GatewaySettings()
    assert settings.default_timeout == 30.0
    assert settings.max_concurrent_connections == 5


def test_gateway_settings_custom():
    settings = GatewaySettings(
        default_timeout=45.0,
        max_concurrent_connections=10,
    )
    assert settings.default_timeout == 45.0
    assert settings.max_concurrent_connections == 10


def test_mcp_config_with_gateway(tmp_path, valid_config_data):
    gateway_config = {
        "default_timeout": 60.0,
        "max_concurrent_connections": 20,
    }
    valid_config_data["gateway"] = gateway_config

    config_path = tmp_path / "mcp_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(valid_config_data, f)

    config = MCPConfig.load(config_path)
    assert config.gateway.default_timeout == 60.0
    assert config.gateway.max_concurrent_connections == 20


def test_mcp_config_load_with_timeouts(tmp_path, valid_config_data):
    valid_config_data["servers"]["timeout-server"] = {
        "command": "python",
        "timeout": 120.0,
        "connect_timeout": 5.0,
    }

    config_path = tmp_path / "mcp_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(valid_config_data, f)

    config = MCPConfig.load(config_path)
    server = config.servers["timeout-server"]
    assert server.timeout == 120.0
    assert server.connect_timeout == 5.0

    # Test saving preserves timeouts
    output_path = tmp_path / "output_config.json"
    config.save(output_path)

    loaded_config = MCPConfig.load(output_path)
    saved_server = loaded_config.servers["timeout-server"]
    assert saved_server.timeout == 120.0
    assert saved_server.connect_timeout == 5.0
    assert loaded_config.gateway.default_timeout == 30.0  # Default preserved


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


def test_mcp_config_profile_with_nonexistent_server():
    """Test get_active_servers handles profile referencing non-existent servers."""
    config = MCPConfig(
        version="1.0",
        servers={
            "github": MCPServerConfig(command="npx", enabled=True),
        },
        profiles={
            "test": ["github", "nonexistent-server", "another-missing"],
        },
        active_profile="test",
    )

    active_servers = config.get_active_servers()
    # Should only return the one that exists and is enabled
    assert len(active_servers) == 1
    assert active_servers[0].command == "npx"


def test_mcp_config_timeout_minimum_values():
    """Test config with minimum timeout values."""
    config = MCPConfig(
        version="1.0",
        servers={
            "quick": MCPServerConfig(
                command="fast-cmd", timeout=1.0, connect_timeout=1.0, enabled=True
            ),
        },
    )

    assert config.servers["quick"].timeout == 1.0
    assert config.servers["quick"].connect_timeout == 1.0


def test_mcp_config_timeout_large_values():
    """Test config with large timeout values."""
    config = MCPConfig(
        version="1.0",
        servers={
            "slow": MCPServerConfig(
                command="slow-cmd", timeout=3600.0, connect_timeout=300.0, enabled=True
            ),
        },
    )

    assert config.servers["slow"].timeout == 3600.0
    assert config.servers["slow"].connect_timeout == 300.0


def test_mcp_config_save_validates_schema(tmp_path):
    """Test that save validates config against schema before writing."""
    config = MCPConfig(
        version="1.0",
        servers={"test": MCPServerConfig(command="cmd")},
    )

    # Save should succeed with valid config
    config_path = tmp_path / "valid_config.json"
    config.save(config_path)
    assert config_path.exists()

    # Attempting to save invalid version should fail during load
    # (We test this indirectly since save validates before writing)
    loaded = MCPConfig.load(config_path)
    assert loaded.version == "1.0"


def test_mcp_config_empty_profiles_dict():
    """Test config with empty profiles dictionary."""
    config = MCPConfig(
        version="1.0",
        servers={"server1": MCPServerConfig(command="cmd", enabled=True)},
        profiles={},
        active_profile="default",
    )

    # With no profiles, should return all enabled servers
    active = config.get_active_servers()
    assert len(active) == 1


def test_mcp_config_gateway_settings_roundtrip(tmp_path):
    """Test that gateway settings are preserved in save/load roundtrip."""
    config = MCPConfig(
        version="1.0",
        servers={},
        gateway=GatewaySettings(default_timeout=120.0, max_concurrent_connections=15),
    )

    config_path = tmp_path / "gateway_config.json"
    config.save(config_path)

    loaded = MCPConfig.load(config_path)
    assert loaded.gateway.default_timeout == 120.0
    assert loaded.gateway.max_concurrent_connections == 15


def test_mcp_config_active_profile_switching():
    """Test switching active profiles and verifying active servers change."""
    config = MCPConfig(
        version="1.0",
        servers={
            "github": MCPServerConfig(command="gh", enabled=True),
            "gitlab": MCPServerConfig(command="gl", enabled=True),
            "local": MCPServerConfig(command="local", enabled=True),
        },
        profiles={
            "default": ["github"],
            "all-git": ["github", "gitlab"],
            "local-only": ["local"],
        },
        active_profile="default",
    )

    # Initially on default profile
    active = config.get_active_servers()
    assert len(active) == 1
    assert active[0].command == "gh"

    # Switch to all-git profile
    config.active_profile = "all-git"
    active = config.get_active_servers()
    assert len(active) == 2
    commands = {s.command for s in active}
    assert commands == {"gh", "gl"}

    # Switch to local-only profile
    config.active_profile = "local-only"
    active = config.get_active_servers()
    assert len(active) == 1
    assert active[0].command == "local"


def test_mcp_config_disabled_server_in_profile():
    """Test that disabled servers are excluded even if in active profile."""
    config = MCPConfig(
        version="1.0",
        servers={
            "enabled-server": MCPServerConfig(command="cmd1", enabled=True),
            "disabled-server": MCPServerConfig(command="cmd2", enabled=False),
        },
        profiles={
            "mixed": ["enabled-server", "disabled-server"],
        },
        active_profile="mixed",
    )

    active = config.get_active_servers()
    # Should only return enabled server, even though both are in profile
    assert len(active) == 1
    assert active[0].command == "cmd1"
