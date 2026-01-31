import json

import pytest

from nexus_dev.mcp_config import GatewaySettings, MCPConfig, MCPServerConfig


@pytest.fixture
def global_config_data():
    return {
        "version": "1.0",
        "servers": {
            "global-server": {"command": "echo", "args": ["global"], "enabled": True},
            "override-me": {"command": "echo", "args": ["original"], "enabled": True},
        },
        "profiles": {"default": ["global-server"]},
        "gateway": {"max_concurrent_connections": 5},
    }


@pytest.fixture
def local_config_data():
    return {
        "version": "1.0",
        "servers": {
            "local-server": {"command": "echo", "args": ["local"], "enabled": True},
            "override-me": {"command": "echo", "args": ["overridden"], "enabled": True},
        },
        "profiles": {"local-profile": ["local-server"]},
        "active_profile": "local-profile",
        "gateway": {"max_concurrent_connections": 10},
    }


def test_mcp_config_merge(global_config_data, local_config_data):
    """Test merging local config into global config."""
    # Create instances manually to avoid file I/O for this unit test
    global_config = MCPConfig(
        version=global_config_data["version"],
        servers={
            name: MCPServerConfig(**cfg) for name, cfg in global_config_data["servers"].items()
        },
        profiles=global_config_data["profiles"],
        gateway=GatewaySettings(**global_config_data["gateway"]),
    )

    local_config = MCPConfig(
        version=local_config_data["version"],
        servers={
            name: MCPServerConfig(**cfg) for name, cfg in local_config_data["servers"].items()
        },
        profiles=local_config_data["profiles"],
        active_profile=local_config_data.get("active_profile", "default"),
        gateway=GatewaySettings(**local_config_data["gateway"]),
    )

    merged = global_config.merge(local_config)

    # 1. Check Servers Merging
    assert "global-server" in merged.servers
    assert "local-server" in merged.servers
    assert "override-me" in merged.servers

    # Check override behavior
    assert merged.servers["override-me"].args == ["overridden"]

    # 2. Check Profiles Merging
    assert "default" in merged.profiles
    assert "local-profile" in merged.profiles

    # 3. Check Active Profile (Local takes precedence)
    assert merged.active_profile == "local-profile"

    # 4. Check Gateway (Local takes precedence)
    assert merged.gateway.max_concurrent_connections == 10


def test_load_hierarchical_both_exist(tmp_path, global_config_data, local_config_data):
    """Test loading when both global and local files exist."""
    global_path = tmp_path / "global_mcp_config.json"
    local_path = tmp_path / "local_mcp_config.json"

    with open(global_path, "w") as f:
        json.dump(global_config_data, f)
    with open(local_path, "w") as f:
        json.dump(local_config_data, f)

    config = MCPConfig.load_hierarchical(global_path, local_path)

    assert config is not None
    assert "global-server" in config.servers
    assert "local-server" in config.servers
    assert config.gateway.max_concurrent_connections == 10


def test_load_hierarchical_only_global(tmp_path, global_config_data):
    """Test loading when only global file exists."""
    global_path = tmp_path / "global_mcp_config.json"
    local_path = tmp_path / "non_existent.json"

    with open(global_path, "w") as f:
        json.dump(global_config_data, f)

    config = MCPConfig.load_hierarchical(global_path, local_path)

    assert config is not None
    assert "global-server" in config.servers
    assert "local-server" not in config.servers
    assert config.gateway.max_concurrent_connections == 5


def test_load_hierarchical_only_local(tmp_path, local_config_data):
    """Test loading when only local file exists."""
    global_path = tmp_path / "non_existent.json"
    local_path = tmp_path / "local_mcp_config.json"

    with open(local_path, "w") as f:
        json.dump(local_config_data, f)

    config = MCPConfig.load_hierarchical(global_path, local_path)

    assert config is not None
    assert "global-server" not in config.servers
    assert "local-server" in config.servers
    assert config.gateway.max_concurrent_connections == 10


def test_load_hierarchical_none_exist(tmp_path):
    """Test loading when neither file exists."""
    global_path = tmp_path / "non_existent_1.json"
    local_path = tmp_path / "non_existent_2.json"

    config = MCPConfig.load_hierarchical(global_path, local_path)
    assert config is None
