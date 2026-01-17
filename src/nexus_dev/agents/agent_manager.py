"""Load and manage agent configurations."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import yaml

from .agent_config import AgentConfig

logger = logging.getLogger(__name__)


class AgentManager:
    """Scan and load agent YAML configurations.

    This manager:
    - Scans the `agents/` directory in the project root
    - Loads and validates each YAML file as an AgentConfig
    - Provides access to loaded agents by name
    """

    def __init__(self, agents_dir: Path | None = None) -> None:
        """Initialize the agent manager.

        Args:
            agents_dir: Path to the agents directory. Defaults to `./agents/`.
        """
        self.agents_dir = agents_dir or Path.cwd() / "agents"
        self.agents: dict[str, AgentConfig] = {}
        self._load_agents()

    def _load_agents(self) -> None:
        """Load all valid agent configs from the agents directory.

        Invalid configs are logged as warnings but don't prevent other agents
        from loading.
        """
        if not self.agents_dir.exists():
            logger.debug("Agents directory not found: %s", self.agents_dir)
            return

        yaml_files = list(self.agents_dir.glob("*.yaml")) + list(self.agents_dir.glob("*.yml"))

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not data:
                    logger.warning("Empty agent file: %s", yaml_file.name)
                    continue

                agent = AgentConfig(**data)
                self.agents[agent.name] = agent
                logger.info("Loaded agent: %s from %s", agent.name, yaml_file.name)

            except yaml.YAMLError as e:
                logger.warning("Invalid YAML in %s: %s", yaml_file.name, e)
            except Exception as e:
                logger.warning("Failed to load agent %s: %s", yaml_file.name, e)

        logger.info("Loaded %d agents from %s", len(self.agents), self.agents_dir)

    def reload(self) -> None:
        """Reload agents from disk.

        This clears the current agents and reloads from the directory.
        """
        self.agents.clear()
        self._load_agents()

    def get_agent(self, name: str) -> AgentConfig | None:
        """Get agent by name.

        Args:
            name: The agent's internal name.

        Returns:
            AgentConfig if found, None otherwise.
        """
        return self.agents.get(name)

    def list_agents(self) -> list[AgentConfig]:
        """List all loaded agents.

        Returns:
            List of AgentConfig instances.
        """
        return list(self.agents.values())

    def __len__(self) -> int:
        """Return number of loaded agents."""
        return len(self.agents)

    def __iter__(self) -> Iterator[AgentConfig]:
        """Iterate over loaded agents."""
        return iter(self.agents.values())

    def __contains__(self, name: str) -> bool:
        """Check if an agent is loaded."""
        return name in self.agents
