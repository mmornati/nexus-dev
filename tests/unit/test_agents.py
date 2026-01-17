"""Unit tests for the agents package."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus_dev.agents import (
    AgentConfig,
    AgentExecutor,
    AgentManager,
    AgentMemory,
    AgentProfile,
    LLMConfig,
    PromptFactory,
)
from nexus_dev.database import SearchResult


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_mcp():
    mcp = MagicMock()
    ctx = MagicMock()
    ctx.session = AsyncMock()
    mcp.get_context.return_value = ctx
    return mcp


@pytest.fixture
def agent_config():
    return AgentConfig(
        name="test_agent",
        display_name="Test Agent",
        description="A test agent",
        profile=AgentProfile(
            role="Tester", goal="Test everything", backstory="Born in a unit test", tone="Serious"
        ),
        memory=AgentMemory(enabled=True, rag_limit=5, search_types=["code", "documentation"]),
        llm_config=LLMConfig(model_hint="gpt-4", max_tokens=1000, temperature=0.7),
        tools=["search_code"],
    )


def make_search_result(text="some context"):
    return SearchResult(
        id="1",
        text=text,
        score=0.9,
        project_id="test-project",
        file_path="src/test.py",
        doc_type="code",
        chunk_type="function",
        language="python",
        name="test_func",
        start_line=1,
        end_line=10,
    )


class TestAgentExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self, agent_config, mock_db, mock_mcp):
        # Setup mocks
        mock_db.search = AsyncMock(return_value=[make_search_result()])

        mock_mcp.get_context().session.create_message = AsyncMock(
            return_value=MagicMock(content=MagicMock(text="Agent response"))
        )

        executor = AgentExecutor(agent_config, mock_db, mock_mcp)
        response = await executor.execute("do something")

        assert response == "Agent response"
        mock_db.search.assert_called()
        mock_mcp.get_context().session.create_message.assert_called()

    @pytest.mark.asyncio
    async def test_execute_rag_disabled(self, agent_config, mock_db, mock_mcp):
        agent_config.memory.enabled = False

        mock_mcp.get_context().session.create_message = AsyncMock(
            return_value=MagicMock(content=MagicMock(text="Response without RAG"))
        )

        executor = AgentExecutor(agent_config, mock_db, mock_mcp)
        response = await executor.execute("do something")

        assert response == "Response without RAG"
        mock_db.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_sampling_unsupported(self, agent_config, mock_db, mock_mcp):
        """Test fallback to Insight Mode when sampling is not supported."""
        # 1. Setup mock RAG results so we have context to display
        mock_db.search = AsyncMock(
            return_value=[
                make_search_result(text="function hello_world() {\n  console.log('test');\n}")
            ]
        )

        # 2. Setup MCP to fail with "not supported"
        mock_mcp.get_context().session.create_message = AsyncMock(
            side_effect=Exception("client does not support CreateMessage")
        )

        executor = AgentExecutor(agent_config, mock_db, mock_mcp)
        response = await executor.execute("do something")

        # 3. Verify the response is the formatted markdown fallback
        assert "# Agent: Test Agent (Insight Mode)" in response
        # We expect 2 items because the agent config has 2 search_types ("code", "documentation")
        # and our mock returns results for every call.
        assert "Found 2 relevant items" in response
        assert "Recommended System Prompt" in response
        # Verify context content is present
        assert "hello_world" in response

    @pytest.mark.asyncio
    async def test_execute_general_failure(self, agent_config, mock_db, mock_mcp):
        mock_db.search = AsyncMock(side_effect=Exception("Database error"))

        executor = AgentExecutor(agent_config, mock_db, mock_mcp)

        mock_mcp.get_context().session.create_message = AsyncMock(
            side_effect=Exception("Sampling failed")
        )
        response = await executor.execute("do something")
        assert "Agent execution failed" in response


class TestAgentManager:
    def test_manager_init_and_scan(self, tmp_path):
        # Create dummy agent files
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent_content = """
name: test_agent
display_name: "Test Agent"
description: "A test agent"
profile:
  role: "Tester"
  goal: "Test"
  backstory: "Born in test"
memory:
  enabled: true
llm_config:
  model_hint: "gpt-4"
"""
        (agents_dir / "test.yaml").write_text(agent_content)
        (agents_dir / "not_agent.txt").write_text("random")

        manager = AgentManager(agents_dir=agents_dir)
        assert len(manager) == 1
        agent_names = [a.name for a in manager.list_agents()]
        assert "test_agent" in agent_names

    def test_get_agent(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        agent_content = """
name: test_agent
display_name: "Test"
description: "Test"
profile:
  role: "Tester"
  goal: "Test"
  backstory: "Born in test"
"""
        (agents_dir / "test.yaml").write_text(agent_content)

        manager = AgentManager(agents_dir=agents_dir)
        agent = manager.get_agent("test_agent")
        assert agent is not None
        assert agent.name == "test_agent"

        assert manager.get_agent("nonexistent") is None

    def test_manager_no_dir(self, tmp_path):
        # Use a non-existent path in tmp_path to avoid picking up project defaults
        non_existent = tmp_path / "non_existent_agents"
        manager = AgentManager(agents_dir=non_existent)
        assert len(manager) == 0


class TestPromptFactory:
    def test_build_prompt(self, agent_config):
        context = ["file1 content", "file2 content"]
        tools = ["search_code", "index_file"]

        prompt = PromptFactory.build(agent_config, context, tools)

        assert "<role_definition>" in prompt
        assert "Tester" in prompt
        assert "<nexus_memory>" in prompt
        assert "file1 content" in prompt
        assert "<available_tools>" in prompt
        assert "search_code" in prompt
