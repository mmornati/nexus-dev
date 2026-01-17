"""Generate structured prompts using XML tags."""

from __future__ import annotations

from .agent_config import AgentConfig


class PromptFactory:
    """Build system prompts with XML structure for Claude/Gemini.

    This factory generates prompts using XML tags that are well-understood
    by modern LLMs like Claude and Gemini. The structure clearly separates:
    - Role definition (identity, goal, tone)
    - Backstory (expertise and background)
    - Memory (RAG context from the project)
    - Available tools
    - Instructions
    """

    @staticmethod
    def build(
        agent: AgentConfig,
        context_items: list[str],
        available_tools: list[str] | None = None,
    ) -> str:
        """Build the complete system prompt.

        Args:
            agent: Agent configuration.
            context_items: RAG search results (text snippets).
            available_tools: List of tool names the agent can use.

        Returns:
            Formatted system prompt with XML structure.
        """
        # Memory block from RAG
        memory_block = ""
        if context_items:
            items_str = "\n".join([f"- {item}" for item in context_items])
            memory_block = f"""
<nexus_memory>
Project context from RAG (use this to inform your responses):
{items_str}
</nexus_memory>
"""

        # Tools block
        tools_block = ""
        if available_tools:
            tools_str = ", ".join(available_tools)
            tools_block = f"""
<available_tools>
You can use these tools: {tools_str}
</available_tools>
"""

        return f"""<role_definition>
You are {agent.display_name}.
ROLE: {agent.profile.role}
OBJECTIVE: {agent.profile.goal}
TONE: {agent.profile.tone}
</role_definition>

<backstory>
{agent.profile.backstory}
</backstory>
{memory_block}{tools_block}
<instructions>
1. Analyze the user's request carefully.
2. Use your project context to provide accurate, project-specific responses.
3. If you need to perform actions, use the available tools.
4. Be concise but thorough.
</instructions>
"""
