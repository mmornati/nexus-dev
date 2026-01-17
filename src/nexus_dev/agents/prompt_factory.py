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
CRITICAL RAG USAGE POLICY:
- You MUST use search_knowledge, search_code, search_docs, or search_lessons
  BEFORE answering ANY question about the project.
- Do NOT rely on your internal knowledge or the <nexus_memory> context alone
  when the user asks about specific implementations, configurations, or docs.
- If your first search yields no results, try:
  1. Broadening your search query
  2. Searching different content types (code vs docs vs lessons)
  3. Breaking down the question into smaller searchable parts
- Only after exhausting RAG searches should you answer based on general
  knowledge, and you must acknowledge that you couldn't find project-specific
  information.

WORKFLOW:
1. Analyze the user's request carefully.
2. If the request involves project-specific information, SEARCH FIRST using
   RAG tools.
3. Use your retrieved context and <nexus_memory> to provide accurate,
   project-specific responses.
4. If you need to perform actions, use the available tools.
5. Be concise but thorough.
</instructions>
"""
