"""System prompt tools for Nexus-Dev."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

GATEWAY_PROMPT_CONTENT = """## Gateway Tool Usage (MANDATORY)

When you need external MCP tools:
1. search_tools('') - Find available tools
 2. get_tool_schema(server, tool) - Get parameters
3. invoke_tool(server, tool, args) - Execute

Example:
1. search_tools('create GitHub issue')
2. get_tool_schema('github', 'create_issue')
3. invoke_tool(server='github', tool='create_issue', arguments={...})

NEVER call external tools directly!"""


async def get_gateway_prompt() -> str:
    """Get Gateway system prompt for inclusion in LLM context.

    Returns the gateway workflow instructions as formatted markdown
    that can be injected into a system prompt to ensure the LLM
    always has access to the proper tool usage workflow.

    Returns:
        Formatted markdown with gateway tool usage instructions.
    """
    return GATEWAY_PROMPT_CONTENT
