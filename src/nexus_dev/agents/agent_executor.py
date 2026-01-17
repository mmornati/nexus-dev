"""Execute agent tasks using MCP Sampling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mcp.types import (
    CreateMessageRequestParams,
    ModelHint,
    ModelPreferences,
    SamplingMessage,
    TextContent,
)

from .agent_config import AgentConfig
from .prompt_factory import PromptFactory

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ..database import NexusDatabase

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Orchestrate agent execution with RAG and MCP Sampling.

    This executor:
    1. Retrieves relevant context from the project's LanceDB (RAG)
    2. Builds a structured system prompt using PromptFactory
    3. Sends a sampling request to the IDE via MCP
    4. Returns the IDE's response
    """

    def __init__(
        self,
        config: AgentConfig,
        database: NexusDatabase,
        mcp_server: FastMCP,
    ) -> None:
        """Initialize the agent executor.

        Args:
            config: Agent configuration from YAML.
            database: NexusDatabase instance for RAG search.
            mcp_server: FastMCP server instance for sampling requests.
        """
        self.config = config
        self.database = database
        self.mcp_server = mcp_server

    async def execute(self, user_task: str, project_id: str | None = None) -> str:
        """Execute a task with the configured agent.

        The execution flow:
        1. Retrieve relevant context from RAG based on the task
        2. Build system prompt with agent persona and context
        3. Send sampling request to IDE
        4. Return the response

        Args:
            user_task: The task description from the user.
            project_id: Optional project ID for RAG filtering.

        Returns:
            The agent's response text.
        """
        from ..database import DocumentType

        # 1. RAG Retrieval
        context_items: list[str] = []
        if self.config.memory.enabled:
            for search_type in self.config.memory.search_types:
                try:
                    doc_type = DocumentType(search_type)
                    results = await self.database.search(
                        query=user_task,
                        project_id=project_id,
                        doc_type=doc_type,
                        limit=self.config.memory.rag_limit,
                    )
                    # Truncate each result to avoid context overflow
                    context_items.extend([r.text[:500] for r in results])
                except Exception as e:
                    logger.warning("RAG search failed for %s: %s", search_type, e)

        logger.debug(
            "Agent %s retrieved %d context items for task: %s",
            self.config.name,
            len(context_items),
            user_task[:100],
        )

        # 2. Build Prompt
        system_prompt = PromptFactory.build(
            agent=self.config,
            context_items=context_items,
            available_tools=self.config.tools if self.config.tools else None,
        )

        # 3. MCP Sampling Request
        try:
            # Create the sampling request parameters
            model_prefs = ModelPreferences(
                hints=[ModelHint(name=self.config.llm_config.model_hint)],
            )

            request_params = CreateMessageRequestParams(
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text=user_task),
                    )
                ],
                systemPrompt=system_prompt,
                modelPreferences=model_prefs,
                maxTokens=self.config.llm_config.max_tokens,
            )

            # Access the session from the request context
            # Note: This requires the server to be in a request context
            ctx = self.mcp_server.get_context()
            result = await ctx.session.create_message(
                messages=request_params.messages,
                system_prompt=request_params.systemPrompt,
                model_preferences=request_params.modelPreferences,
                max_tokens=request_params.maxTokens,
            )

            # 4. Extract and return response text
            if hasattr(result.content, "text"):
                return str(result.content.text)
            return str(result.content)

        except Exception as e:
            error_msg = str(e)
            if "does not support CreateMessage" in error_msg or "Method not found" in error_msg:
                logger.warning(
                    "MCP Sampling not supported by client. Returning context-only response."
                )
                return self._create_fallback_response(user_task, context_items, system_prompt)

            logger.error("MCP Sampling failed for agent %s: %s", self.config.name, e)
            return f"Agent execution failed: {e}"

    def _create_fallback_response(
        self, user_task: str, context_items: list[str], system_prompt: str
    ) -> str:
        """Create a Markdown response when full agent execution is not supported.

        Args:
            user_task: The original task.
            context_items: List of retrieved context strings.
            system_prompt: The constructed system prompt.

        Returns:
            Formatted Markdown string.
        """
        response = [
            f"# Agent: {self.config.display_name} (Insight Mode)",
            "",
            "> **Note**: Your IDE does not support fully autonomous agent "
            "execution (MCP Sampling).",
            "> I have retrieved the relevant context and prepared the prompt for you below.",
            "",
            "## 1. Retrieved Context",
            f"Found {len(context_items)} relevant items in the knowledge base:",
            "",
        ]

        if context_items:
            for i, item in enumerate(context_items, 1):
                # Try to extract a clean title or first line
                preview = item.split("\n")[0][:80]
                response.append(f"**Item {i}**: `{preview}...`")
                # Indent the content for better readability in the details block
                response.append("<details>")
                response.append(f"<summary>View Content</summary>\n\n{item}\n")
                response.append("</details>\n")
        else:
            response.append("*No relevant context found.*")

        response.extend(
            [
                "",
                "## 2. Recommended System Prompt",
                "You can use this prompt with your own LLM:",
                "",
                "```text",
                system_prompt,
                "```",
            ]
        )

        return "\n".join(response)
