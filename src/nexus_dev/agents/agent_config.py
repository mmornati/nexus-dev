"""Pydantic models for agent configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentProfile(BaseModel):
    """Agent persona definition.

    Attributes:
        role: Role title (e.g., 'Code Reviewer').
        goal: Primary objective of the agent.
        backstory: Agent's background and expertise.
        tone: Communication style.
    """

    role: str = Field(..., description="Role title (e.g., 'Code Reviewer')")
    goal: str = Field(..., description="Primary objective")
    backstory: str = Field(..., description="Agent's background and expertise")
    tone: str = Field(default="Professional and direct")


class AgentMemory(BaseModel):
    """RAG configuration for the agent.

    Attributes:
        enabled: Whether to enable project RAG search.
        rag_limit: Maximum number of context chunks to retrieve.
        search_types: Document types to include in RAG search.
    """

    enabled: bool = Field(default=True, description="Enable project RAG search")
    rag_limit: int = Field(default=5, ge=1, le=20, description="Max context chunks")
    search_types: list[Literal["code", "documentation", "lesson"]] = Field(
        default=["code", "documentation", "lesson"],
        description="Document types to include in RAG",
    )


class LLMConfig(BaseModel):
    """Model hint for MCP Sampling.

    The model_hint is sent to the IDE as a preference. The IDE may use a different
    model based on its configuration and available models.

    Attributes:
        model_hint: Suggested model name for the IDE.
        fallback_hints: Alternative models if primary is unavailable.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
    """

    model_hint: str = Field(
        default="claude-sonnet-4.5",
        description="Model preference sent to IDE (hint only)",
    )
    fallback_hints: list[str] = Field(
        default_factory=lambda: ["auto"],
        description="Fallback models if primary unavailable",
    )
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=100, le=32000)


class AgentConfig(BaseModel):
    """Complete agent configuration.

    This model represents a complete agent definition loaded from a YAML file.
    It includes the agent's identity, memory settings, tool access, and LLM preferences.

    Attributes:
        name: Internal identifier (lowercase with underscores).
        display_name: Human-readable name for the agent.
        description: Tool description for AI discovery.
        profile: Agent persona definition.
        memory: RAG configuration.
        tools: List of allowed MCP tool names (empty = all tools).
        llm_config: Model preferences for MCP Sampling.
    """

    name: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Internal ID (lowercase, underscores allowed)",
    )
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Tool description for AI discovery")
    profile: AgentProfile
    memory: AgentMemory = Field(default_factory=AgentMemory)
    tools: list[str] = Field(
        default_factory=list,
        description="Allowed MCP tool names (empty = all project tools)",
    )
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
