"""Insight and Implementation search tools for Nexus-Dev."""

from __future__ import annotations

import logging

from nexus_dev.app_state import get_database
from nexus_dev.database import DocumentType

logger = logging.getLogger(__name__)


async def search_insights(
    query: str,
    category: str | None = None,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search recorded insights from past development sessions.

    Use when:
    - Starting work on similar features
    - Debugging issues that might have been seen before
    - Looking for optimization patterns
    - Checking if a mistake was already made

    Args:
        query: Description of what you're looking for.
               Examples: "httpx compatibility", "authentication mistakes",
               "database optimization patterns"
        category: Optional filter - "mistake", "discovery", "backtrack", or "optimization"
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant insights with category, description, and reasoning.
    """
    database = get_database()
    limit = min(max(1, limit), 20)

    # Validate category if provided
    if category:
        valid_categories = {"mistake", "discovery", "backtrack", "optimization"}
        if category not in valid_categories:
            return f"Error: category must be one of {valid_categories}, got '{category}'"

    try:
        results = await database.search(
            query=query,
            project_id=project_id,
            doc_type=DocumentType.INSIGHT,
            limit=limit,
        )

        # Filter by category if specified
        if category and results:
            results = [r for r in results if category in r.name]

        if not results:
            msg = f"No insights found for: '{query}'"
            if category:
                msg += f" (category: {category})"
            return msg + "\n\nTip: Use record_insight to save insights for future reference."

        output_parts = [f"## Insights Found: '{query}'", ""]
        if category:
            output_parts[0] += f" (category: {category})"

        for i, result in enumerate(results, 1):
            output_parts.append(f"### Insight {i}")
            output_parts.append(f"**ID:** {result.name}")
            output_parts.append(f"**Project:** {result.project_id}")
            output_parts.append("")
            output_parts.append(result.text)
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Insight search failed: {e!s}"


async def search_implementations(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search recorded implementations.

    Use to find:
    - How similar features were built
    - Design patterns used in the project
    - Past technical approaches
    - Implementation history

    Args:
        query: What you're looking for.
               Examples: "authentication implementation", "database refactor",
               "API design patterns"
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant implementations with summaries and design decisions.
    """
    database = get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,
            doc_type=DocumentType.IMPLEMENTATION,
            limit=limit,
        )

        if not results:
            return (
                f"No implementations found for: '{query}'\n\n"
                "Tip: Use record_implementation after completing features to save them."
            )

        output_parts = [f"## Implementations Found: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### Implementation {i}")
            output_parts.append(f"**ID:** {result.name}")
            output_parts.append(f"**Project:** {result.project_id}")
            output_parts.append("")
            # Truncate long content
            output_parts.append(result.text[:3000])
            if len(result.text) > 3000:
                output_parts.append("\n... (truncated)")
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Implementation search failed: {e!s}"
