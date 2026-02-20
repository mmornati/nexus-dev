"""Search tools for Nexus-Dev."""

from __future__ import annotations

import logging

from nexus_dev.app_state import get_database
from nexus_dev.database import DocumentType
from nexus_dev.query_router import HybridQueryRouter, QueryType

logger = logging.getLogger(__name__)


async def search_knowledge(
    query: str,
    content_type: str = "all",
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search all indexed knowledge including code, documentation, and lessons.

    This is the main search tool that can find relevant information across all
    indexed content types. Use the content_type parameter to filter results.

    Args:
        query: Natural language search query describing what you're looking for.
               Examples: "function that handles user authentication",
               "how to configure the database", "error with null pointer".
        content_type: Filter by content type. Options:
                     - "all": Search everything (default)
                     - "code": Only search code (functions, classes, methods)
                     - "documentation": Only search docs (markdown, rst, txt)
                     - "lesson": Only search recorded lessons
        project_id: Optional project identifier to limit search scope.
                    If not provided, searches across all projects.
        limit: Maximum number of results to return (default: 5, max: 20).

    Returns:
        Formatted search results with file paths, content, and relevance info.
    """
    database = get_database()

    # Only filter by project if explicitly specified
    # None = search across all projects

    # Clamp limit
    limit = min(max(1, limit), 20)

    # Map content_type to DocumentType
    doc_type_filter = None
    if content_type == "code":
        doc_type_filter = DocumentType.CODE
    elif content_type == "documentation":
        doc_type_filter = DocumentType.DOCUMENTATION
    elif content_type == "lesson":
        doc_type_filter = DocumentType.LESSON
    # "all" means no filter

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects
            doc_type=doc_type_filter,
            limit=limit,
        )

        if not results:
            return f"No results found for query: '{query}'" + (
                f" (filtered by {content_type})" if content_type != "all" else ""
            )

        # Format results
        content_label = f" [{content_type.upper()}]" if content_type != "all" else ""
        output_parts = [f"## Search Results{content_label}: '{query}'", ""]

        for i, result in enumerate(results, 1):
            type_badge = f"[{result.doc_type.upper()}]"
            output_parts.append(f"### Result {i}: {type_badge} {result.name}")
            output_parts.append(f"**File:** `{result.file_path}`")
            output_parts.append(f"**Type:** {result.chunk_type} ({result.language})")
            if result.start_line > 0:
                output_parts.append(f"**Lines:** {result.start_line}-{result.end_line}")
            output_parts.append("")
            output_parts.append("```" + result.language)
            output_parts.append(result.text[:2000])  # Truncate long content
            if len(result.text) > 2000:
                output_parts.append("... (truncated)")
            output_parts.append("```")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Search failed: {e!s}"


async def search_docs(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search specifically in documentation (Markdown, RST, text files).

    Use this tool when you need to find information in project documentation,
    README files, or other text documentation. This is more targeted than
    search_knowledge when you know the answer is in the docs.

    Args:
        query: Natural language search query.
               Examples: "how to install", "API configuration", "usage examples".
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Formatted documentation search results.
    """
    database = get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects
            doc_type=DocumentType.DOCUMENTATION,
            limit=limit,
        )

        if not results:
            return f"No documentation found for: '{query}'"

        output_parts = [f"## Documentation Search: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### {i}. {result.name}")
            output_parts.append(f"**Source:** `{result.file_path}`")
            output_parts.append("")
            # For docs, render as markdown directly
            output_parts.append(result.text[:2500])
            if len(result.text) > 2500:
                output_parts.append("\n... (truncated)")
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Documentation search failed: {e!s}"


async def search_lessons(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search in recorded lessons (problems and solutions).

    Use this tool when you encounter an error or problem that might have been
    solved before. Lessons contain problem descriptions and their solutions,
    making them ideal for troubleshooting similar issues.

    Args:
        query: Description of the problem or error you're facing.
               Examples: "TypeError with None", "database connection timeout",
               "how to fix import error".
        project_id: Optional project identifier. Searches all projects if not specified,
                    enabling cross-project learning.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant lessons with problems and solutions.
    """
    database = get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects (cross-project learning)
            doc_type=DocumentType.LESSON,
            limit=limit,
        )

        if not results:
            return (
                f"No lessons found matching: '{query}'\n\n"
                "Tip: Use record_lesson to save problems and solutions for future reference."
            )

        output_parts = [f"## Lessons Found: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### Lesson {i}")
            output_parts.append(f"**ID:** {result.name}")
            output_parts.append(f"**Project:** {result.project_id}")
            output_parts.append("")
            output_parts.append(result.text)
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Lesson search failed: {e!s}"


async def search_code(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search specifically in indexed code (functions, classes, methods).

    Use this tool when you need to find code implementations, function definitions,
    or class structures. This is more targeted than search_knowledge when you
    specifically need code, not documentation.

    Args:
        query: Description of the code you're looking for.
               Examples: "function that handles authentication",
               "class for database connections", "method to validate input".
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant code snippets with file locations.
    """
    database = get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects
            doc_type=DocumentType.CODE,
            limit=limit,
        )

        if not results:
            return f"No code found for: '{query}'"

        output_parts = [f"## Code Search: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### {i}. {result.chunk_type}: {result.name}")
            output_parts.append(f"**File:** `{result.file_path}`")
            output_parts.append(f"**Lines:** {result.start_line}-{result.end_line}")
            output_parts.append(f"**Language:** {result.language}")
            output_parts.append("")
            output_parts.append("```" + result.language)
            output_parts.append(result.text[:2000])
            if len(result.text) > 2000:
                output_parts.append("... (truncated)")
            output_parts.append("```")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Code search failed: {e!s}"


async def smart_search(
    query: str,
    project_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Intelligent search that routes to the best tool (Graph, KV, or Vector).

    Use this as the default search tool. It analyzes the query to determine if
    you are looking for:
    - Code structure/relations (Graph): "who calls function X", "what imports file Y"
    - Session context (KV): "what was the last error", "summarize session"
    - General knowledge (Vector): "how to implement auth", "explain this error"

    Args:
        query: Natural language query.
        project_id: Optional project identifier.
        session_id: Optional session ID for context queries.

    Returns:
        Formatted search results from the appropriate backend.
    """
    # Import here to avoid circular dependencies
    from nexus_dev.tools.context import get_recent_context
    from nexus_dev.tools.graph import find_callers, find_implementations, search_dependencies

    router = HybridQueryRouter()
    intent = router.route(query)

    # 1. Graph Intent
    if intent.query_type == QueryType.GRAPH and intent.extracted_entity:
        entity = intent.extracted_entity
        q_lower = query.lower()

        # Determine specific graph tool based on query patterns
        if "calls" in q_lower or "callers" in q_lower:
            return await find_callers(entity, project_id)

        elif "imports" in q_lower or "dependencies" in q_lower:
            # Default to 'both' unless direction is clear
            direction = "both"
            if "what imports" in q_lower or "who imports" in q_lower:
                direction = "imported_by"
            elif "what does" in q_lower and "import" in q_lower:
                direction = "imports"
            return await search_dependencies(entity, direction=direction, project_id=project_id)

        elif "implements" in q_lower or "extends" in q_lower or "subclasses" in q_lower:
            return await find_implementations(entity, project_id)

    # 2. KV Intent (Session Context)
    elif intent.query_type == QueryType.KV:
        if session_id:
            return await get_recent_context(session_id)
        else:
            return (
                "Query appears to be about session history, but no 'session_id' was provided. "
                "Please provide a session_id to search context, or rephrase for general search."
            )

    # 3. Vector Intent (Default Fallback)
    return await search_knowledge(query, project_id=project_id)
