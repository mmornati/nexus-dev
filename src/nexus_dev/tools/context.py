"""Context management tools for Nexus-Dev."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nexus_dev.app_state import get_hybrid_db

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def get_recent_context(
    session_id: str,
    limit: int = 20,
) -> str:
    """Get recent chat messages from the session history.

    Use this tool to recall previous interactions, user requests, or decisions
    made earlier in the current session. This uses the high-speed KV store.

    Args:
        session_id: The session ID to retrieve history for.
        limit: Maximum number of messages to return (default: 20).

    Returns:
        Formatted chat history or a status message if no history found.
    """
    hybrid_db = get_hybrid_db()

    # Check if hybrid DB is enabled
    if not hybrid_db.config.enable_hybrid_db:
        return "Hybrid database is not enabled in configuration."

    try:
        # Connect if needed
        hybrid_db.connect()

        # Get messages from KV store
        messages = hybrid_db.kv.get_recent_messages(session_id, limit=limit)

        if not messages:
            return f"No chat history found for session: {session_id}"

        output = [f"## Recent Context (Session: {session_id})", ""]

        for msg in messages:
            role = msg["role"].upper()
            ts = msg.get("timestamp", "unknown time")
            content = msg["content"]

            output.append(f"### {role} ({ts})")
            output.append(content)
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error retrieving context: {e!s}"


async def set_session_context(
    session_id: str,
    current_task: str | None = None,
    recent_files: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> str:
    """Store session context for search suggestions.

    Stores current task and recent files to enable proactive search suggestions.
    This helps Nexus-Dev remember what the user is working on.

    Args:
        session_id: The session ID to store context for.
        current_task: Description of the current task (e.g., "Implementing user auth").
        recent_files: List of recently edited file paths.
        metadata: Additional context metadata (e.g., {"language": "python"}).

    Returns:
        Confirmation message with stored context summary.
    """
    hybrid_db = get_hybrid_db()

    if not hybrid_db.config.enable_hybrid_db:
        return "Hybrid database is not enabled in configuration."

    try:
        hybrid_db.connect()

        if current_task is None and recent_files is None and metadata is None:
            return (
                "Error: At least one of current_task, recent_files, or metadata must be provided."
            )

        hybrid_db.kv.set_session_context(
            session_id=session_id,
            current_task=current_task,
            recent_files=recent_files,
            metadata=metadata,
        )

        context = hybrid_db.kv.get_session_context(session_id)

        output = ["## Session Context Updated", ""]
        if context.get("current_task"):
            output.append(f"**Current Task:** {context['current_task']}")
        if context.get("recent_files"):
            output.append("**Recent Files:**")
            for f in context["recent_files"]:
                output.append(f"  - {f}")
        if context.get("metadata"):
            output.append(f"**Metadata:** {context['metadata']}")

        return "\n".join(output)

    except Exception as e:
        return f"Error storing context: {e!s}"


async def get_session_context(session_id: str) -> str:
    """Get session context for search suggestions.

    Retrieves the stored session context including current task and recent files.
    This helps understand what the user is working on.

    Args:
        session_id: The session ID to retrieve context for.

    Returns:
        Formatted session context or message if not found.
    """
    hybrid_db = get_hybrid_db()

    if not hybrid_db.config.enable_hybrid_db:
        return "Hybrid database is not enabled in configuration."

    try:
        hybrid_db.connect()

        context = hybrid_db.kv.get_session_context(session_id)

        if not context.get("current_task") and not context.get("recent_files"):
            return f"No session context found for session: {session_id}"

        output = [f"## Session Context (Session: {session_id})", ""]

        if context.get("current_task"):
            output.append(f"**Current Task:** {context['current_task']}")

        if context.get("recent_files"):
            output.append("**Recent Files:**")
            for f in context["recent_files"]:
                output.append(f"  - {f}")

        if context.get("metadata"):
            output.append(f"**Metadata:** {context['metadata']}")

        if context.get("updated_at"):
            output.append(f"**Last Updated:** {context['updated_at']}")

        return "\n".join(output)

    except Exception as e:
        return f"Error retrieving context: {e!s}"


async def get_search_suggestions(
    session_id: str,
    limit: int = 5,
) -> str:
    """Get search suggestions based on session context.

    Generates relevant search queries based on current task and recent files.
    This enables proactive suggestions for the LLM.

    Args:
        session_id: The session ID to get suggestions for.
        limit: Maximum number of suggestions to return (default: 5).

    Returns:
        Formatted search suggestions or message if no context available.
    """
    hybrid_db = get_hybrid_db()

    if not hybrid_db.config.enable_hybrid_db:
        return "Hybrid database is not enabled in configuration."

    try:
        hybrid_db.connect()

        context = hybrid_db.kv.get_session_context(session_id)

        if not context.get("current_task") and not context.get("recent_files"):
            return (
                f"No session context available for session: {session_id}. "
                "Use set_session_context to store context first."
            )

        suggestions: list[str] = []
        current_task = context.get("current_task")
        recent_files = context.get("recent_files", [])

        if current_task:
            suggestions.append(f"Search for: {current_task}")
            suggestions.append(f"Find related code to: {current_task}")
            suggestions.append(f"Find documentation about: {current_task}")

        for file_path in recent_files[:3]:
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
            base_name = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
            suggestions.append(f"Search code for: {base_name}")

        suggestions = suggestions[:limit]

        output = [f"## Search Suggestions (Session: {session_id})", ""]

        if suggestions:
            output.append("Based on your session context:")
            for i, suggestion in enumerate(suggestions, 1):
                output.append(f"{i}. {suggestion}")
        else:
            output.append("No suggestions available. Set session context first.")

        if recent_files:
            output.append("")
            output.append("**Tip:** Use `set_session_context` to update your current task.")

        return "\n".join(output)

    except Exception as e:
        return f"Error generating suggestions: {e!s}"
