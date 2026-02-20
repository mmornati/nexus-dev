"""Context management tools for Nexus-Dev."""

from __future__ import annotations

import logging

from nexus_dev.app_state import get_hybrid_db

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
