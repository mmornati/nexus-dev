"""Redis-based key-value store (FalkorDBLite) for fast exact lookups."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis import Redis


class KVStore:
    """Redis-based key-value store for session state.

    Provides fast exact lookups for:
    - Session metadata and state (Hash)
    - Chat history (List of JSON)
    - Configuration cache (String with TTL)

    Attributes:
        client: Redis client instance
    """

    def __init__(self, client: Redis) -> None:
        """Initialize KV store.

        Args:
            client: Configured Redis client
        """
        self.client = client

    def connect(self) -> None:
        """Verify connection.

        Using external client, so we just ping.
        """
        self.client.ping()

    # Session methods

    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def create_session(
        self, session_id: str, project_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            project_id: Project this session belongs to
            metadata: Optional session metadata
        """
        key = self._session_key(session_id)
        now = datetime.now(UTC).isoformat()

        data = {
            "session_id": session_id,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
            "metadata": json.dumps(metadata or {}),
        }

        self.client.hset(key, mapping=data)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        key = self._session_key(session_id)
        data = self.client.hgetall(key)

        if not data:
            return None

        # Decode bytes to strings if needed (redis-py usually handles decoding if configured,
        # but let's be safe or assume decode_responses=True in client config)
        # We will assume decode_responses=True for simplicity.

        return {
            "session_id": data["session_id"],
            "project_id": data["project_id"],
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "metadata": json.loads(data["metadata"]),
        }

    def update_session(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Update session metadata.

        Args:
            session_id: Session identifier
            metadata: New metadata (replaces existing)
        """
        key = self._session_key(session_id)
        if not self.client.exists(key):
            # mimics sqlite behavior of not updating if not exists?
            # SQLite UPDATE WHERE would do nothing.
            return

        now = datetime.now(UTC).isoformat()
        self.client.hset(key, mapping={"metadata": json.dumps(metadata), "updated_at": now})

    def delete_session(self, session_id: str) -> None:
        """Delete session and all associated chat history.

        Args:
            session_id: Session identifier
        """
        # Delete session key
        self.client.delete(self._session_key(session_id))

        # Delete chat history key
        self.client.delete(f"chat:{session_id}")

    # Chat history methods

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """Add a message to chat history.

        Args:
            session_id: Session identifier
            role: Message role ('user', 'assistant', or 'system')
            content: Message content

        Returns:
            Message ID (index in list + 1)
        """
        msg = {"role": role, "content": content, "timestamp": datetime.now(UTC).isoformat()}

        # Store as JSON string in a list
        # RPUSH returns the length of the list after push
        count = self.client.rpush(f"chat:{session_id}", json.dumps(msg))
        return count

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent messages for a session.

        Args:
            session_id: Session identifier
            limit: Maximum messages to return

        Returns:
            List of messages in chronological order (oldest first)
        """
        key = f"chat:{session_id}"
        if limit <= 0:
            return []

        # Get last 'limit' messages
        # LRANGE start stop (inclusive)
        # To get last N: start = -N, stop = -1
        raw_msgs = self.client.lrange(key, -limit, -1)

        return [json.loads(msg) for msg in raw_msgs]

    def get_message_count(self, session_id: str) -> int:
        """Get total message count for session.

        Args:
            session_id: Session identifier

        Returns:
            Number of messages
        """
        return self.client.llen(f"chat:{session_id}")

    # Config cache methods

    def set_cache(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set a cache entry.

        Args:
            key: Cache key
            value: Value (will be JSON serialized)
            ttl_seconds: Time to live in seconds (None = no expiration)
        """
        # Redis handles TTL natively
        self.client.set(f"cache:{key}", json.dumps(value), ex=ttl_seconds)

    def get_cache(self, key: str) -> Any | None:
        """Get a cache entry.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        val = self.client.get(f"cache:{key}")
        if val is None:
            return None
        return json.loads(val)

    def delete_cache(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: Cache key
        """
        self.client.delete(f"cache:{key}")

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries deleted (Redis does this automatically, so strictly 0)
        """
        # Redis handles active/passive expiration.
        # We don't need manual cleanup.
        return 0

    def close(self) -> None:
        """Close database connection."""
        try:
            self.client.close()
        except Exception:
            pass

    def __enter__(self) -> KVStore:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
