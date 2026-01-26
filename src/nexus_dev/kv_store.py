"""SQLite-based key-value store for fast exact lookups."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class KVStore:
    """SQLite-based key-value store for session state.

    Provides fast exact lookups for:
    - Session metadata and state
    - Chat history (message-by-message recall)
    - Configuration cache with TTL support

    Attributes:
        db_path: Path to SQLite database file
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize KV store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Connect and initialize schema.

        Creates database file and tables if they don't exist.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if not exist."""
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_session
                ON chat_history(session_id);
            CREATE INDEX IF NOT EXISTS idx_chat_timestamp
                ON chat_history(timestamp DESC);

            CREATE TABLE IF NOT EXISTS config_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at TEXT
            );
        """)
        self._conn.commit()

    # Session methods

    def create_session(
        self, session_id: str, project_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            project_id: Project this session belongs to
            metadata: Optional session metadata
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        metadata_json = json.dumps(metadata or {})
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, project_id, metadata, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (session_id, project_id, metadata_json),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        if not row:
            return None

        return {
            "session_id": row["session_id"],
            "project_id": row["project_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata"]),
        }

    def update_session(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Update session metadata.

        Args:
            session_id: Session identifier
            metadata: New metadata (replaces existing)
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        metadata_json = json.dumps(metadata)
        self._conn.execute(
            """UPDATE sessions
               SET metadata = ?, updated_at = CURRENT_TIMESTAMP
               WHERE session_id = ?""",
            (metadata_json, session_id),
        )
        self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        """Delete session and all associated chat history.

        Args:
            session_id: Session identifier
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        # Delete chat history first (foreign key constraint)
        self._conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self._conn.commit()

    # Chat history methods

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """Add a message to chat history.

        Args:
            session_id: Session identifier
            role: Message role ('user', 'assistant', or 'system')
            content: Message content

        Returns:
            Message ID
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        cursor = self._conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent messages for a session.

        Args:
            session_id: Session identifier
            limit: Maximum messages to return

        Returns:
            List of messages in chronological order (oldest first)
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        rows = self._conn.execute(
            """SELECT role, content, timestamp FROM chat_history
               WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()

        # Reverse to chronological order (oldest first)
        return [
            {"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
            for row in list(reversed(rows))
        ]

    def get_message_count(self, session_id: str) -> int:
        """Get total message count for session.

        Args:
            session_id: Session identifier

        Returns:
            Number of messages
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        row = self._conn.execute(
            "SELECT COUNT(*) as count FROM chat_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["count"] if row else 0

    # Config cache methods

    def set_cache(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set a cache entry.

        Args:
            key: Cache key
            value: Value (will be JSON serialized)
            ttl_seconds: Time to live in seconds (None = no expiration)
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(UTC).timestamp() + ttl_seconds

        self._conn.execute(
            "INSERT OR REPLACE INTO config_cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), expires_at),
        )
        self._conn.commit()

    def get_cache(self, key: str) -> Any | None:
        """Get a cache entry (returns None if expired).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        row = self._conn.execute(
            "SELECT value, expires_at FROM config_cache WHERE key = ?", (key,)
        ).fetchone()

        if not row:
            return None

        # Check expiration
        if row["expires_at"] and float(row["expires_at"]) < datetime.now(UTC).timestamp():
            # Expired - delete and return None
            self._conn.execute("DELETE FROM config_cache WHERE key = ?", (key,))
            self._conn.commit()
            return None

        return json.loads(row["value"])

    def delete_cache(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: Cache key
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        self._conn.execute("DELETE FROM config_cache WHERE key = ?", (key,))
        self._conn.commit()

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries deleted
        """
        if self._conn is None:
            msg = "Not connected to database"
            raise RuntimeError(msg)

        now = datetime.now(UTC).timestamp()
        cursor = self._conn.execute(
            "DELETE FROM config_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> KVStore:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
