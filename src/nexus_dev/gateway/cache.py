"""Tool result caching for MCP gateway."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with TTL."""

    value: Any
    expires_at: float


class ToolCache:
    """LRU cache with TTL for tool invocation results."""

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 1000,
    ) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 5 minutes).
            max_entries: Maximum number of cache entries (default: 1000).
        """
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def ttl_seconds(self) -> float:
        """Get TTL in seconds."""
        return self._ttl_seconds

    @property
    def max_entries(self) -> int:
        """Get maximum entries."""
        return self._max_entries

    @property
    def hits(self) -> int:
        """Get number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Get number of cache misses."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100

    def _generate_key(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> str:
        """Generate cache key from server, tool, and arguments.

        Args:
            server: MCP server name.
            tool: Tool name.
            arguments: Tool arguments dictionary.

        Returns:
            SHA256 hash key.
        """
        key_data = json.dumps(
            {"server": server, "tool": tool, "arguments": arguments},
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is expired.

        Args:
            entry: Cache entry to check.

        Returns:
            True if expired, False otherwise.
        """
        return time.monotonic() > entry.expires_at

    def get(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> Any | None:
        """Get a cached result.

        Args:
            server: MCP server name.
            tool: Tool name.
            arguments: Tool arguments dictionary.

        Returns:
            Cached result or None if not found/expired.
        """
        key = self._generate_key(server, tool, arguments)

        if key not in self._cache:
            self._misses += 1
            logger.debug("[Cache] MISS: %s/%s", server, tool)
            return None

        entry = self._cache[key]

        if self._is_expired(entry):
            del self._cache[key]
            self._misses += 1
            logger.debug("[Cache] EXPIRED: %s/%s", server, tool)
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        logger.debug("[Cache] HIT: %s/%s", server, tool)
        return entry.value

    def set(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
        value: Any,
    ) -> None:
        """Store a result in cache.

        Args:
            server: MCP server name.
            tool: Tool name.
            arguments: Tool arguments dictionary.
            value: Result to cache.
        """
        key = self._generate_key(server, tool, arguments)
        expires_at = time.monotonic() + self._ttl_seconds

        # If key exists, update it and move to end
        if key in self._cache:
            self._cache.move_to_end(key)

        # Evict oldest entry if at capacity
        while len(self._cache) >= self._max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug("[Cache] EVICTED oldest entry")

        self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
        logger.debug("[Cache] SET: %s/%s", server, tool)

    def invalidate(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> None:
        """Invalidate a specific cache entry.

        Args:
            server: MCP server name.
            tool: Tool name.
            arguments: Tool arguments dictionary.
        """
        key = self._generate_key(server, tool, arguments)
        if key in self._cache:
            del self._cache[key]
            logger.debug("[Cache] INVALIDATED: %s/%s", server, tool)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("[Cache] CLEARED")

    def clear_server(self, server: str) -> int:
        """Clear all cache entries for a specific server.

        Args:
            server: MCP server name.

        Returns:
            Number of entries cleared.
        """
        keys_to_remove = [key for key, entry in self._cache.items() if key.startswith(server)]
        for key in keys_to_remove:
            del self._cache[key]
        if keys_to_remove:
            logger.debug("[Cache] CLEARED %d entries for server: %s", len(keys_to_remove), server)
        return len(keys_to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(self.hit_rate, 2),
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl_seconds,
        }


# Mutation tool prefixes/suffixes that should not be cached
MUTATION_PATTERNS = (
    "create",
    "add",
    "new",
    "update",
    "edit",
    "modify",
    "delete",
    "remove",
    "set_",
    "put_",
    "post_",
    "toggle",
    "enable",
    "disable",
)


def is_mutation_tool(tool: str) -> bool:
    """Check if a tool is likely a mutation (write) operation.

    Args:
        tool: Tool name to check.

    Returns:
        True if likely a mutation, False otherwise.
    """
    tool_lower = tool.lower()
    return any(
        tool_lower.startswith(prefix) or tool_lower.endswith(prefix) for prefix in MUTATION_PATTERNS
    )
