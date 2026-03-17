"""Gateway usage metrics tracking."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GatewayMetrics:
    """Tracks gateway tool usage metrics.

    Monitors search_tools and invoke_tool calls, cache performance,
    and tools accessed per server.
    """

    _search_tools_calls: int = 0
    _invoke_tool_calls: int = 0
    _server_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _search_tools_timestamps: list[float] = field(default_factory=list)
    _invoke_tool_timestamps: list[float] = field(default_factory=list)

    def record_search_tools(self) -> None:
        """Record a search_tools call."""
        self._search_tools_calls += 1
        self._search_tools_timestamps.append(time.time())
        logger.debug("[Metrics] search_tools call recorded (total: %d)", self._search_tools_calls)

    def record_invoke_tool(self, server: str) -> None:
        """Record an invoke_tool call with server name.

        Args:
            server: The MCP server name that was invoked.
        """
        self._invoke_tool_calls += 1
        self._server_calls[server] += 1
        self._invoke_tool_timestamps.append(time.time())
        logger.debug(
            "[Metrics] invoke_tool call recorded: %s (total: %d)", server, self._invoke_tool_calls
        )

    @property
    def search_tools_calls(self) -> int:
        """Get total search_tools call count."""
        return self._search_tools_calls

    @property
    def invoke_tool_calls(self) -> int:
        """Get total invoke_tool call count."""
        return self._invoke_tool_calls

    @property
    def server_calls(self) -> dict[str, int]:
        """Get tool calls per server."""
        return dict(self._server_calls)

    @property
    def total_tool_calls(self) -> int:
        """Get total tool-related calls (search + invoke)."""
        return self._search_tools_calls + self._invoke_tool_calls

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics from the connection manager.

        Returns:
            Dictionary with cache hits, misses, and hit rate.
        """
        from nexus_dev.app_state import get_connection_manager

        try:
            conn_manager = get_connection_manager()
            cache = conn_manager._get_cache()  # noqa: SLF001
            if cache is not None:
                return {"hits": cache.hits, "misses": cache.misses}
        except Exception as exc:
            logger.warning("[Metrics] Failed to get cache stats: %s", exc)

        return {"hits": 0, "misses": 0}

    def get_stats_24h(self) -> dict[str, Any]:
        """Get metrics for the last 24 hours.

        Returns:
            Dictionary with usage statistics.
        """
        now = time.time()
        cutoff = now - (24 * 60 * 60)

        recent_search_calls = sum(1 for ts in self._search_tools_timestamps if ts >= cutoff)
        recent_invoke_calls = sum(1 for ts in self._invoke_tool_timestamps if ts >= cutoff)

        cache_stats = self.get_cache_stats()

        return {
            "search_tools_calls": recent_search_calls,
            "invoke_tool_calls": recent_invoke_calls,
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "server_calls": dict(self._server_calls),
        }

    def get_summary(self) -> str:
        """Get a formatted summary of gateway usage.

        Returns:
            Formatted string with usage statistics.
        """
        stats = self.get_stats_24h()

        search_tools_calls = stats["search_tools_calls"]
        invoke_tool_calls = stats["invoke_tool_calls"]
        cache_hits = stats["cache_hits"]
        cache_misses = stats["cache_misses"]

        total_cache = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_cache * 100) if total_cache > 0 else 0

        lines = [
            "Gateway Usage (last 24h):",
            f"- search_tools calls: {search_tools_calls}",
            f"- invoke_tool calls: {invoke_tool_calls}",
            f"- Cache hits: {cache_hits} ({cache_hit_rate:.1f}%)",
            f"- Cache misses: {cache_misses}",
            "",
            "Tools by server:",
        ]

        server_calls = stats.get("server_calls", {})
        if server_calls:
            for server, count in sorted(server_calls.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {server}: {count} calls")
        else:
            lines.append("- No data available")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        self._search_tools_calls = 0
        self._invoke_tool_calls = 0
        self._server_calls.clear()
        self._search_tools_timestamps.clear()
        self._invoke_tool_timestamps.clear()
        logger.debug("[Metrics] Metrics reset")


_gateway_metrics: GatewayMetrics | None = None


def get_gateway_metrics() -> GatewayMetrics:
    """Get the global GatewayMetrics instance.

    Returns:
        The singleton GatewayMetrics instance.
    """
    global _gateway_metrics
    if _gateway_metrics is None:
        _gateway_metrics = GatewayMetrics()
    return _gateway_metrics
