"""Tool output summarization for MCP gateway."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SummarizeSettings:
    """Summarization configuration for tool outputs."""

    enabled: bool = True
    max_list_items: int = 10
    max_output_chars: int = 500


class OutputSummarizer:
    """Summarizes verbose tool outputs to reduce token usage."""

    def __init__(
        self,
        enabled: bool = True,
        max_list_items: int = 10,
        max_output_chars: int = 500,
    ) -> None:
        """Initialize the summarizer.

        Args:
            enabled: Whether summarization is enabled.
            max_list_items: Maximum number of items to show in lists.
            max_output_chars: Maximum characters in output.
        """
        self.enabled = enabled
        self.max_list_items = max_list_items
        self.max_output_chars = max_output_chars

    def summarize(self, result: Any) -> Any:
        """Summarize a tool result to reduce token usage.

        Args:
            result: The raw tool result.

        Returns:
            Summarized result.
        """
        if not self.enabled:
            return result

        # Handle None
        if result is None:
            return result

        # Handle string results
        if isinstance(result, str):
            return self._summarize_string(result)

        # Handle list/tuple results
        if isinstance(result, (list, tuple)):
            return self._summarize_list(result)

        # Handle dict results
        if isinstance(result, dict):
            return self._summarize_dict(result)

        # Handle other types - convert to string
        return self._summarize_string(str(result))

    def _summarize_string(self, text: str) -> str:
        """Summarize a string by truncating if too long.

        Args:
            text: Input string.

        Returns:
            Truncated string with note.
        """
        if len(text) <= self.max_output_chars:
            return text

        truncated = text[: self.max_output_chars]
        return f"{truncated}\n\n[Output truncated, {len(text)} -> {self.max_output_chars} chars]"

    def _summarize_list(self, items: list[Any] | tuple[Any, ...]) -> list[Any] | tuple[Any, ...]:
        """Summarize a list by showing first N items.

        Args:
            items: Input list or tuple.

        Returns:
            Truncated list with note.
        """
        if len(items) <= self.max_list_items:
            return items

        shown = list(items[: self.max_list_items])
        remaining = len(items) - self.max_list_items
        shown.append(f"... (showing {self.max_list_items} of {len(items)} items, {remaining} more)")

        return shown if isinstance(items, list) else tuple(shown)

    def _summarize_dict(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Summarize a dictionary by showing first N key-value pairs.

        Args:
            obj: Input dictionary.

        Returns:
            Truncated dict with note.
        """
        if len(obj) <= self.max_list_items:
            return obj

        keys = list(obj.keys())[: self.max_list_items]
        shown = {k: obj[k] for k in keys}
        remaining = len(obj) - self.max_list_items
        shown[f"... ({remaining} more fields)"] = "Use get_tool_schema for full schema"

        return shown


def serialize_for_summarization(result: Any) -> Any:
    """Prepare a result for summarization by serializing MCP content.

    MCP results often have a specific structure. This function extracts
    the readable content for summarization.

    Args:
        result: Raw tool result from MCP.

    Returns:
        Serializable content for summarization.
    """
    # Handle MCP CallToolResult with content
    if hasattr(result, "content"):
        contents = []
        for item in result.content:
            if hasattr(item, "text"):
                # Try to parse as JSON
                text = item.text
                try:
                    parsed = json.loads(text)
                    contents.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    contents.append(text)
            else:
                contents.append(str(item))

        if len(contents) == 1:
            return contents[0]
        return contents

    return result
