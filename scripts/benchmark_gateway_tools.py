#!/usr/bin/env python3
"""Benchmark MCP Gateway tool consolidation benefits.

This script measures the token savings from using the nexus-dev MCP gateway
instead of directly exposing all tools from multiple MCP servers.

Key insight: AI models receive tool definitions in their system prompt.
Each tool definition consumes ~100-300 tokens. The gateway consolidates
potentially 100+ tools into just 3:
- search_tools
- get_tool_schema
- invoke_tool

Usage:
    python scripts/benchmark_gateway_tools.py
    python scripts/benchmark_gateway_tools.py --servers github,homeassistant,filesystem

Requirements:
    pip install tiktoken
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    sys.exit(1)


# Example tool definitions from common MCP servers
# These are representative samples of what each server exposes
SAMPLE_TOOL_DEFINITIONS = {
    "github": [
        {
            "name": "create_issue",
            "description": "Create a new issue in a GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "assignees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["owner", "repo", "title"],
            },
        },
        {
            "name": "list_issues",
            "description": "List issues in a GitHub repository with filtering",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "sort": {"type": "string", "enum": ["created", "updated", "comments"]},
                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                    "per_page": {"type": "integer"},
                },
                "required": ["owner", "repo"],
            },
        },
        {
            "name": "create_pull_request",
            "description": "Create a pull request in a GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "head": {"type": "string"},
                    "base": {"type": "string"},
                    "draft": {"type": "boolean"},
                },
                "required": ["owner", "repo", "title", "head", "base"],
            },
        },
        {
            "name": "search_code",
            "description": "Search for code in GitHub repositories",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "sort": {"type": "string"},
                    "order": {"type": "string"},
                    "per_page": {"type": "integer"},
                },
                "required": ["q"],
            },
        },
        {
            "name": "get_file_contents",
            "description": "Get contents of a file from a GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "ref": {"type": "string"},
                },
                "required": ["owner", "repo", "path"],
            },
        },
        {
            "name": "create_or_update_file",
            "description": "Create or update a file in a GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "message": {"type": "string"},
                    "content": {"type": "string"},
                    "branch": {"type": "string"},
                    "sha": {"type": "string"},
                },
                "required": ["owner", "repo", "path", "message", "content"],
            },
        },
        {
            "name": "list_commits",
            "description": "List commits in a repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "sha": {"type": "string"},
                    "path": {"type": "string"},
                    "author": {"type": "string"},
                    "since": {"type": "string"},
                    "until": {"type": "string"},
                },
                "required": ["owner", "repo"],
            },
        },
        {
            "name": "get_commit",
            "description": "Get a specific commit by SHA",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "ref": {"type": "string"},
                },
                "required": ["owner", "repo", "ref"],
            },
        },
        {
            "name": "create_branch",
            "description": "Create a new branch in a repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                    "from_branch": {"type": "string"},
                },
                "required": ["owner", "repo", "branch"],
            },
        },
        {
            "name": "fork_repository",
            "description": "Fork a repository to your account",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "organization": {"type": "string"},
                },
                "required": ["owner", "repo"],
            },
        },
    ],
    "homeassistant": [
        {
            "name": "turn_on",
            "description": "Turn on a Home Assistant entity (light, switch, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Entity ID to turn on"},
                    "brightness": {"type": "integer", "description": "Brightness level 0-255"},
                    "color_temp": {"type": "integer"},
                    "rgb_color": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "turn_off",
            "description": "Turn off a Home Assistant entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "toggle",
            "description": "Toggle a Home Assistant entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "get_state",
            "description": "Get the current state of an entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "call_service",
            "description": "Call any Home Assistant service",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "service": {"type": "string"},
                    "service_data": {"type": "object"},
                    "target": {"type": "object"},
                },
                "required": ["domain", "service"],
            },
        },
        {
            "name": "get_history",
            "description": "Get historical state data for an entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "set_climate",
            "description": "Set climate control settings (thermostat)",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "temperature": {"type": "number"},
                    "hvac_mode": {"type": "string"},
                    "preset_mode": {"type": "string"},
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "trigger_automation",
            "description": "Manually trigger an automation",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                },
                "required": ["entity_id"],
            },
        },
    ],
    "filesystem": [
        {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_directory",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "create_directory",
            "description": "Create a new directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "delete_file",
            "description": "Delete a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "move_file",
            "description": "Move or rename a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["source", "destination"],
            },
        },
        {
            "name": "search_files",
            "description": "Search for files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["path", "pattern"],
            },
        },
        {
            "name": "get_file_info",
            "description": "Get metadata about a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    ],
    "database": [
        {
            "name": "query",
            "description": "Execute a SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {"type": "array"},
                },
                "required": ["sql"],
            },
        },
        {
            "name": "list_tables",
            "description": "List all tables in the database",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "describe_table",
            "description": "Get schema information for a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                },
                "required": ["table"],
            },
        },
        {
            "name": "insert",
            "description": "Insert a row into a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "data": {"type": "object"},
                },
                "required": ["table", "data"],
            },
        },
        {
            "name": "update",
            "description": "Update rows in a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "data": {"type": "object"},
                    "where": {"type": "object"},
                },
                "required": ["table", "data", "where"],
            },
        },
        {
            "name": "delete",
            "description": "Delete rows from a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "where": {"type": "object"},
                },
                "required": ["table", "where"],
            },
        },
    ],
    "slack": [
        {
            "name": "send_message",
            "description": "Send a message to a Slack channel",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "text": {"type": "string"},
                    "thread_ts": {"type": "string"},
                },
                "required": ["channel", "text"],
            },
        },
        {
            "name": "list_channels",
            "description": "List available Slack channels",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_messages",
            "description": "Get messages from a channel",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["channel"],
            },
        },
        {
            "name": "add_reaction",
            "description": "Add a reaction to a message",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["channel", "timestamp", "name"],
            },
        },
    ],
}

# Nexus-Dev Gateway tools (what actually gets exposed)
GATEWAY_TOOLS = [
    {
        "name": "search_tools",
        "description": "Search for MCP tools matching a description. Use this tool to find other MCP tools when you need to perform an action but don't know which tool to use. Returns tool names, descriptions, and parameter schemas.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you want to do",
                },
                "server": {
                    "type": "string",
                    "description": "Optional server name to filter results",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 5, max: 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_tool_schema",
        "description": "Get the full JSON schema for a specific MCP tool. Use this after search_tools to get complete parameter details before calling invoke_tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": 'Server name (e.g., "github")'},
                "tool": {"type": "string", "description": 'Tool name (e.g., "create_pull_request")'},
            },
            "required": ["server", "tool"],
        },
    },
    {
        "name": "invoke_tool",
        "description": "Invoke a tool on a backend MCP server. Use search_tools first to find the right tool, then use this to execute it.",
        "parameters": {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": 'MCP server name (e.g., "github", "homeassistant")',
                },
                "tool": {
                    "type": "string",
                    "description": 'Tool name (e.g., "create_issue", "turn_on_light")',
                },
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments as dictionary",
                },
            },
            "required": ["server", "tool"],
        },
    },
]


@dataclass
class GatewayBenchmarkResult:
    """Results from gateway benchmark."""

    direct_tools_count: int
    direct_tokens: int
    gateway_tools_count: int
    gateway_tokens: int
    servers: list[str]

    @property
    def savings_percent(self) -> float:
        if self.direct_tokens == 0:
            return 0.0
        return (1 - self.gateway_tokens / self.direct_tokens) * 100

    @property
    def tokens_saved(self) -> int:
        return self.direct_tokens - self.gateway_tokens


class TokenCounter:
    """Token counter using tiktoken."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def count_tool_definition(self, tool: dict[str, Any]) -> int:
        """Count tokens in a tool definition as it appears in system prompt."""
        # Tool definitions are typically serialized as JSON in the system prompt
        tool_text = json.dumps(tool, indent=2)
        return self.count(tool_text)


def run_gateway_benchmark(
    servers: list[str],
    counter: TokenCounter,
) -> GatewayBenchmarkResult:
    """Run the gateway consolidation benchmark.

    Args:
        servers: List of server names to include.
        counter: Token counter instance.

    Returns:
        Benchmark results.
    """
    # Count direct tool exposure tokens
    direct_tools = []
    for server in servers:
        if server in SAMPLE_TOOL_DEFINITIONS:
            direct_tools.extend(SAMPLE_TOOL_DEFINITIONS[server])

    direct_tokens = sum(counter.count_tool_definition(t) for t in direct_tools)

    # Count gateway tool exposure tokens
    gateway_tokens = sum(counter.count_tool_definition(t) for t in GATEWAY_TOOLS)

    return GatewayBenchmarkResult(
        direct_tools_count=len(direct_tools),
        direct_tokens=direct_tokens,
        gateway_tools_count=len(GATEWAY_TOOLS),
        gateway_tokens=gateway_tokens,
        servers=servers,
    )


def generate_markdown_report(result: GatewayBenchmarkResult) -> str:
    """Generate markdown report."""
    lines = [
        "# MCP Gateway Tool Consolidation Benchmark",
        "",
        "## Configuration",
        "",
        f"Servers included: **{', '.join(result.servers)}**",
        "",
        "## Results",
        "",
        "| Metric | Direct Exposure | Gateway | Reduction |",
        "|--------|-----------------|---------|-----------|",
        f"| Tools in system prompt | {result.direct_tools_count} | {result.gateway_tools_count} | {result.direct_tools_count - result.gateway_tools_count} fewer |",
        f"| Tokens per request | {result.direct_tokens:,} | {result.gateway_tokens:,} | **{result.savings_percent:.1f}%** |",
        f"| Tokens saved | — | — | {result.tokens_saved:,} |",
        "",
        "## Impact Analysis",
        "",
        f"### Per-Request Savings",
        f"- **{result.tokens_saved:,} tokens saved** per request",
        f"- At $2.50/1M tokens (GPT-4o input): **${result.tokens_saved * 2.5 / 1_000_000:.6f}** per request",
        "",
        "### Session Savings (100 requests/session)",
        f"- Tokens saved: {result.tokens_saved * 100:,}",
        f"- Cost saved: ${result.tokens_saved * 100 * 2.5 / 1_000_000:.4f}",
        "",
        "### Monthly Savings (1000 sessions × 100 requests)",
        f"- Tokens saved: {result.tokens_saved * 100_000:,}",
        f"- Cost saved: ${result.tokens_saved * 100_000 * 2.5 / 1_000_000:.2f}",
        "",
        "## The Trade-off",
        "",
        "The gateway approach has a trade-off:",
        "",
        "| Aspect | Direct Tools | Gateway |",
        "|--------|--------------|---------|",
        "| System prompt size | Large (all tool schemas) | Small (3 tools) |",
        "| Tool discovery | Immediate (in prompt) | Requires `search_tools` call |",
        "| Extra API calls | None | 1-2 for discovery |",
        "| Total tokens/session | Higher baseline | Lower baseline + discovery overhead |",
        "",
        "**Break-even point**: The gateway is more efficient when tool definitions",
        f"exceed ~{result.gateway_tokens * 3} tokens (current direct: {result.direct_tokens:,}).",
        "",
        "## Detailed Tool Breakdown",
        "",
    ]

    for server in result.servers:
        if server in SAMPLE_TOOL_DEFINITIONS:
            tools = SAMPLE_TOOL_DEFINITIONS[server]
            server_tokens = sum(TokenCounter().count_tool_definition(t) for t in tools)
            lines.append(f"### {server}")
            lines.append(f"- Tools: {len(tools)}")
            lines.append(f"- Tokens: {server_tokens:,}")
            lines.append("")
            for tool in tools:
                tokens = TokenCounter().count_tool_definition(tool)
                lines.append(f"  - `{tool['name']}`: {tokens} tokens")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MCP Gateway tool consolidation benefits"
    )
    parser.add_argument(
        "--servers",
        default="github,homeassistant,filesystem",
        help="Comma-separated list of servers to include (default: github,homeassistant,filesystem)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model for token counting (default: gpt-4o)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    servers = [s.strip() for s in args.servers.split(",")]
    counter = TokenCounter(model=args.model)

    result = run_gateway_benchmark(servers, counter)

    if args.format == "json":
        output = json.dumps(
            {
                "servers": result.servers,
                "direct_tools_count": result.direct_tools_count,
                "direct_tokens": result.direct_tokens,
                "gateway_tools_count": result.gateway_tools_count,
                "gateway_tokens": result.gateway_tokens,
                "savings_percent": result.savings_percent,
                "tokens_saved": result.tokens_saved,
            },
            indent=2,
        )
    else:
        output = generate_markdown_report(result)

    if args.output:
        args.output.write_text(output)
        print(f"Report written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
