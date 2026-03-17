"""Gateway related commands."""

from __future__ import annotations

import click

from nexus_dev.gateway.metrics import get_gateway_metrics


@click.group("gateway")
def gateway_group() -> None:
    """Manage gateway operations."""


@gateway_group.command("stats")
def gateway_stats_command() -> None:
    """Show gateway usage statistics.

    Displays metrics for the last 24 hours including:
    - search_tools and invoke_tool call counts
    - Cache hit/miss ratio
    - Tools accessed per server

    Examples:
        nexus gateway stats
    """
    metrics = get_gateway_metrics()
    summary = metrics.get_summary()
    click.echo(summary)
