"""Agent related commands."""

from __future__ import annotations

import re
from pathlib import Path

import click
import yaml

from nexus_dev.agent_templates import get_template_path, list_templates


@click.group("agent")
def agent_group() -> None:
    """Manage custom agents."""


@agent_group.command("templates")
def agent_templates_command() -> None:
    """List available agent templates."""
    templates = list_templates()

    if not templates:
        click.echo("No templates found.")
        return

    click.echo("📋 Available Agent Templates:")
    click.echo("")

    # Load and display each template
    for template_name in sorted(templates):
        try:
            template_path = get_template_path(template_name)
            with open(template_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            display_name = data.get("display_name", template_name)
            role = data.get("profile", {}).get("role", "Unknown")
            model = data.get("llm_config", {}).get("model_hint", "auto")

            click.echo(f"  • {display_name} ({template_name})")
            click.echo(f"    Role: {role}")
            click.echo(f"    Model: {model}")
            click.echo("")
        except Exception as e:
            click.echo(f"  ⚠️  {template_name}: Failed to load ({e})")


@agent_group.command("init")
@click.argument("name")
@click.option("--from-template", "-t", "template_name", help="Create from template")
@click.option("--model", "-m", "custom_model", help="Override template model")
@click.option("--role", prompt=False, default=None)
@click.option("--goal", prompt=False, default=None)
@click.option(
    "--backstory",
    prompt=False,
    default=None,
)
def agent_init_command(
    name: str,
    template_name: str | None,
    custom_model: str | None,
    role: str | None,
    goal: str | None,
    backstory: str | None,
) -> None:
    """Create a new agent configuration.

    NAME is the agent identifier (lowercase with underscores).

    Examples:
        nexus-agent init my_reviewer --from-template code_reviewer
        nexus-agent init security_check -t security_auditor --model claude-opus-4.5
        nexus-agent init my_custom_agent
    """
    agents_dir = Path.cwd() / "agents"
    agents_dir.mkdir(exist_ok=True)

    # Normalize name
    agent_name = name.lower().replace(" ", "_").replace("-", "_")

    # Validate name format
    if not re.match(r"^[a-z][a-z0-9_]*$", agent_name):
        click.echo(f"❌ Invalid agent name: {agent_name}", err=True)
        click.echo(
            "   Name must start with a letter and contain only lowercase letters, "
            "numbers, and underscores."
        )
        return

    # Load from template if specified
    if template_name:
        available_templates = list_templates()
        if template_name not in available_templates:
            click.echo(f"❌ Template '{template_name}' not found.", err=True)
            click.echo(f"   Available templates: {', '.join(available_templates)}")
            return

        template_path = get_template_path(template_name)
        with open(template_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Customize the template
        config["name"] = agent_name
        config["display_name"] = name.replace("_", " ").title()
        config["description"] = f"Delegate tasks to the {name.replace('_', ' ').title()} agent."

        # Override model if specified
        if custom_model:
            config["llm_config"]["model_hint"] = custom_model

        click.echo(f"✅ Created agent from template: {template_name}")
    else:
        # Interactive mode
        if not role:
            role = click.prompt("Agent role (e.g., 'Code Reviewer')")
        if not goal:
            goal = click.prompt("Agent goal (e.g., 'Review code for best practices')")
        if not backstory:
            backstory = click.prompt(
                "Agent backstory", default="Expert developer with years of experience."
            )

        # Generate YAML content
        config = {
            "name": agent_name,
            "display_name": name.replace("_", " ").title(),
            "description": f"Delegate tasks to the {name.replace('_', ' ').title()} agent.",
            "profile": {
                "role": role,
                "goal": goal,
                "backstory": backstory,
                "tone": "Professional and helpful",
            },
            "memory": {
                "enabled": True,
                "rag_limit": 5,
                "search_types": ["code", "documentation", "lesson"],
            },
            "tools": [],
            "llm_config": {
                "model_hint": custom_model or "claude-sonnet-4.5",
                "fallback_hints": ["auto"],
                "temperature": 0.5,
                "max_tokens": 4000,
            },
        }

    output_file = agents_dir / f"{agent_name}.yaml"

    if output_file.exists() and not click.confirm(
        f"Agent {agent_name}.yaml already exists. Overwrite?"
    ):
        click.echo("Aborted.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    click.echo(f"✅ Created agent: {output_file}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. Edit {output_file} to customize your agent")
    click.echo("  2. Restart the MCP server to activate this agent")
    click.echo(f"  3. Use the 'ask_{agent_name}' tool in your IDE")


@agent_group.command("list")
def agent_list_command() -> None:
    """List all configured agents."""
    agents_dir = Path.cwd() / "agents"

    if not agents_dir.exists():
        click.echo("No agents directory found.")
        click.echo("Create an agent with: nexus-agent init <name>")
        return

    yaml_files = list(agents_dir.glob("*.yaml")) + list(agents_dir.glob("*.yml"))

    if not yaml_files:
        click.echo("No agents found.")
        click.echo("Create an agent with: nexus-agent init <name>")
        return

    click.echo("📋 Custom Agents:")
    click.echo("")

    for yaml_file in sorted(yaml_files):
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            name = data.get("name", yaml_file.stem)
            display_name = data.get("display_name", name)
            role = data.get("profile", {}).get("role", "Unknown")
            click.echo(f"  • {display_name} (ask_{name})")
            click.echo(f"    Role: {role}")
            click.echo("")
        except Exception as e:
            click.echo(f"  ⚠️  {yaml_file.name}: Failed to load ({e})")
