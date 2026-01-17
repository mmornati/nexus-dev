"""Agent templates package."""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent


def get_template_path(template_name: str) -> Path:
    """Get the path to a template file.

    Args:
        template_name: Name of the template (without .yaml extension).

    Returns:
        Path to the template YAML file.
    """
    return TEMPLATES_DIR / f"{template_name}.yaml"


def list_templates() -> list[str]:
    """List all available template names.

    Returns:
        List of template names (without .yaml extension).
    """
    return [p.stem for p in TEMPLATES_DIR.glob("*.yaml") if not p.name.startswith("_")]
