"""Tests for agent templates utility functions."""

from nexus_dev import agent_templates


def test_list_templates_standard(monkeypatch, tmp_path):
    """Test standard listing of templates."""
    # Create mock templates
    (tmp_path / "template1.yaml").touch()
    (tmp_path / "template2.yaml").touch()

    # Mock TEMPLATES_DIR
    monkeypatch.setattr(agent_templates, "TEMPLATES_DIR", tmp_path)

    templates = agent_templates.list_templates()
    assert sorted(templates) == ["template1", "template2"]


def test_list_templates_filtering(monkeypatch, tmp_path):
    """Test filtering of non-yaml files."""
    # Create mock templates and other files
    (tmp_path / "template1.yaml").touch()
    (tmp_path / "readme.md").touch()
    (tmp_path / "template2.json").touch()

    # Mock TEMPLATES_DIR
    monkeypatch.setattr(agent_templates, "TEMPLATES_DIR", tmp_path)

    templates = agent_templates.list_templates()
    assert templates == ["template1"]


def test_list_templates_hidden(monkeypatch, tmp_path):
    """Test hidden files starting with _ are ignored."""
    # Create mock templates including hidden ones
    (tmp_path / "template1.yaml").touch()
    (tmp_path / "_internal.yaml").touch()

    # Mock TEMPLATES_DIR
    monkeypatch.setattr(agent_templates, "TEMPLATES_DIR", tmp_path)

    templates = agent_templates.list_templates()
    assert templates == ["template1"]


def test_list_templates_empty(monkeypatch, tmp_path):
    """Test empty directory returns empty list."""
    # Mock TEMPLATES_DIR with empty dir
    monkeypatch.setattr(agent_templates, "TEMPLATES_DIR", tmp_path)

    templates = agent_templates.list_templates()
    assert templates == []


def test_get_template_path(monkeypatch, tmp_path):
    """Test get_template_path returns correct path."""
    # Mock TEMPLATES_DIR
    monkeypatch.setattr(agent_templates, "TEMPLATES_DIR", tmp_path)

    path = agent_templates.get_template_path("my_template")
    assert path == tmp_path / "my_template.yaml"
