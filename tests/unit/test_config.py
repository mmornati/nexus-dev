"""Tests for Nexus-Dev configuration module."""

import json

import pytest

from nexus_dev.config import NexusConfig


class TestNexusConfig:
    """Test suite for NexusConfig."""

    def test_create_new_config(self):
        """Test creating a new configuration."""
        config = NexusConfig.create_new("my-project")

        assert config.project_name == "my-project"
        assert config.project_id  # Should have UUID
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"

    def test_create_new_with_ollama(self):
        """Test creating config with Ollama provider."""
        config = NexusConfig.create_new("my-project", embedding_provider="ollama")

        assert config.embedding_provider == "ollama"
        assert config.embedding_model == "nomic-embed-text"

    def test_create_new_with_custom_model(self):
        """Test creating config with custom model."""
        config = NexusConfig.create_new(
            "my-project",
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
        )

        assert config.embedding_model == "text-embedding-3-large"

    def test_save_and_load(self, temp_dir):
        """Test saving and loading configuration."""
        config = NexusConfig.create_new("test-project")
        config_path = temp_dir / "nexus_config.json"

        config.save(config_path)
        assert config_path.exists()

        loaded = NexusConfig.load(config_path)
        assert loaded.project_id == config.project_id
        assert loaded.project_name == config.project_name
        assert loaded.embedding_provider == config.embedding_provider

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            NexusConfig.load(temp_dir / "nonexistent.json")

    def test_load_or_default_exists(self, temp_dir):
        """Test load_or_default when file exists."""
        config = NexusConfig.create_new("test-project")
        config_path = temp_dir / "nexus_config.json"
        config.save(config_path)

        loaded = NexusConfig.load_or_default(config_path)
        assert loaded is not None
        assert loaded.project_name == "test-project"

    def test_load_or_default_missing(self, temp_dir):
        """Test load_or_default when file doesn't exist."""
        result = NexusConfig.load_or_default(temp_dir / "missing.json")
        assert result is None

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON file."""
        config_path = temp_dir / "invalid.json"
        config_path.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            NexusConfig.load(config_path)

    def test_load_missing_required_field(self, temp_dir):
        """Test loading config with missing required fields."""
        config_path = temp_dir / "incomplete.json"
        config_path.write_text('{"project_name": "test"}')

        with pytest.raises(ValueError, match="project_id"):
            NexusConfig.load(config_path)

    def test_get_db_path_expands_tilde(self):
        """Test that database path expands ~ correctly."""
        config = NexusConfig.create_new("test")
        config.db_path = "~/.nexus-dev/db"

        db_path = config.get_db_path()
        assert "~" not in str(db_path)
        assert db_path.is_absolute()

    def test_get_embedding_dimensions_openai(self):
        """Test getting dimensions for OpenAI models."""
        config = NexusConfig.create_new("test")
        config.embedding_model = "text-embedding-3-small"

        assert config.get_embedding_dimensions() == 1536

    def test_get_embedding_dimensions_ollama(self):
        """Test getting dimensions for Ollama models."""
        config = NexusConfig.create_new("test", embedding_provider="ollama")
        config.embedding_model = "nomic-embed-text"

        assert config.get_embedding_dimensions() == 768

    def test_get_embedding_dimensions_unknown_model(self):
        """Test getting dimensions for unknown model defaults to 1536."""
        config = NexusConfig.create_new("test")
        config.embedding_model = "unknown-model"

        assert config.get_embedding_dimensions() == 1536

    def test_default_patterns(self):
        """Test that default patterns are set correctly."""
        config = NexusConfig.create_new("test")

        assert "**/*.py" in config.include_patterns
        assert "**/*.js" in config.include_patterns
        assert "**/node_modules/**" in config.exclude_patterns
        assert "**/__pycache__/**" in config.exclude_patterns

    def test_default_docs_folders(self):
        """Test that default docs folders are set."""
        config = NexusConfig.create_new("test")

        assert "docs/" in config.docs_folders
        assert "README.md" in config.docs_folders

    def test_config_serialization_roundtrip(self, temp_dir, nexus_config_dict):
        """Test that all fields survive save/load cycle."""
        config = NexusConfig(**nexus_config_dict)
        config_path = temp_dir / "nexus_config.json"

        config.save(config_path)
        loaded = NexusConfig.load(config_path)

        assert loaded.project_id == nexus_config_dict["project_id"]
        assert loaded.project_name == nexus_config_dict["project_name"]
        assert loaded.embedding_provider == nexus_config_dict["embedding_provider"]
        assert loaded.embedding_model == nexus_config_dict["embedding_model"]
        assert loaded.db_path == nexus_config_dict["db_path"]
