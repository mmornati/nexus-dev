"""Configuration management for Nexus-Dev."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class NexusConfig:
    """Nexus-Dev project configuration.

    Attributes:
        project_id: Unique identifier for the project (UUID).
        project_name: Human-readable project name.
        embedding_provider: Embedding provider to use ("openai" or "ollama").
        embedding_model: Model name for embeddings.
        ollama_url: URL for local Ollama server.
        ollama_batch_size: Number of texts to embed per Ollama API request (default: 10).
        ollama_max_text_tokens: Maximum tokens per text before splitting (default: 1000).
        db_path: Path to LanceDB database directory.
        include_patterns: Glob patterns for files to index.
        exclude_patterns: Glob patterns for files to exclude.
        docs_folders: Folders containing documentation to index.
    """

    project_id: str
    project_name: str
    embedding_provider: Literal["openai", "ollama", "google", "aws", "voyage", "cohere"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    ollama_url: str = "http://localhost:11434"
    ollama_batch_size: int = 10
    ollama_max_text_tokens: int = 1000
    # Google Vertex AI configuration
    google_project_id: str | None = None
    google_location: str | None = None
    # AWS Bedrock configuration
    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    # Voyage AI configuration
    voyage_api_key: str | None = None
    # Cohere configuration
    cohere_api_key: str | None = None
    db_path: str = "~/.nexus-dev/db"
    include_patterns: list[str] = field(
        default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.java"]
    )
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.git/**",
            "**/.next/**",
        ]
    )
    docs_folders: list[str] = field(
        default_factory=lambda: ["docs/", "documentation/", "README.md"]
    )

    @classmethod
    def create_new(
        cls,
        project_name: str,
        embedding_provider: Literal[
            "openai", "ollama", "google", "aws", "voyage", "cohere"
        ] = "openai",
        embedding_model: str | None = None,
    ) -> NexusConfig:
        """Create a new configuration with a generated project ID.

        Args:
            project_name: Human-readable project name.
            embedding_provider: Embedding provider to use.
            embedding_model: Optional model override.

        Returns:
            New NexusConfig instance.
        """
        # Default model based on provider
        if embedding_model is None:
            embedding_model = (
                "text-embedding-3-small" if embedding_provider == "openai" else "nomic-embed-text"
            )

        # Set default models for other providers if not specified (second fallback)
        if embedding_model is None:
            defaults = {
                "google": "text-embedding-004",
                "aws": "amazon.titan-embed-text-v1",
                "voyage": "voyage-large-2",
                "cohere": "embed-multilingual-v3.0",
            }
            embedding_model = defaults.get(embedding_provider, "nomic-embed-text")

        return cls(
            project_id=str(uuid.uuid4()),
            project_name=project_name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )

    @classmethod
    def load(cls, path: str | Path = "nexus_config.json") -> NexusConfig:
        """Load configuration from a JSON file.

        Args:
            path: Path to the configuration file.

        Returns:
            Loaded NexusConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        if "project_id" not in data:
            raise ValueError("Missing required field: project_id")
        if "project_name" not in data:
            raise ValueError("Missing required field: project_name")

        return cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            embedding_provider=data.get("embedding_provider", "openai"),
            embedding_model=data.get("embedding_model", "text-embedding-3-small"),
            ollama_url=data.get("ollama_url", "http://localhost:11434"),
            ollama_batch_size=data.get("ollama_batch_size", 10),
            ollama_max_text_tokens=data.get("ollama_max_text_tokens", 1000),
            google_project_id=data.get("google_project_id"),
            google_location=data.get("google_location"),
            aws_region=data.get("aws_region"),
            aws_access_key_id=data.get("aws_access_key_id"),
            aws_secret_access_key=data.get("aws_secret_access_key"),
            voyage_api_key=data.get("voyage_api_key"),
            cohere_api_key=data.get("cohere_api_key"),
            db_path=data.get("db_path", "~/.nexus-dev/db"),
            include_patterns=data.get(
                "include_patterns", ["**/*.py", "**/*.js", "**/*.ts", "**/*.java"]
            ),
            exclude_patterns=data.get(
                "exclude_patterns",
                [
                    "**/node_modules/**",
                    "**/.venv/**",
                    "**/venv/**",
                    "**/__pycache__/**",
                ],
            ),
            docs_folders=data.get("docs_folders", ["docs/", "documentation/", "README.md"]),
        )

    @classmethod
    def load_or_default(cls, path: str | Path = "nexus_config.json") -> NexusConfig | None:
        """Load configuration if it exists, otherwise return None.

        Args:
            path: Path to the configuration file.

        Returns:
            NexusConfig instance or None if file doesn't exist.
        """
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None

    def save(self, path: str | Path = "nexus_config.json") -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the configuration file.
        """
        path = Path(path)

        data = {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "ollama_url": self.ollama_url,
            "ollama_batch_size": self.ollama_batch_size,
            "ollama_max_text_tokens": self.ollama_max_text_tokens,
            "google_project_id": self.google_project_id,
            "google_location": self.google_location,
            "aws_region": self.aws_region,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "voyage_api_key": self.voyage_api_key,
            "cohere_api_key": self.cohere_api_key,
            "db_path": self.db_path,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "docs_folders": self.docs_folders,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_db_path(self) -> Path:
        """Get the expanded database path.

        Returns:
            Expanded Path to the database directory.
        """
        # Expand ~ and environment variables
        expanded = os.path.expanduser(self.db_path)
        expanded = os.path.expandvars(expanded)
        return Path(expanded)

    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the configured model.

        Returns:
            Number of dimensions for the embedding model.
        """
        # Known dimensions for common models
        dimensions_map = {
            # OpenAI
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # Ollama
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            # Google
            "text-embedding-004": 768,
            "text-multilingual-embedding-002": 768,
            "textembedding-gecko@003": 768,
            "textembedding-gecko-multilingual@001": 768,
            # AWS Bedrock
            "amazon.titan-embed-text-v1": 1536,
            "amazon.titan-embed-text-v2:0": 1024,
            # Voyage
            "voyage-large-2": 1536,
            "voyage-code-2": 1536,
            "voyage-2": 1024,
            # Cohere
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
        }
        return dimensions_map.get(self.embedding_model, 1536)
