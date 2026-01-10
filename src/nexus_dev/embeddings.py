"""Embedding providers for Nexus-Dev.

⚠️ IMPORTANT: Embedding Portability Warning

Embeddings are NOT portable between different models or providers:
- OpenAI text-embedding-3-small produces 1536-dimensional vectors
- Ollama nomic-embed-text produces 768-dimensional vectors
- Different models produce incompatible vector spaces

Once you choose an embedding provider for a project, you MUST keep
using the same provider and model. Changing providers requires
re-indexing ALL documents.

The embedding provider is configured ONCE at MCP server startup via
nexus_config.json and cannot be changed at runtime.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .config import NexusConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vectors."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small by default."""

    DIMENSIONS_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS_MAP.get(self._model, 1536)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text using OpenAI API."""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            httpx.HTTPStatusError: If API request fails.
        """
        if not texts:
            return []

        client = await self._get_client()

        # OpenAI has a limit of ~8000 tokens per request, batch if needed
        batch_size = 100
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            response = await client.post(
                "/embeddings",
                json={
                    "model": self._model,
                    "input": batch,
                },
            )
            response.raise_for_status()

            data = response.json()
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            batch_embeddings = [item["embedding"] for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class OllamaEmbedder(EmbeddingProvider):
    """Local Ollama embedding provider."""

    DIMENSIONS_MAP = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        """Initialize Ollama embedder.

        Args:
            model: Ollama embedding model name.
            base_url: Ollama server URL.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS_MAP.get(self._model, 768)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=120.0,  # Ollama can be slow on first request
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text using Ollama API."""
        client = await self._get_client()

        response = await client.post(
            "/api/embed",
            json={
                "model": self._model,
                "input": text,
            },
        )
        response.raise_for_status()

        data = response.json()
        # Ollama returns embeddings in different formats depending on version
        if "embeddings" in data:
            return data["embeddings"][0]
        elif "embedding" in data:
            return data["embedding"]
        else:
            raise ValueError(f"Unexpected Ollama response format: {data.keys()}")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using Ollama API.

        Note: Ollama processes requests sequentially, so this is slower than OpenAI.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        client = await self._get_client()

        # Ollama supports batch embedding in newer versions
        response = await client.post(
            "/api/embed",
            json={
                "model": self._model,
                "input": texts,
            },
        )
        response.raise_for_status()

        data = response.json()
        if "embeddings" in data:
            return data["embeddings"]

        # Fallback: process one by one for older Ollama versions
        return [await self.embed(text) for text in texts]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def create_embedder(config: NexusConfig) -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Args:
        config: Nexus-Dev configuration.

    Returns:
        Configured embedding provider.

    Raises:
        ValueError: If provider is not supported.
    """
    if config.embedding_provider == "openai":
        return OpenAIEmbedder(model=config.embedding_model)
    elif config.embedding_provider == "ollama":
        return OllamaEmbedder(
            model=config.embedding_model,
            base_url=config.ollama_url,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


# Simple LRU cache for recent embeddings (in-memory)
@lru_cache(maxsize=1000)
def _cached_embedding_key(text: str) -> str:
    """Generate a cache key for embeddings."""
    return text[:500]  # Truncate for cache key efficiency
