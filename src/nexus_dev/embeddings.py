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
    """Local Ollama embedding provider with smart batching and text chunking.

    Handles large documents by:
    1. Splitting texts that exceed token limits
    2. Batching requests to avoid memory issues
    """

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
        batch_size: int = 10,
        max_text_tokens: int = 1000,
    ) -> None:
        """Initialize Ollama embedder.

        Args:
            model: Ollama embedding model name.
            base_url: Ollama server URL.
            batch_size: Number of texts per API request (default: 10).
            max_text_tokens: Maximum tokens per text before splitting (default: 1000).
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._max_text_tokens = max_text_tokens
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS_MAP.get(self._model, 768)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text.

        Uses rough approximation: ~4 characters = 1 token for English.
        This is conservative to avoid overshooting actual token limits.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Approximate token count.
        """
        # Rough estimation: 1 token ≈ 4 characters for English
        # This is conservative; actual tokenization may be more efficient
        return max(1, len(text) // 4)

    def _split_text_by_tokens(self, text: str) -> list[str]:
        """Split text if it exceeds token limit.

        Args:
            text: Text to split if needed.

        Returns:
            List of text chunks, each under token limit.
        """
        estimated_tokens = self._estimate_tokens(text)

        # If text is under limit, return as-is
        if estimated_tokens <= self._max_text_tokens:
            return [text]

        # Calculate approximate characters per chunk
        chars_per_chunk = (len(text) // estimated_tokens) * self._max_text_tokens
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chars_per_chunk, len(text))

            # Try to break at a sentence boundary (period + space)
            if end < len(text):
                # Look backwards for a good break point
                sentence_break = text.rfind(". ", start, end)
                if sentence_break > start:
                    end = sentence_break + 2
                else:
                    # Try line break as fallback
                    newline_break = text.rfind("\n", start, end)
                    if newline_break > start:
                        end = newline_break + 1

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            start = end

        return chunks if chunks else [text]  # Fallback to original if splitting failed

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
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using Ollama API.

        Handles large batches by:
        1. Pre-splitting texts that exceed token limits
        2. Batching requests to avoid memory issues

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (in same order as input texts).
        """
        if not texts:
            return []

        client = await self._get_client()

        # Pre-process: split any texts that exceed token limits
        processed_texts: list[str] = []
        text_chunk_mapping: list[list[int]] = []  # Maps original text index to chunk indices

        for text in texts:
            chunks = self._split_text_by_tokens(text)
            chunk_indices = list(range(len(processed_texts), len(processed_texts) + len(chunks)))
            text_chunk_mapping.append(chunk_indices)
            processed_texts.extend(chunks)

        # Process texts in batches to avoid memory issues
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(processed_texts), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(processed_texts))
            batch = processed_texts[batch_start:batch_end]

            # For single text, use the simple endpoint
            if len(batch) == 1:
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": self._model,
                        "input": batch[0],
                    },
                )
            else:
                # For multiple texts, use array input
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": self._model,
                        "input": batch,
                    },
                )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise httpx.HTTPStatusError(
                    f"Ollama embedding request failed for batch of {len(batch)} texts: {e}",
                    request=e.request,
                    response=e.response,
                ) from e

            data = response.json()

            # Extract embeddings based on response format
            if "embeddings" in data:
                batch_embeddings = data["embeddings"]
            elif "embedding" in data:
                # Single embedding response
                batch_embeddings = [data["embedding"]]
            else:
                raise ValueError(f"Unexpected Ollama response format: {data.keys()}")

            all_embeddings.extend(batch_embeddings)

        # Map chunk embeddings back to original texts
        # For texts that were split, average their chunk embeddings
        result_embeddings: list[list[float]] = []
        embedding_dim = len(all_embeddings[0]) if all_embeddings else 0

        for chunk_indices in text_chunk_mapping:
            if len(chunk_indices) == 1:
                # Single chunk, use its embedding directly
                result_embeddings.append(all_embeddings[chunk_indices[0]])
            else:
                # Multiple chunks, average their embeddings
                chunk_embeddings = [all_embeddings[i] for i in chunk_indices]
                avg_embedding = [
                    sum(emb[j] for emb in chunk_embeddings) / len(chunk_embeddings)
                    for j in range(embedding_dim)
                ]
                result_embeddings.append(avg_embedding)

        return result_embeddings

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class VertexAIEmbedder(EmbeddingProvider):
    """Google Vertex AI embedding provider."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        project_id: str | None = None,
        location: str | None = None,
    ) -> None:
        """Initialize Vertex AI embedder.

        Args:
            model: Vertex AI embedding model name.
            project_id: Google Cloud project ID.
            location: Google Cloud region (e.g., "us-central1").
        """
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
        except ImportError:
            raise ImportError(
                "Google Vertex AI dependencies not found. "
                "Please run `pip install nexus-dev[google]`."
            ) from None

        self._model_name = model

        # Initialize Vertex AI SDK if project/location provided or not already initialized
        # User can also rely on gcloud default auth and config
        if project_id or location:
            vertexai.init(project=project_id, location=location)

        try:
            self._model = TextEmbeddingModel.from_pretrained(model)
        except Exception as e:
            raise ValueError(f"Failed to load Vertex AI model '{model}': {e}") from e

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        # Default to 768 for most Vertex models if unknown
        return 768

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Vertex AI has a limit of 5 texts per request for Gecko models,
        but up to 250 for newer models like text-embedding-004.
        We'll use a conservative batch size of 5 for safety or 100 for newer ones.
        """
        if not texts:
            return []

        # Determine batch size based on model
        batch_size = 100 if "text-embedding-004" in self._model_name else 5
        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._model.get_embeddings(list(batch))
            all_embeddings.extend([e.values for e in embeddings])

        return all_embeddings


class BedrockEmbedder(EmbeddingProvider):
    """AWS Bedrock embedding provider."""

    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v1",
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        """Initialize AWS Bedrock embedder.

        Args:
            model: Bedrock model ID.
            region_name: AWS region.
            aws_access_key_id: AWS access key.
            aws_secret_access_key: AWS secret key.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "AWS Bedrock dependencies not found. Please run `pip install nexus-dev[aws]`."
            ) from None

        self._model = model
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        # Defaults
        if "titan-embed-text-v2" in self._model:
            return 1024
        if "titan" in self._model:
            return 1536
        return 1024

    async def embed(self, text: str) -> list[float]:
        import json

        # Bedrock API format varies by model provider (Amazon vs Cohere)
        if "cohere" in self._model:
            body = json.dumps({"texts": [text], "input_type": "search_query"})
        else:
            # Amazon Titan format
            body = json.dumps({"inputText": text})

        response = self._client.invoke_model(
            body=body,
            modelId=self._model,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())

        if "cohere" in self._model:
            return response_body.get("embeddings")[0]
        else:
            return response_body.get("embedding")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Bedrock invoke_model typically handles one string for Titan
        # Cohere models on Bedrock support batching
        if "cohere" in self._model:
            try:
                import json

                body = json.dumps({"texts": texts, "input_type": "search_query"})
                response = self._client.invoke_model(
                    body=body,
                    modelId=self._model,
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = json.loads(response.get("body").read())
                return response_body.get("embeddings")
            except Exception:
                # Fallback to sequential if batch fails
                pass

        # Sequential fallback for Titan or if batching fails
        embeddings = []
        for text in texts:
            embeddings.append(await self.embed(text))
        return embeddings


class VoyageEmbedder(EmbeddingProvider):
    """Voyage AI embedding provider."""

    def __init__(
        self,
        model: str = "voyage-large-2",
        api_key: str | None = None,
    ) -> None:
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "Voyage AI dependencies not found. Please run `pip install nexus-dev[voyage]`."
            ) from None

        self._model = model
        self._client = voyageai.AsyncClient(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return 1536  # Most Voyage models are 1536 (check specific docs if needed)

    async def embed(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Voyage handles batching internally, but we can respect a safe limit
        batch_size = 128
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embed(
                batch,
                model=self._model,
                input_type="document",  # optimized for retrieval
            )
            all_embeddings.extend(list(response.embeddings))

        return all_embeddings


class CohereEmbedder(EmbeddingProvider):
    """Cohere embedding provider."""

    def __init__(
        self,
        model: str = "embed-multilingual-v3.0",
        api_key: str | None = None,
    ) -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere dependencies not found. Please run `pip install nexus-dev[cohere]`."
            ) from None

        self._model = model
        self._client = cohere.AsyncClient(api_key=api_key or os.environ.get("CO_API_KEY"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return 1024  # Default for v3 models

    async def embed(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self._client.embed(
            texts=texts, model=self._model, input_type="search_document", embedding_types=["float"]
        )
        return response.embeddings.float


def validate_embedding_config(config: NexusConfig) -> tuple[bool, str | None]:
    """Validate embedding configuration before creating embedder.

    Checks if the required credentials and configuration are present for the
    selected embedding provider. This allows CLI commands to fail early with
    helpful error messages instead of crashing during embedder initialization.

    Args:
        config: Nexus-Dev configuration.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if config.embedding_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return False, (
                "OpenAI embedding requires OPENAI_API_KEY environment variable. "
                "Set it with: export OPENAI_API_KEY=your-key"
            )
    elif (
        config.embedding_provider == "voyage"
        and not config.voyage_api_key
        and not os.environ.get("VOYAGE_API_KEY")
    ):
        return False, (
            "Voyage AI embedding requires voyage_api_key in config "
            "or VOYAGE_API_KEY environment variable"
        )
    elif (
        config.embedding_provider == "cohere"
        and not config.cohere_api_key
        and not os.environ.get("CO_API_KEY")
    ):
        return False, (
            "Cohere embedding requires cohere_api_key in config or CO_API_KEY environment variable"
        )
    # ollama: No API key required (local server)
    # google: Uses gcloud default credentials (SDK handles it)
    # aws: Uses boto3 default credentials (SDK handles it)
    return True, None


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
            batch_size=config.ollama_batch_size,
            max_text_tokens=config.ollama_max_text_tokens,
        )
    elif config.embedding_provider == "google":
        return VertexAIEmbedder(
            model=config.embedding_model,
            project_id=config.google_project_id,
            location=config.google_location,
        )
    elif config.embedding_provider == "aws":
        return BedrockEmbedder(
            model=config.embedding_model,
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
    elif config.embedding_provider == "voyage":
        return VoyageEmbedder(
            model=config.embedding_model,
            api_key=config.voyage_api_key,
        )
    elif config.embedding_provider == "cohere":
        return CohereEmbedder(
            model=config.embedding_model,
            api_key=config.cohere_api_key,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


# Simple LRU cache for recent embeddings (in-memory)
@lru_cache(maxsize=1000)
def _cached_embedding_key(text: str) -> str:
    """Generate a cache key for embeddings."""
    return text[:500]  # Truncate for cache key efficiency
