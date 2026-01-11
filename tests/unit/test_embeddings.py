"""Tests for embedding providers with mocked API clients."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.embeddings import (
    OllamaEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


class TestOpenAIEmbedder:
    """Test suite for OpenAI embedding provider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder._api_key == "test-key"
        assert embedder._model == "text-embedding-3-small"

    def test_init_with_env_var(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            embedder = OpenAIEmbedder()
            assert embedder._api_key == "env-key"

    def test_init_without_key_raises(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY if it exists
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIEmbedder()

    def test_model_name_property(self):
        """Test model_name property."""
        embedder = OpenAIEmbedder(model="text-embedding-3-large", api_key="test")
        assert embedder.model_name == "text-embedding-3-large"

    def test_dimensions_property(self):
        """Test dimensions property for different models."""
        embedder_small = OpenAIEmbedder(model="text-embedding-3-small", api_key="test")
        embedder_large = OpenAIEmbedder(model="text-embedding-3-large", api_key="test")
        embedder_ada = OpenAIEmbedder(model="text-embedding-ada-002", api_key="test")
        embedder_unknown = OpenAIEmbedder(model="unknown-model", api_key="test")

        assert embedder_small.dimensions == 1536
        assert embedder_large.dimensions == 3072
        assert embedder_ada.dimensions == 1536
        assert embedder_unknown.dimensions == 1536  # Default

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = OpenAIEmbedder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        result = await embedder.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test embedding multiple texts."""
        embedder = OpenAIEmbedder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        result = await embedder.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Test embedding empty list."""
        embedder = OpenAIEmbedder(api_key="test-key")
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_maintains_order(self):
        """Test that batch embedding maintains order when response is unordered."""
        embedder = OpenAIEmbedder(api_key="test-key")

        # Response comes back in different order
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 2, "embedding": [0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        result = await embedder.embed_batch(["a", "b", "c"])

        # Should be sorted by original order
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the HTTP client."""
        embedder = OpenAIEmbedder(api_key="test-key")
        mock_client = AsyncMock()
        embedder._client = mock_client

        await embedder.close()

        mock_client.aclose.assert_called_once()
        assert embedder._client is None

    @pytest.mark.asyncio
    async def test_get_client_creates_once(self):
        """Test that client is created only once."""
        embedder = OpenAIEmbedder(api_key="test-key")

        with patch("nexus_dev.embeddings.httpx.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_async_client.return_value = mock_client

            client1 = await embedder._get_client()
            client2 = await embedder._get_client()

            assert client1 is client2
            mock_async_client.assert_called_once()


class TestOllamaEmbedder:
    """Test suite for Ollama embedding provider."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        embedder = OllamaEmbedder()
        assert embedder._model == "nomic-embed-text"
        assert embedder._base_url == "http://localhost:11434"

    def test_init_custom(self):
        """Test initialization with custom values."""
        embedder = OllamaEmbedder(model="mxbai-embed-large", base_url="http://custom:8080/")
        assert embedder._model == "mxbai-embed-large"
        assert embedder._base_url == "http://custom:8080"  # Trailing slash removed

    def test_model_name_property(self):
        """Test model_name property."""
        embedder = OllamaEmbedder(model="all-minilm")
        assert embedder.model_name == "all-minilm"

    def test_dimensions_property(self):
        """Test dimensions for different models."""
        assert OllamaEmbedder(model="nomic-embed-text").dimensions == 768
        assert OllamaEmbedder(model="mxbai-embed-large").dimensions == 1024
        assert OllamaEmbedder(model="all-minilm").dimensions == 384
        assert OllamaEmbedder(model="unknown").dimensions == 768  # Default

    @pytest.mark.asyncio
    async def test_embed_with_embeddings_key(self):
        """Test embedding with 'embeddings' response format."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        result = await embedder.embed("test text")

        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_with_embedding_key(self):
        """Test embedding with 'embedding' response format (older Ollama)."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        result = await embedder.embed("test text")

        assert result == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_unexpected_format_raises(self):
        """Test that unexpected response format raises ValueError."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        with pytest.raises(ValueError, match="Unexpected Ollama response format"):
            await embedder.embed("test text")

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        result = await embedder.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Test embedding empty list."""
        embedder = OllamaEmbedder()
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the HTTP client."""
        embedder = OllamaEmbedder()
        mock_client = AsyncMock()
        embedder._client = mock_client

        await embedder.close()

        mock_client.aclose.assert_called_once()
        assert embedder._client is None


class TestCreateEmbedder:
    """Test suite for create_embedder factory function."""

    def test_create_openai_embedder(self):
        """Test creating OpenAI embedder."""
        config = NexusConfig.create_new("test", embedding_provider="openai")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embedder = create_embedder(config)

        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.model_name == "text-embedding-3-small"

    def test_create_ollama_embedder(self):
        """Test creating Ollama embedder."""
        config = NexusConfig.create_new("test", embedding_provider="ollama")
        embedder = create_embedder(config)

        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.model_name == "nomic-embed-text"

    def test_create_ollama_with_custom_url(self):
        """Test creating Ollama embedder with custom URL."""
        config = NexusConfig.create_new("test", embedding_provider="ollama")
        config.ollama_url = "http://custom:8080"

        embedder = create_embedder(config)

        assert isinstance(embedder, OllamaEmbedder)
        assert embedder._base_url == "http://custom:8080"

    def test_create_unsupported_raises(self):
        """Test that unsupported provider raises ValueError."""
        config = NexusConfig.create_new("test")
        config.embedding_provider = "unsupported"

        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            create_embedder(config)

    def test_create_with_custom_model(self):
        """Test creating embedder with custom model."""
        config = NexusConfig.create_new("test", embedding_provider="ollama")
        config.embedding_model = "mxbai-embed-large"

        embedder = create_embedder(config)

        assert embedder.model_name == "mxbai-embed-large"
        assert embedder.dimensions == 1024
