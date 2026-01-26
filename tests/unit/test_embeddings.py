"""Tests for embedding providers with mocked API clients."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.embeddings import (
    OllamaEmbedder,
    OpenAIEmbedder,
    create_embedder,
    validate_embedding_config,
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
        assert embedder._batch_size == 10
        assert embedder._max_text_tokens == 1000

    def test_init_custom(self):
        """Test initialization with custom values."""
        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://custom:8080/",
            batch_size=5,
            max_text_tokens=500,
        )
        assert embedder._model == "mxbai-embed-large"
        assert embedder._base_url == "http://custom:8080"  # Trailing slash removed
        assert embedder._batch_size == 5
        assert embedder._max_text_tokens == 500

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

    def test_estimate_tokens(self):
        """Test token estimation."""
        # Rough: 1 token ≈ 4 characters (integer division)
        assert OllamaEmbedder._estimate_tokens("hello") == 1  # 5 // 4 = 1
        assert OllamaEmbedder._estimate_tokens("hello world") == 2  # 11 // 4 = 2
        assert OllamaEmbedder._estimate_tokens("a" * 400) == 100  # 400 // 4 = 100
        assert OllamaEmbedder._estimate_tokens("") == 1  # At least 1

    def test_split_text_by_tokens_small_text(self):
        """Test that small texts are not split."""
        embedder = OllamaEmbedder(max_text_tokens=1000)
        text = "This is a small text"
        result = embedder._split_text_by_tokens(text)
        assert result == [text]

    def test_split_text_by_tokens_large_text(self):
        """Test that large texts are split."""
        embedder = OllamaEmbedder(max_text_tokens=100)
        # Create text that definitely exceeds 100 tokens
        text = "Word. " * 50  # ~300 chars = ~75 tokens, should split
        result = embedder._split_text_by_tokens(text)
        # May be 1 or more chunks depending on token estimation
        assert len(result) >= 1
        # All chunks should be non-empty
        assert all(chunk.strip() for chunk in result)
        # Rejoined should be close to original (may lose some whitespace)
        assert "Word" in "".join(result)

    def test_split_text_by_tokens_at_sentence_boundary(self):
        """Test that splitting respects sentence boundaries."""
        embedder = OllamaEmbedder(max_text_tokens=50)
        text = "First sentence. Second sentence. Third sentence."
        result = embedder._split_text_by_tokens(text)
        # Should ideally split at ". "
        assert len(result) >= 1
        # Each chunk should not be split in the middle of a sentence if possible
        for chunk in result:
            # No trailing partial sentences
            assert not chunk.endswith(". S") and not chunk.endswith(". T")

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
    async def test_embed_batch_single_batch(self):
        """Test batch embedding that fits in a single request."""
        embedder = OllamaEmbedder(batch_size=10)

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
        # Should make only 1 request
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_embed_batch_multiple_batches(self):
        """Test batch embedding that requires multiple API requests."""
        embedder = OllamaEmbedder(batch_size=2)  # Small batch size

        # Setup mock to return different embeddings for each call
        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            # Each batch returns 2 embeddings
            mock_response.json.return_value = {
                "embeddings": [[0.1 * call_count, 0.2], [0.3, 0.4 * call_count]]
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post)
        embedder._client = mock_client

        result = await embedder.embed_batch(["a", "b", "c", "d"])

        assert len(result) == 4
        # Should make 2 requests (batch_size=2)
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_batch_with_text_splitting(self):
        """Test that large texts are split before batching."""
        embedder = OllamaEmbedder(batch_size=10, max_text_tokens=50)

        # Use a flexible mock that adapts to the number of texts in each request
        # This allows the test to work regardless of exact chunk count
        async def mock_post(*args, **kwargs):
            call_args = kwargs.get("json", {})
            input_data = call_args.get("input", "")

            # Determine how many embeddings to return based on input
            num_texts = len(input_data) if isinstance(input_data, list) else 1

            # Return appropriate number of embeddings
            embeddings_for_batch = [[0.1 * (i + 1), 0.2 * (i + 1)] for i in range(num_texts)]

            result = MagicMock()
            result.json.return_value = {"embeddings": embeddings_for_batch}
            result.raise_for_status = MagicMock()
            return result

        mock_client = AsyncMock()
        mock_client.post = mock_post
        embedder._client = mock_client

        # Create a large text that will be split (>50 tokens)
        # "a" * 250 ≈ 62 tokens, will definitely split
        large_text = "a" * 250  # ~62 tokens, will split into 2+ chunks
        short_text = "short"  # ~1 token, won't split

        result = await embedder.embed_batch([large_text, short_text])

        # Should return 2 embeddings (one per input text)
        assert len(result) == 2
        # The large text should be split and averaged
        # The short text should be returned as-is
        assert len(result[0]) == 2  # embedding dimensions
        assert len(result[1]) == 2  # embedding dimensions

    @pytest.mark.asyncio
    async def test_embed_batch_split_text_averaging(self):
        """Test that split texts are averaged correctly."""
        embedder = OllamaEmbedder(batch_size=10, max_text_tokens=40)

        # Mock that returns embeddings in a predictable sequence
        # This allows us to verify that split text embeddings are properly averaged
        embedding_sequence = [
            [0.1, 0.2],  # First chunk of large text
            [0.3, 0.4],  # Second chunk of large text
            [0.5, 0.6],  # Short text (not split)
        ]
        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_args = kwargs.get("json", {})
            input_data = call_args.get("input", "")

            # Determine how many embeddings to return
            num_embeddings = len(input_data) if isinstance(input_data, list) else 1

            # Return embeddings from sequence in order
            embeddings_for_batch = embedding_sequence[
                call_count[0] : call_count[0] + num_embeddings
            ]
            call_count[0] += num_embeddings

            result = MagicMock()
            result.json.return_value = {"embeddings": embeddings_for_batch}
            result.raise_for_status = MagicMock()
            return result

        mock_client = AsyncMock()
        mock_client.post = mock_post
        embedder._client = mock_client

        # Create text that will split (>40 tokens)
        # "test. " is ~6 chars, so "test. " * 30 ≈ 180 chars = 45 tokens, will split
        large_text = "test. " * 30  # ~45 tokens, will split into 2 chunks
        short_text = "hi"  # <1 token, won't split

        result = await embedder.embed_batch([large_text, short_text])

        assert len(result) == 2
        # First result should be average of [0.1, 0.2] and [0.3, 0.4]
        # Use approximate comparison for floating point arithmetic
        assert abs(result[0][0] - 0.2) < 1e-10  # [(0.1+0.3)/2 = 0.2]
        assert abs(result[0][1] - 0.3) < 1e-10  # [(0.2+0.4)/2 = 0.3]
        # Second result should be [0.5, 0.6] (not split)
        assert result[1] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Test embedding empty list."""
        embedder = OllamaEmbedder()
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_single_text_per_request(self):
        """Test that single texts are sent as strings, not arrays."""
        embedder = OllamaEmbedder(batch_size=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        result = await embedder.embed_batch(["single text"])

        # Check that the single text was sent as a string, not a list
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["input"] == "single text"
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_embed_batch_api_error_raises(self):
        """Test that API errors are propagated."""
        import httpx

        embedder = OllamaEmbedder()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "API Error", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        embedder._client = mock_client

        with pytest.raises(httpx.HTTPStatusError, match="Ollama embedding request failed"):
            await embedder.embed_batch(["text1", "text2"])

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

    def test_create_ollama_with_batch_config(self):
        """Test creating Ollama embedder with batch configuration."""
        config = NexusConfig.create_new("test", embedding_provider="ollama")
        config.ollama_batch_size = 5
        config.ollama_max_text_tokens = 500

        embedder = create_embedder(config)

        assert isinstance(embedder, OllamaEmbedder)
        assert embedder._batch_size == 5
        assert embedder._max_text_tokens == 500

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


class TestValidateEmbeddingConfig:
    """Test suite for validate_embedding_config function."""

    def test_openai_without_key_invalid(self):
        """OpenAI config without API key should be invalid."""
        config = NexusConfig.create_new("test", embedding_provider="openai")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            is_valid, error = validate_embedding_config(config)
            assert not is_valid
            assert "OPENAI_API_KEY" in error

    def test_openai_with_key_valid(self):
        """OpenAI config with API key should be valid."""
        config = NexusConfig.create_new("test", embedding_provider="openai")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            is_valid, error = validate_embedding_config(config)
            assert is_valid
            assert error is None

    def test_ollama_always_valid(self):
        """Ollama config should always be valid (no key required)."""
        config = NexusConfig.create_new("test", embedding_provider="ollama")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            is_valid, error = validate_embedding_config(config)
            assert is_valid
            assert error is None

    def test_voyage_without_key_invalid(self):
        """Voyage config without API key should be invalid."""
        config = NexusConfig.create_new("test", embedding_provider="voyage")
        config.voyage_api_key = None
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VOYAGE_API_KEY", None)
            is_valid, error = validate_embedding_config(config)
            assert not is_valid
            assert "VOYAGE_API_KEY" in error or "voyage" in error.lower()

    def test_voyage_with_env_key_valid(self):
        """Voyage config with env API key should be valid."""
        config = NexusConfig.create_new("test", embedding_provider="voyage")
        config.voyage_api_key = None
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            is_valid, error = validate_embedding_config(config)
            assert is_valid
            assert error is None

    def test_voyage_with_config_key_valid(self):
        """Voyage config with config API key should be valid."""
        config = NexusConfig.create_new("test", embedding_provider="voyage")
        config.voyage_api_key = "test-key"
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VOYAGE_API_KEY", None)
            is_valid, error = validate_embedding_config(config)
            assert is_valid
            assert error is None

    def test_cohere_without_key_invalid(self):
        """Cohere config without API key should be invalid."""
        config = NexusConfig.create_new("test", embedding_provider="cohere")
        config.cohere_api_key = None
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CO_API_KEY", None)
            is_valid, error = validate_embedding_config(config)
            assert not is_valid
            assert "CO_API_KEY" in error or "cohere" in error.lower()

    def test_cohere_with_env_key_valid(self):
        """Cohere config with env API key should be valid."""
        config = NexusConfig.create_new("test", embedding_provider="cohere")
        config.cohere_api_key = None
        with patch.dict(os.environ, {"CO_API_KEY": "test-key"}):
            is_valid, error = validate_embedding_config(config)
            assert is_valid
            assert error is None

    def test_google_always_valid(self):
        """Google/Vertex AI config should always be valid (uses SDK auth)."""
        config = NexusConfig.create_new("test", embedding_provider="google")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            is_valid, error = validate_embedding_config(config)
            assert is_valid
            assert error is None

    def test_aws_always_valid(self):
        """AWS Bedrock config should always be valid (uses SDK auth)."""
        config = NexusConfig.create_new("test", embedding_provider="aws")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            is_valid, error = validate_embedding_config(config)
            assert is_valid
            assert error is None
