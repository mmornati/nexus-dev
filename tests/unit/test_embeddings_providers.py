import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.embeddings import (
    BedrockEmbedder,
    CohereEmbedder,
    OpenRouterEmbedder,
    VertexAIEmbedder,
    VoyageEmbedder,
    create_embedder,
    validate_embedding_config,
)


# Mock missing dependencies to test graceful failure
def test_missing_dependencies():
    with (
        patch.dict(sys.modules, {"vertexai": None}),
        pytest.raises(ImportError, match=r"nexus-dev\[google\]"),
    ):
        VertexAIEmbedder()

    with (
        patch.dict(sys.modules, {"boto3": None}),
        pytest.raises(ImportError, match=r"nexus-dev\[aws\]"),
    ):
        BedrockEmbedder()

    with (
        patch.dict(sys.modules, {"voyageai": None}),
        pytest.raises(ImportError, match=r"nexus-dev\[voyage\]"),
    ):
        VoyageEmbedder()

    with (
        patch.dict(sys.modules, {"cohere": None}),
        pytest.raises(ImportError, match=r"nexus-dev\[cohere\]"),
    ):
        CohereEmbedder()


# --- Google Vertex AI Tests ---
async def test_vertex_ai_embedder():
    # Mock the module import structure
    mock_vertexai = MagicMock()
    mock_text_embedding_model = MagicMock()

    # Setup mocks
    mock_model_instance = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_model_instance.get_embeddings.return_value = [mock_embedding]
    mock_text_embedding_model.TextEmbeddingModel.from_pretrained.return_value = mock_model_instance

    with patch.dict(
        sys.modules,
        {"vertexai": mock_vertexai, "vertexai.language_models": mock_text_embedding_model},
    ):
        # Re-import or use the class inside the patch context if needed
        # Since the class does runtime import, we just instantiate it
        embedder = VertexAIEmbedder(project_id="test-project", location="us-central1")

        # Verify init
        mock_vertexai.init.assert_called_with(project="test-project", location="us-central1")
        assert embedder.model_name == "text-embedding-004"
        assert embedder.dimensions == 768

        # Test embed
        vec = await embedder.embed("hello")
        assert vec == [0.1, 0.2, 0.3]
        mock_model_instance.get_embeddings.assert_called()


# --- AWS Bedrock Tests ---
async def test_bedrock_embedder_titan():
    mock_boto3 = MagicMock()
    mock_client_instance = MagicMock()
    mock_boto3.client.return_value = mock_client_instance

    # Mock response
    import json
    from io import BytesIO

    response_body = {"embedding": [0.1, 0.2, 0.3]}
    mock_response = {"body": BytesIO(json.dumps(response_body).encode("utf-8"))}
    mock_client_instance.invoke_model.return_value = mock_response

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        embedder = BedrockEmbedder(model="amazon.titan-embed-text-v1", region_name="us-east-1")

        assert embedder.dimensions == 1536

        # Test embed
        vec = await embedder.embed("hello")
        assert vec == [0.1, 0.2, 0.3]

        # Verify arguments
        call_args = mock_client_instance.invoke_model.call_args[1]
        assert json.loads(call_args["body"]) == {"inputText": "hello"}


# --- Voyage AI Tests ---
async def test_voyage_embedder():
    mock_voyageai = MagicMock()
    mock_client_instance = AsyncMock()
    mock_voyageai.AsyncClient.return_value = mock_client_instance

    # Mock response object
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3]]
    mock_client_instance.embed.return_value = mock_response

    with patch.dict(sys.modules, {"voyageai": mock_voyageai}):
        embedder = VoyageEmbedder(api_key="test-key")

        # Test embed
        vec = await embedder.embed("hello")
        assert vec == [0.1, 0.2, 0.3]

        # Verify call
        mock_client_instance.embed.assert_called_with(
            ["hello"], model="voyage-large-2", input_type="document"
        )


# --- Cohere Tests ---
async def test_cohere_embedder():
    mock_cohere = MagicMock()
    mock_client_instance = AsyncMock()
    mock_cohere.AsyncClient.return_value = mock_client_instance

    # Mock response object
    mock_response = MagicMock()
    mock_response.embeddings.float = [[0.9, 0.8, 0.7]]
    mock_client_instance.embed.return_value = mock_response

    with patch.dict(sys.modules, {"cohere": mock_cohere}):
        embedder = CohereEmbedder(api_key="test-key")

        # Test embed
        vec = await embedder.embed("hello")
        assert vec == [0.9, 0.8, 0.7]  # Cohere returns list of lists for batch

        # Verify call
        mock_client_instance.embed.assert_called()


# --- Factory Test ---
def test_create_embedder_factory():
    # Test Google
    config = NexusConfig.create_new("test", embedding_provider="google")
    config.google_project_id = "p-id"

    # Mock VertexAI for factory since it imports it
    mock_vertexai = MagicMock()
    mock_text_embedding_model = MagicMock()

    with (
        patch.dict(
            sys.modules,
            {"vertexai": mock_vertexai, "vertexai.language_models": mock_text_embedding_model},
        ),
        patch("nexus_dev.embeddings.VertexAIEmbedder", return_value=MagicMock()) as mock_vertex,
    ):
        # We need to make sure import inside __init__ succeeds
        create_embedder(config)
        mock_vertex.assert_called()

    # Test OpenRouter
    config = NexusConfig(
        project_id="test",
        project_name="test",
        embedding_provider="openrouter",
        embedding_model="openai/text-embedding-3-small",
        openrouter_api_key="test-key",
    )

    embedder = create_embedder(config)
    assert isinstance(embedder, OpenRouterEmbedder)
    assert embedder.model_name == "openai/text-embedding-3-small"


# --- OpenRouter Tests ---
def test_openrouter_embedder_init():
    """Test OpenRouter embedder initialization."""
    embedder = OpenRouterEmbedder(api_key="test-key")

    assert embedder.model_name == "openai/text-embedding-3-small"
    assert embedder.dimensions == 1536


def test_openrouter_embedder_init_with_model():
    """Test OpenRouter embedder with custom model."""
    embedder = OpenRouterEmbedder(model="cohere/embed-multilingual-v3.0", api_key="test-key")

    assert embedder.model_name == "cohere/embed-multilingual-v3.0"
    assert embedder.dimensions == 1024


def test_openrouter_embedder_requires_api_key():
    """Test that OpenRouter embedder requires API key."""
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbedder()


def test_openrouter_embedder_env_api_key(monkeypatch):
    """Test that OpenRouter embedder uses environment variable."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-test-key")
    embedder = OpenRouterEmbedder()

    assert embedder._api_key == "env-test-key"


async def test_openrouter_embedder_embed():
    """Test OpenRouter embed single text."""
    embedder = OpenRouterEmbedder(api_key="test-key")

    mock_response = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}

    mock_response_obj = MagicMock()
    mock_response_obj.json.return_value = mock_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    embedder._client = mock_client

    vec = await embedder.embed("hello")
    assert vec == [0.1, 0.2, 0.3]


async def test_openrouter_embedder_batch():
    """Test OpenRouter embed batch of texts."""
    embedder = OpenRouterEmbedder(api_key="test-key")

    mock_response = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
        ]
    }

    mock_response_obj = MagicMock()
    mock_response_obj.json.return_value = mock_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response_obj
    embedder._client = mock_client

    vectors = await embedder.embed_batch(["hello", "world"])
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


async def test_openrouter_embedder_close():
    """Test OpenRouter embedder close."""
    embedder = OpenRouterEmbedder(api_key="test-key")

    mock_client = AsyncMock()
    mock_client.aclose = AsyncMock()
    embedder._client = mock_client

    await embedder.close()
    mock_client.aclose.assert_called_once()


# --- Validation Tests ---
def test_validate_openrouter_config_with_key():
    """Test validation passes with openrouter API key in config."""
    config = NexusConfig(
        project_id="test",
        project_name="test",
        embedding_provider="openrouter",
        openrouter_api_key="test-key",
    )

    is_valid, error = validate_embedding_config(config)
    assert is_valid is True
    assert error is None


def test_validate_openrouter_config_with_env(monkeypatch):
    """Test validation passes with OPENROUTER_API_KEY env var."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    config = NexusConfig(project_id="test", project_name="test", embedding_provider="openrouter")

    is_valid, error = validate_embedding_config(config)
    assert is_valid is True
    assert error is None


def test_validate_openrouter_config_missing_key():
    """Test validation fails without openrouter API key."""
    config = NexusConfig(project_id="test", project_name="test", embedding_provider="openrouter")

    is_valid, error = validate_embedding_config(config)
    assert is_valid is False
    assert error is not None
    assert "OPENROUTER_API_KEY" in error
