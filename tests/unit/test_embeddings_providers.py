import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_dev.config import NexusConfig
from nexus_dev.embeddings import (
    BedrockEmbedder,
    CohereEmbedder,
    VertexAIEmbedder,
    VoyageEmbedder,
    create_embedder,
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
