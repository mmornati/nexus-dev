# Supported Embedding Providers

Nexus-Dev supports multiple embedding providers. Choose the one that best fits your needs.

## 1. OpenAI (Default)
- **Best for:** General purpose, ease of use.
- **Provider:** `openai`
- **Default Model:** `text-embedding-3-small`
- **Configuration:**
  ```json
  {
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small"
  }
  ```
- **Environment:** Set `OPENAI_API_KEY`.

## 2. Local Ollama (Privacy / Offline)
- **Best for:** Privacy, local execution, cost savings.
- **Provider:** `ollama`
- **Default Model:** `nomic-embed-text`
- **Configuration:**
  ```json
  {
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    "ollama_url": "http://localhost:11434"
  }
  ```

## 3. Google Vertex AI (Enterprise)
- **Best for:** Enterprise GCP users, high scalability.
- **Provider:** `google`
- **Install:** `pip install nexus-dev[google]`
- **Default Model:** `text-embedding-004`
- **Configuration:**
  ```json
  {
    "embedding_provider": "google",
    "google_project_id": "your-project-id",
    "google_location": "us-central1"
  }
  ```
- **Environment:** Uses standard Google Cloud credentials (ADC).

## 4. AWS Bedrock (Enterprise)
- **Best for:** Enterprise AWS users.
- **Provider:** `aws`
- **Install:** `pip install nexus-dev[aws]`
- **Default Model:** `amazon.titan-embed-text-v1`
- **Configuration:**
  ```json
  {
    "embedding_provider": "aws",
    "aws_region": "us-east-1"
  }
  ```

## 5. Voyage AI (High Performance)
- **Best for:** State-of-the-art retrieval quality (RAG specialist).
- **Provider:** `voyage`
- **Install:** `pip install nexus-dev[voyage]`
- **Default Model:** `voyage-large-2`
- **Configuration:**
  ```json
  {
    "embedding_provider": "voyage",
    "voyage_api_key": "your-key"
  }
  ```

## 6. Cohere (Multilingual)
- **Best for:** Multilingual search and reranking.
- **Provider:** `cohere`
- **Install:** `pip install nexus-dev[cohere]`
- **Default Model:** `embed-multilingual-v3.0`
- **Configuration:**
  ```json
  {
    "embedding_provider": "cohere",
    "cohere_api_key": "your-key"
  }
  ```

> ⚠️ **Warning**: Embeddings are NOT portable between providers. Changing providers requires re-indexing all documents.
