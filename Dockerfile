FROM python:3.13-slim

LABEL maintainer="Marco Mornati <dev.g6tdt@mornati.simplelogin.com>"
LABEL description="Nexus-Dev MCP Server - Persistent Memory for AI Coding Agents"

WORKDIR /app

# Install build dependencies for tree-sitter native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install dependencies
RUN uv pip install --system -e .

# Create data directory for LanceDB
RUN mkdir -p /data/nexus-dev

# Set environment variables
ENV NEXUS_DB_PATH=/data/nexus-dev/db

# Expose port for SSE transport
EXPOSE 8080

# Expose the working directory for config mounting
VOLUME ["/workspace", "/data/nexus-dev"]

WORKDIR /workspace

# Default command runs MCP server via SSE transport for Docker networking
ENTRYPOINT ["python", "-m", "nexus_dev.server"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]
