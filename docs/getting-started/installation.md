# Installation

Nexus-Dev can be installed in several ways depending on your use case.

---

## Recommended: Isolated Global Installation

To avoid conflicts with project-specific virtual environments, install Nexus-Dev globally using `pipx` or `uv tool`.

=== "pipx"

    ```bash
    # Install pipx if not already installed
    python -m pip install --user pipx
    python -m pipx ensurepath
    
    # Install nexus-dev
    pipx install nexus-dev
    ```

=== "uv"

    ```bash
    # Install uv if not already installed
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Install nexus-dev as a tool
    uv tool install nexus-dev
    ```

### Verify Installation

```bash
nexus-init --help
```

**Expected output:**

```
Usage: nexus-init [OPTIONS]

  Initialize Nexus-Dev in the current repository.

Options:
  --project-name TEXT          Human-readable name for the project
  --embedding-provider [openai|ollama]
                               Embedding provider to use (default: openai)
  --install-hook / --no-hook   Install pre-commit hook for automatic indexing
  --help                       Show this message and exit.
```

---

## Optional Dependencies

Nexus-Dev supports multiple embedding providers. Install extras as needed:

=== "Google Vertex AI"

    ```bash
    pipx install nexus-dev[google]
    # or
    uv tool install nexus-dev[google]
    ```

=== "AWS Bedrock"

    ```bash
    pipx install nexus-dev[aws]
    ```

=== "Voyage AI"

    ```bash
    pipx install nexus-dev[voyage]
    ```

=== "Cohere"

    ```bash
    pipx install nexus-dev[cohere]
    ```

See [Embedding Providers](configuration.md#embedding-providers) for configuration details.

---

## Docker Installation

For containerized deployments:

```bash
# Pull the image
docker pull ghcr.io/mmornati/nexus-dev:latest

# Run with your project mounted
docker run -it --rm \
    -v /path/to/your-project:/workspace:ro \
    -v nexus-dev-data:/data/nexus-dev \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    ghcr.io/mmornati/nexus-dev:latest
```

See [Docker Deployment](configuration.md#docker-deployment) for detailed configuration.

---

## Development Installation

If you're contributing to Nexus-Dev:

```bash
# Clone repository
git clone https://github.com/mmornati/nexus-dev.git
cd nexus-dev

# Option A: Use the Makefile (handles pyenv + venv)
make setup
source .venv/bin/activate

# Option B: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | Full dev environment setup |
| `make install-dev` | Install with dev dependencies |
| `make lint` | Run ruff linter |
| `make format` | Format code + auto-fix |
| `make check` | Run all CI checks |
| `make test` | Run tests |
| `make test-cov` | Run tests with coverage |

---

## Upgrading

=== "pipx"

    ```bash
    pipx upgrade nexus-dev
    ```

=== "uv"

    ```bash
    uv tool upgrade nexus-dev
    ```

=== "pip"

    ```bash
    pip install --upgrade nexus-dev
    ```

!!! warning "Database Compatibility"
    Major version upgrades may require re-indexing. Check the [changelog](https://github.com/mmornati/nexus-dev/releases) before upgrading.

---

## Next Steps

- [Quick Start](../quickstart.md) - Initialize your first project
- [Configuration Guide](configuration.md) - Set up embedding providers and project options
- [Multi-Repository Projects](../advanced/multi-repo-projects.md) - Setup for mono-repo projects
