# Nexus-Dev Makefile
# ====================
# Manage development environment, build, test, and Docker operations

.PHONY: help setup install install-dev test lint format type-check check build \
        docker-build docker-run docker-stop docker-logs clean distclean

# Default Python version
PYTHON_VERSION ?= 3.13
VENV_DIR := .venv
DOCKER_IMAGE := nexus-dev
DOCKER_CONTAINER := nexus-dev-local

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo "$(BLUE)Nexus-Dev Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Environment Setup:$(NC)"
	@grep -E '^(setup|install|install-dev):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@grep -E '^(lint|format|type-check|check):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@grep -E '^(test|test-cov):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Build & Package:$(NC)"
	@grep -E '^(build|publish):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@grep -E '^(docker-.*):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@grep -E '^(clean|distclean):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------

setup: ## Setup complete dev environment (pyenv + venv + dependencies)
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if command -v pyenv &> /dev/null; then \
		echo "$(GREEN)✓ pyenv found$(NC)"; \
		pyenv install -s $(PYTHON_VERSION); \
		pyenv local $(PYTHON_VERSION); \
	else \
		echo "$(YELLOW)⚠ pyenv not found, using system Python$(NC)"; \
	fi
	@python3 -m venv $(VENV_DIR)
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install -e ".[dev]"
	@echo ""
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo ""
	@echo "Activate with: $(YELLOW)source $(VENV_DIR)/bin/activate$(NC)"

$(VENV_DIR)/bin/activate:
	@python3 -m venv $(VENV_DIR)

install: $(VENV_DIR)/bin/activate ## Install package in development mode
	@echo "$(BLUE)Installing package...$(NC)"
	@$(VENV_DIR)/bin/pip install -e .
	@echo "$(GREEN)✓ Package installed$(NC)"

install-dev: $(VENV_DIR)/bin/activate ## Install package with dev dependencies
	@echo "$(BLUE)Installing package with dev dependencies...$(NC)"
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install -e ".[dev]"
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

#------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------

lint: ## Run linter (ruff) - will fail on errors
	@echo "$(BLUE)Running linter...$(NC)"
	@$(VENV_DIR)/bin/ruff check src/ tests/
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code (ruff format)
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(VENV_DIR)/bin/ruff format src/ tests/
	@$(VENV_DIR)/bin/ruff check --fix src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without modifying (CI-style)
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@$(VENV_DIR)/bin/ruff format --check src/ tests/
	@echo "$(GREEN)✓ Formatting check complete$(NC)"

type-check: ## Run type checker (mypy)
	@echo "$(BLUE)Running type checker...$(NC)"
	@$(VENV_DIR)/bin/mypy src/
	@echo "$(GREEN)✓ Type checking complete$(NC)"

check: lint format-check type-check ## Run all code quality checks (CI-aligned)

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	@$(VENV_DIR)/bin/pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(VENV_DIR)/bin/pytest tests/ -v --cov=nexus_dev --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

#------------------------------------------------------------------------------
# Build & Package
#------------------------------------------------------------------------------

build: ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	@$(VENV_DIR)/bin/pip install build
	@$(VENV_DIR)/bin/python -m build
	@echo "$(GREEN)✓ Packages built in dist/$(NC)"

publish: build ## Publish to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD)
	@echo "$(BLUE)Publishing to PyPI...$(NC)"
	@$(VENV_DIR)/bin/pip install twine
	@$(VENV_DIR)/bin/twine upload dist/*
	@echo "$(GREEN)✓ Published to PyPI$(NC)"

#------------------------------------------------------------------------------
# Docker
#------------------------------------------------------------------------------

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):latest .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE):latest$(NC)"

docker-run: ## Run local Docker container for testing
	@echo "$(BLUE)Starting Docker container...$(NC)"
	@docker rm -f $(DOCKER_CONTAINER) 2>/dev/null || true
	docker run -d \
		--name $(DOCKER_CONTAINER) \
		-v $(PWD):/workspace:ro \
		-v nexus-dev-data:/data/nexus-dev \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-it \
		$(DOCKER_IMAGE):latest
	@echo "$(GREEN)✓ Container started: $(DOCKER_CONTAINER)$(NC)"
	@echo ""
	@echo "To interact: $(YELLOW)docker attach $(DOCKER_CONTAINER)$(NC)"
	@echo "To view logs: $(YELLOW)docker logs -f $(DOCKER_CONTAINER)$(NC)"

docker-shell: ## Open shell in running container
	@echo "$(BLUE)Opening shell in container...$(NC)"
	docker exec -it $(DOCKER_CONTAINER) /bin/bash

docker-stop: ## Stop and remove Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	@docker stop $(DOCKER_CONTAINER) 2>/dev/null || true
	@docker rm $(DOCKER_CONTAINER) 2>/dev/null || true
	@echo "$(GREEN)✓ Container stopped$(NC)"

docker-logs: ## Show Docker container logs
	docker logs -f $(DOCKER_CONTAINER)

docker-test: docker-build ## Build and run Docker container for testing
	@echo "$(BLUE)Testing Docker container...$(NC)"
	docker run --rm $(DOCKER_IMAGE):latest python -c "from nexus_dev import server; print('✓ Server module loaded successfully')"
	@echo "$(GREEN)✓ Docker test passed$(NC)"

docker-clean: ## Remove Docker image and volumes
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	@docker stop $(DOCKER_CONTAINER) 2>/dev/null || true
	@docker rm $(DOCKER_CONTAINER) 2>/dev/null || true
	@docker rmi $(DOCKER_IMAGE):latest 2>/dev/null || true
	@docker volume rm nexus-dev-data 2>/dev/null || true
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------

clean: ## Remove build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf src/*.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .ruff_cache/
	@rm -rf .mypy_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Clean complete$(NC)"

distclean: clean docker-clean ## Remove everything including venv and Docker
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)✓ Distclean complete$(NC)"

#------------------------------------------------------------------------------
# Development Shortcuts
#------------------------------------------------------------------------------

run: ## Run MCP server locally (for testing with MCP Inspector)
	@echo "$(BLUE)Starting Nexus-Dev MCP server...$(NC)"
	@$(VENV_DIR)/bin/python -m nexus_dev.server

init-here: ## Initialize Nexus-Dev in current directory
	@$(VENV_DIR)/bin/nexus-init --project-name "nexus-dev" --embedding-provider openai

index-self: ## Index this project's source code
	@echo "$(BLUE)Indexing Nexus-Dev source code...$(NC)"
	@$(VENV_DIR)/bin/nexus-index src/ README.md CONTRIBUTING.md -r
	@echo "$(GREEN)✓ Self-indexing complete$(NC)"
