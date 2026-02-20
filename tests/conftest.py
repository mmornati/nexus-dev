"""Pytest configuration for nexus-dev."""

from pathlib import Path

import pytest

# Monkeypatch redislite/falkordb compatibility issues
try:
    import redis
    import redislite.client

    # 1. Fix AttributeError in __del__
    # redislite tries to access self.connection_pool in cleanup,
    # which might not exist or be accessible
    original_cleanup = redislite.client.RedisMixin._cleanup

    def patched_cleanup(self, *args, **kwargs):
        try:
            original_cleanup(self, *args, **kwargs)
        except AttributeError:
            pass
        except Exception:
            pass

    redislite.client.RedisMixin._cleanup = patched_cleanup

    # 2. Fix TypeError in __init__
    # redislite passes 'dir' and other args to redis.Redis, which strict Redis 5+ rejects.
    # We patch redis.Redis.__init__ to ignore these specific args.

    original_redis_init = redis.Redis.__init__

    def patched_redis_init(self, *args, **kwargs):
        # Remove arguments that redislite passes but redis doesn't accept
        kwargs.pop("dir", None)
        kwargs.pop("dbfilename", None)
        kwargs.pop("serverconfig", None)  # Just in case

        original_redis_init(self, *args, **kwargs)

    redis.Redis.__init__ = patched_redis_init
    # Also patch StrictRedis if used explicitly
    redis.StrictRedis.__init__ = patched_redis_init

except ImportError:
    pass


@pytest.fixture
def tmp_path_str(tmp_path: Path) -> str:
    """Return string representation of tmp_path."""
    return str(tmp_path)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Alias for tmp_path for backwards compatibility."""
    return tmp_path


# =============================================================================
# Module-scoped FalkorDB fixtures for performance optimization
# =============================================================================
# Starting a FalkorDB/redislite server takes ~5-12 seconds to close.
# By using module-scoped fixtures, we start the server once per module
# and use flushdb() between tests instead of full server restart.
# This reduces test time from ~200s to ~20s for 27 tests.
# =============================================================================


@pytest.fixture(scope="module")
def shared_falkor_server(tmp_path_factory):
    """Module-scoped FalkorDB server (shared across all tests in a module).

    This fixture is expensive to create (~1s) and expensive to teardown (~7s).
    By sharing it across an entire test module, we avoid repeated server restarts.
    """
    from redislite import FalkorDB

    tmpdir = tmp_path_factory.mktemp("falkor_shared")
    server = FalkorDB(dir=str(tmpdir))
    yield server
    server.close()


@pytest.fixture
def redis_client(shared_falkor_server):
    """Per-test Redis client with automatic cleanup.

    Uses flushdb() for fast cleanup (~0.001s) instead of server restart (~7s).
    """
    client = shared_falkor_server.client
    yield client
    # Fast cleanup: flush database instead of restarting server
    client.flushdb()


@pytest.fixture
def graph_client(shared_falkor_server):
    """Per-test FalkorDB graph client with automatic cleanup.

    Uses graph.delete() + flushdb() for fast cleanup instead of server restart.
    """
    server = shared_falkor_server
    yield server
    # Clean up all graphs and data
    try:
        server.client.flushdb()
    except Exception:
        pass


@pytest.fixture
def nexus_config_dict():
    """Provide a sample NexusConfig as dict for testing."""
    import uuid

    return {
        "project_id": str(uuid.uuid4()),
        "project_name": "test-project",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "db_path": "~/.nexus-dev/db",
        "include_patterns": ["**/*.py", "**/*.js"],
        "exclude_patterns": ["**/node_modules/**"],
        "docs_folders": ["docs/", "README.md"],
    }


@pytest.fixture
def sample_python_code():
    """Provide sample Python code for chunker tests."""
    return '''
def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """A simple calculator class."""

    def __init__(self, name: str = "Calculator"):
        """Initialize calculator with a name."""
        self.name = name
        self.result = 0

    def add(self, x: int, y: int) -> int:
        """Add two numbers and store result."""
        self.result = x + y
        return self.result

    def subtract(self, x: int, y: int) -> int:
        """Subtract two numbers and store result."""
        self.result = x - y
        return self.result


async def async_fetch(url: str) -> str:
    """Fetch data from a URL asynchronously."""
    return f"Data from {url}"
'''


@pytest.fixture
def sample_markdown():
    """Provide sample Markdown for documentation chunker tests."""
    return """# Project Documentation

This is the main documentation for the project.

## Installation

To install this project, run:

```bash
pip install project-name
```

## Configuration

Configure the project with the following options:

### Database Settings

Set up your database connection:

```ini
[database]
host = localhost
port = 5432
```

## Usage

Use the project like this:

```python
from project import main
main()
```
"""


@pytest.fixture
def sample_rst():
    """Provide sample RST for documentation chunker tests."""
    return """
Project Documentation
=====================

This is the main documentation for the project.

Installation
============

To install this project, run::

    pip install project-name

Configuration
=============

Configure the project with the following options:

Database Settings
-----------------

Set up your database connection::

    [database]
    host = localhost
    port = 5432

Usage
=====

Use the project like this::

    from project import main
    main()
"""


@pytest.fixture
def sample_java_code():
    """Provide sample Java code for chunker tests."""
    return """
/**
 * A simple calculator class for basic math operations.
 */
public class Calculator {
    private int result;

    public Calculator(int initial) {
        this.result = initial;
    }

    public int add(int x) {
        result = result + x;
        return result;
    }

    public int subtract(int x) {
        result = result - x;
        return result;
    }

    public int getResult() {
        return result;
    }
}
"""


@pytest.fixture
def sample_javascript_code():
    """Provide sample JavaScript code for chunker tests."""
    return """
// Arrow function
const add = (a, b) => a + b;

// Function declaration
function multiply(a, b) {
    return a * b;
}

// Class
class Calculator {
    constructor(initial = 0) {
        this.result = initial;
    }

    add(x) {
        this.result += x;
        return this.result;
    }

    subtract(x) {
        this.result -= x;
        return this.result;
    }

    getResult() {
        return this.result;
    }
}

// Export
module.exports = { add, multiply, Calculator };
"""


@pytest.fixture
def sample_typescript_code():
    """Provide sample TypeScript code for chunker tests."""
    return """
// Typed arrow function
const add = (a: number, b: number): number => a + b;

// Typed function
function multiply(a: number, b: number): number {
    return a * b;
}

// Typed class
class Calculator {
    private value: number;

    constructor(initial: number = 0) {
        this.value = initial;
    }

    add(x: number): number {
        this.value += x;
        return this.value;
    }

    subtract(x: number): number {
        this.value -= x;
        return this.value;
    }

    getValue(): number {
        return this.value;
    }
}

// Export
export { add, multiply, Calculator };
"""
