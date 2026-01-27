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
