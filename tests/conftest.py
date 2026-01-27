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
