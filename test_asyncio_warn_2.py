import asyncio
from unittest.mock import AsyncMock

def _run_async(coro):
    raise RuntimeError("Fail before awaiting")

try:
    am = AsyncMock(return_value=0)
    _run_async(am())
except Exception as e:
    pass
