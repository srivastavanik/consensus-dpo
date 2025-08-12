from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiolimiter import AsyncLimiter


class TokenBucketLimiter:
    """Simple token bucket limiter wrapper.

    Example:
        limiter = TokenBucketLimiter(rate_per_sec=5)
        async with limiter:
            await call()
    """

    def __init__(self, rate_per_sec: float) -> None:
        self._limiter = AsyncLimiter(max_rate=rate_per_sec, time_period=1)

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[None]:  # pragma: no cover - simple wrapper
        async with self._limiter:
            yield

    async def wait(self) -> None:
        async with self._limiter:
            await asyncio.sleep(0)


