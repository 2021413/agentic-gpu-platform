"""Distributed lock with an owner token (spec section 14).

Used where two orchestrator replicas must not act at once — reaping the same
worker pool, advancing the same run. Two properties make it a lock rather than
a flag:

* **a TTL**, so a holder that dies does not block the platform forever;
* **an owner token**, so releasing only ever removes *your* lock. Without it,
  a holder whose TTL lapsed while it was paused would come back and delete the
  lock of whoever legitimately took over — the classic way a lock silently
  stops being one.

This is deliberately the single-node algorithm: it assumes one Redis (or one
primary), and it is not Redlock. That matches how the platform deploys Redis,
and the honest scope is worth more than the illusion of a stronger guarantee.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import TracebackType
from typing import cast
from uuid import uuid4

from redis.asyncio import Redis

from infrastructure.redis import scripts
from infrastructure.redis.keys import Keyspace

__all__ = ["DEFAULT_RETRY_INTERVAL", "RedisDistributedLock"]

DEFAULT_RETRY_INTERVAL = timedelta(milliseconds=50)


def _milliseconds(delta: timedelta) -> int:
    return max(int(delta.total_seconds() * 1000), 1)


class RedisDistributedLock:
    """A named, time-bounded, owner-checked lock."""

    __slots__ = ("_extend", "_key", "_redis", "_release", "_retry_interval", "_token", "_ttl")

    def __init__(
        self,
        redis: Redis,
        name: str,
        *,
        ttl: timedelta,
        keys: Keyspace | None = None,
        retry_interval: timedelta = DEFAULT_RETRY_INTERVAL,
    ) -> None:
        self._redis = redis
        self._key = (keys or Keyspace()).lock(name)
        self._ttl = ttl
        self._retry_interval = retry_interval
        self._token: str | None = None
        self._release = redis.register_script(scripts.LOCK_RELEASE)
        self._extend = redis.register_script(scripts.LOCK_EXTEND)

    @property
    def is_held(self) -> bool:
        """Whether *this* object believes it holds the lock.

        Optimistic by nature: the TTL may have lapsed since, which is exactly
        why every write goes through a token check instead of trusting this.
        """
        return self._token is not None

    @property
    def token(self) -> str | None:
        return self._token

    async def acquire(self, *, wait: timedelta | None = None) -> bool:
        """Take the lock, optionally waiting for it.

        A fresh token per acquisition: reusing one would let a previous holder's
        late release apply to the lock it has just re-acquired.
        """
        loop = asyncio.get_running_loop()
        deadline = None if wait is None else loop.time() + wait.total_seconds()
        token = uuid4().hex
        while True:
            acquired = await self._redis.set(self._key, token, nx=True, px=_milliseconds(self._ttl))
            if acquired:
                self._token = token
                return True
            if deadline is None or loop.time() >= deadline:
                return False
            await asyncio.sleep(self._retry_interval.total_seconds())

    async def release(self) -> bool:
        """Release only if still ours. ``False`` means the lock had moved on."""
        if self._token is None:
            return False
        released = cast("int", await self._release(keys=[self._key], args=[self._token]))
        self._token = None
        return bool(released)

    async def extend(self, ttl: timedelta | None = None) -> bool:
        """Push the expiry back while still working. ``False`` means the lock was lost."""
        if self._token is None:
            return False
        extended = cast(
            "int",
            await self._extend(
                keys=[self._key], args=[self._token, _milliseconds(ttl or self._ttl)]
            ),
        )
        if not extended:
            self._token = None
        return bool(extended)

    async def __aenter__(self) -> RedisDistributedLock:
        if not await self.acquire(wait=self._ttl):
            raise TimeoutError(f"could not acquire lock {self._key!r} within its ttl")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.release()
