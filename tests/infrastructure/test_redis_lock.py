"""Distributed lock behaviour, against a real Redis.

There is no in-memory twin here on purpose: a lock that only works inside one
process is not a lock, and faking one would test nothing. These are integration
tests, skipped when no Redis is reachable.

The property that matters is the one that is easy to get wrong: a holder whose
lease lapsed must not be able to release — or extend — the lock somebody else
has taken since.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from redis.asyncio import Redis
from test_redis_support import connect

# A fixture is only visible to pytest through the module that names it, so the
# shared Redis endpoint is re-exported here rather than imported for its value.
from test_redis_support import redis_url as redis_url  # noqa: PLC0414

from infrastructure.redis import RedisDistributedLock

pytestmark = pytest.mark.integration

TTL = timedelta(seconds=5)
SHORT = timedelta(milliseconds=200)


@pytest.fixture
async def client(redis_url: str) -> AsyncIterator[Redis]:
    redis = connect(redis_url)
    await redis.flushdb()
    try:
        yield redis
    finally:
        await redis.flushdb()
        await redis.aclose()


def lock(client: Redis, name: str = "run:42", ttl: timedelta = TTL) -> RedisDistributedLock:
    return RedisDistributedLock(client, name, ttl=ttl)


async def test_only_one_holder_at_a_time(client: Redis) -> None:
    first, second = lock(client), lock(client)

    assert await first.acquire() is True
    assert await second.acquire() is False
    assert second.is_held is False

    assert await first.release() is True
    assert await second.acquire() is True
    await second.release()


async def test_releasing_never_touches_someone_elses_lock(client: Redis) -> None:
    """The failure mode this guards: a paused holder wakes up after its ttl and
    deletes the lock of whoever legitimately took over."""
    expiring = lock(client, ttl=SHORT)
    assert await expiring.acquire() is True
    await asyncio.sleep(SHORT.total_seconds() * 2)

    successor = lock(client)
    assert await successor.acquire() is True

    assert await expiring.release() is False
    assert await client.get("agp:lock:run:42") == successor.token
    await successor.release()


async def test_extending_keeps_a_lock_that_would_have_lapsed(client: Redis) -> None:
    held = lock(client, ttl=SHORT)
    assert await held.acquire() is True

    assert await held.extend(TTL) is True
    await asyncio.sleep(SHORT.total_seconds() * 2)

    assert await lock(client).acquire() is False
    await held.release()


async def test_extending_a_lost_lock_reports_the_loss(client: Redis) -> None:
    lost = lock(client, ttl=SHORT)
    assert await lost.acquire() is True
    await asyncio.sleep(SHORT.total_seconds() * 2)
    successor = lock(client)
    await successor.acquire()

    assert await lost.extend() is False
    assert lost.is_held is False
    await successor.release()


async def test_waiting_for_a_lock_gives_up_at_the_deadline(client: Redis) -> None:
    held = lock(client)
    await held.acquire()

    waiter = lock(client)
    acquired = await waiter.acquire(wait=timedelta(milliseconds=150))

    assert acquired is False
    await held.release()


async def test_waiting_succeeds_once_the_holder_lets_go(client: Redis) -> None:
    held = lock(client)
    await held.acquire()

    async def release_soon() -> None:
        await asyncio.sleep(0.05)
        await held.release()

    releasing = asyncio.ensure_future(release_soon())
    waiter = lock(client)

    assert await waiter.acquire(wait=timedelta(seconds=5)) is True
    await releasing
    await waiter.release()


async def test_the_context_manager_always_releases(client: Redis) -> None:
    async with lock(client) as held:
        assert held.is_held is True
        assert await lock(client).acquire() is False

    assert await lock(client).acquire() is True


async def test_a_failure_inside_the_block_still_releases(client: Redis) -> None:
    with pytest.raises(RuntimeError):
        async with lock(client):
            raise RuntimeError("boom")

    assert await lock(client).acquire() is True
