"""One contract suite for ``EventBus``, run against both implementations.

What an SSE client depends on (spec section 21) is what is tested here: a
subscriber sees everything published after it subscribed, sees nothing from
other runs, and is released — rather than left hanging — when the run ends.

``ready()`` is the part worth explaining. ``subscribe`` is synchronous, so the
Redis subscriber has no position in the stream until it asks for one; a test
that published immediately after subscribing would be racing it. Both adapters
therefore expose ``ready()``, which resolves once the subscription is pinned,
and after that nothing published can be missed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from test_redis_support import BACKENDS, MEMORY, T0, connect, later

# A fixture is only visible to pytest through the module that names it, so the
# shared Redis endpoint is re-exported here rather than imported for its value.
from test_redis_support import redis_url as redis_url  # noqa: PLC0414

from domain.enums import JobType, RunStatus
from domain.events.base import DomainEvent
from domain.events.job import JobEnqueued
from domain.events.run import RunCompleted, RunStateChanged
from domain.events.worker import WorkerRegistered
from domain.ports.event_bus import EventBus
from domain.value_objects.identifiers import CandidateId, JobId, RunId, WorkerId
from infrastructure.queue import InMemoryEventBus
from infrastructure.redis import RedisEventBus

DELIVERY_TIMEOUT = 10.0
"""Generous on purpose: a blocking read of a remote stream is not instant, and
a flaky suite teaches people to ignore it."""


@pytest.fixture(params=BACKENDS)
async def bus(request: pytest.FixtureRequest) -> AsyncIterator[EventBus]:
    if request.param == MEMORY:
        yield InMemoryEventBus()
        return
    client = connect(request.getfixturevalue("redis_url"))
    await client.flushdb()
    try:
        yield RedisEventBus(client, block=timedelta(milliseconds=50))
    finally:
        await client.flushdb()
        await client.aclose()


def state_changed(run_id: RunId, *, at: float = 0.0) -> RunStateChanged:
    return RunStateChanged(
        occurred_at=later(at),
        run_id=run_id,
        previous=RunStatus.CREATED,
        current=RunStatus.PLANNING,
    )


def job_enqueued(run_id: RunId, *, at: float = 0.0) -> JobEnqueued:
    return JobEnqueued(
        occurred_at=later(at),
        job_id=JobId.generate(),
        run_id=run_id,
        job_type=JobType.CODE,
        attempt=0,
    )


async def next_event(stream: AsyncIterator[DomainEvent]) -> DomainEvent:
    return await asyncio.wait_for(anext(stream), timeout=DELIVERY_TIMEOUT)


async def drain(stream: AsyncIterator[DomainEvent]) -> list[DomainEvent]:
    """Everything the stream yields until it ends by itself."""

    async def _collect() -> list[DomainEvent]:
        return [event async for event in stream]

    return await asyncio.wait_for(_collect(), timeout=DELIVERY_TIMEOUT)


# -- delivery -----------------------------------------------------------
async def test_a_subscriber_receives_what_is_published_after_it_subscribed(
    bus: EventBus,
) -> None:
    run_id = RunId.generate()
    stream = bus.subscribe(run_id)
    await stream.ready()  # type: ignore[attr-defined]

    first = state_changed(run_id, at=1)
    second = job_enqueued(run_id, at=2)
    await bus.publish([first, second])

    assert await next_event(stream) == first
    assert await next_event(stream) == second
    await stream.aclose()  # type: ignore[attr-defined]


async def test_an_event_survives_the_trip_unchanged(bus: EventBus) -> None:
    """Identifiers, enums and timestamps must come back typed, not stringly."""
    run_id = RunId.generate()
    stream = bus.subscribe(run_id)
    await stream.ready()  # type: ignore[attr-defined]
    published = JobEnqueued(
        occurred_at=T0,
        job_id=JobId.generate(),
        run_id=run_id,
        job_type=JobType.STATIC_ANALYSIS,
        attempt=2,
    )

    await bus.publish([published])

    received = await next_event(stream)
    assert isinstance(received, JobEnqueued)
    assert received.event_id == published.event_id
    assert received.occurred_at == published.occurred_at
    assert received.job_id == published.job_id
    assert received.job_type is JobType.STATIC_ANALYSIS
    assert received.attempt == 2
    await stream.aclose()  # type: ignore[attr-defined]


async def test_a_subscriber_never_sees_another_run(bus: EventBus) -> None:
    mine = RunId.generate()
    someone_else = RunId.generate()
    stream = bus.subscribe(mine)
    await stream.ready()  # type: ignore[attr-defined]

    await bus.publish([state_changed(someone_else, at=1), job_enqueued(someone_else, at=2)])
    expected = state_changed(mine, at=3)
    await bus.publish([expected])

    assert await next_event(stream) == expected
    await stream.aclose()  # type: ignore[attr-defined]


async def test_events_without_a_run_do_not_disturb_a_subscription(bus: EventBus) -> None:
    """Worker lifecycle events belong to no run; they must not leak into one."""
    run_id = RunId.generate()
    stream = bus.subscribe(run_id)
    await stream.ready()  # type: ignore[attr-defined]
    registered = WorkerRegistered(
        occurred_at=later(1),
        worker_id=WorkerId.generate(),
        model_id="Qwen3-Coder-30B-A3B",
        endpoint="http://worker-1:8000",
        max_concurrency=2,
    )
    expected = state_changed(run_id, at=2)

    await bus.publish([registered, expected])

    assert await next_event(stream) == expected
    await stream.aclose()  # type: ignore[attr-defined]


async def test_every_subscriber_of_a_run_gets_the_event(bus: EventBus) -> None:
    run_id = RunId.generate()
    first = bus.subscribe(run_id)
    second = bus.subscribe(run_id)
    await first.ready()  # type: ignore[attr-defined]
    await second.ready()  # type: ignore[attr-defined]
    published = state_changed(run_id, at=1)

    await bus.publish([published])

    assert await next_event(first) == published
    assert await next_event(second) == published
    await first.aclose()  # type: ignore[attr-defined]
    await second.aclose()  # type: ignore[attr-defined]


async def test_publishing_nothing_is_allowed(bus: EventBus) -> None:
    await bus.publish([])


# -- termination --------------------------------------------------------
async def test_a_terminal_event_ends_the_subscription(bus: EventBus) -> None:
    """The client gets the last event and the iterator finishes, so an SSE
    response can be closed instead of timing out."""
    run_id = RunId.generate()
    stream = bus.subscribe(run_id)
    await stream.ready()  # type: ignore[attr-defined]
    progress = state_changed(run_id, at=1)
    completion = RunCompleted(
        occurred_at=later(5), run_id=run_id, candidate_id=CandidateId.generate()
    )

    await bus.publish([progress, completion])

    assert await drain(stream) == [progress, completion]


async def test_closing_a_subscription_releases_the_consumer(bus: EventBus) -> None:
    run_id = RunId.generate()
    stream = bus.subscribe(run_id)
    await stream.ready()  # type: ignore[attr-defined]
    consuming = asyncio.ensure_future(next_event(stream))
    await asyncio.sleep(0)

    await stream.aclose()  # type: ignore[attr-defined]

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(consuming, timeout=DELIVERY_TIMEOUT)
