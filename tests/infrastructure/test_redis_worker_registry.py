"""One contract suite for ``WorkerRegistry``, run against both implementations.

The registry is what makes the pool dynamic (spec section 4), so these tests
are about discovery rather than storage: who may receive a job right now, and
what happens to a worker that stops answering. ``reap_stale`` gets the most
attention — it is the detection half of the mechanism that frees the jobs of a
worker that vanished, and a registry that fails to notice silence quietly
strands work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from test_redis_support import BACKENDS, MEMORY, T0, connect, later, make_worker

# A fixture is only visible to pytest through the module that names it, so the
# shared Redis endpoint is re-exported here rather than imported for its value.
from test_redis_support import redis_url as redis_url  # noqa: PLC0414

from domain.enums import AgentRole, WorkerStatus
from domain.events.worker import WorkerUnavailable
from domain.ports.worker_registry import WorkerRegistry
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import WorkerLoad
from infrastructure.queue import InMemoryWorkerRegistry
from infrastructure.redis import RedisWorkerRegistry

TTL = timedelta(seconds=30)
TIMEOUT = timedelta(seconds=45)


@pytest.fixture(params=BACKENDS)
async def registry(request: pytest.FixtureRequest) -> AsyncIterator[WorkerRegistry]:
    if request.param == MEMORY:
        yield InMemoryWorkerRegistry()
        return
    client = connect(request.getfixturevalue("redis_url"))
    await client.flushdb()
    try:
        yield RedisWorkerRegistry(client)
    finally:
        await client.flushdb()
        await client.aclose()


# -- registration -------------------------------------------------------
async def test_a_registered_worker_comes_back_whole(registry: WorkerRegistry) -> None:
    """Scheduling decides from capabilities, so all of them must survive."""
    worker = make_worker(concurrency=4, roles=frozenset({AgentRole.CODER}))
    await registry.register(worker, ttl=TTL)

    stored = await registry.get(worker.id)

    assert stored is not None
    assert stored.id == worker.id
    assert stored.endpoint == worker.endpoint
    assert stored.capabilities == worker.capabilities
    assert stored.capabilities.gpu == worker.capabilities.gpu
    assert stored.capabilities.supported_roles == frozenset({AgentRole.CODER})
    assert stored.status is WorkerStatus.READY
    assert stored.load == worker.load
    assert stored.registered_at == T0
    assert stored.last_heartbeat_at == T0
    assert stored.metadata == {"pod": "runpod-7"}


async def test_an_unknown_worker_is_simply_absent(registry: WorkerRegistry) -> None:
    assert await registry.get(WorkerId.generate()) is None
    assert list(await registry.list_all()) == []


async def test_registering_again_replaces_the_record(registry: WorkerRegistry) -> None:
    """A worker that restarts re-registers; that must not create a second entry."""
    worker = make_worker()
    await registry.register(worker, ttl=TTL)
    await registry.register(make_worker(worker_id=worker.id, concurrency=8), ttl=TTL)

    workers = await registry.list_all()

    assert len(workers) == 1
    assert workers[0].capabilities.max_concurrency == 8


async def test_deregistering_removes_the_worker(registry: WorkerRegistry) -> None:
    worker = make_worker()
    await registry.register(worker, ttl=TTL)

    await registry.deregister(worker.id)

    assert await registry.get(worker.id) is None
    assert list(await registry.list_all()) == []


# -- availability -------------------------------------------------------
async def test_availability_excludes_draining_full_and_offline_workers(
    registry: WorkerRegistry,
) -> None:
    ready = make_worker(concurrency=2)
    draining = make_worker(concurrency=2)
    saturated = make_worker(concurrency=1)
    offline = make_worker(concurrency=2)
    for worker in (ready, draining, saturated, offline):
        await registry.register(worker, ttl=TTL)

    draining.start_draining(later(1))
    await registry.update(draining)
    saturated.heartbeat(now=later(1), load=WorkerLoad(active_jobs=1))
    await registry.update(saturated)
    offline.mark_unavailable(now=later(1), reason="probe failed")
    await registry.update(offline)

    available = {worker.id for worker in await registry.list_available()}
    assert available == {ready.id}
    # Everything stays visible: the orchestrator still needs to see a draining
    # worker finish and an offline one come back.
    assert len(await registry.list_all()) == 4


async def test_updating_a_deregistered_worker_does_not_resurrect_it(
    registry: WorkerRegistry,
) -> None:
    worker = make_worker()
    await registry.register(worker, ttl=TTL)
    await registry.deregister(worker.id)

    worker.mark_unhealthy(now=later(1), reason="late")
    await registry.update(worker)

    assert await registry.get(worker.id) is None


# -- heartbeats ---------------------------------------------------------
async def test_heartbeat_refreshes_liveness_and_load(registry: WorkerRegistry) -> None:
    worker = make_worker(concurrency=4)
    await registry.register(worker, ttl=TTL)

    accepted = await registry.heartbeat(
        worker.id, load=WorkerLoad(active_jobs=3, queued_jobs=1), at=later(10), ttl=TTL
    )

    assert accepted is True
    stored = await registry.get(worker.id)
    assert stored is not None
    assert stored.last_heartbeat_at == later(10)
    assert stored.load == WorkerLoad(active_jobs=3, queued_jobs=1)
    assert stored.available_slots == 1


async def test_heartbeat_marks_a_full_worker_busy(registry: WorkerRegistry) -> None:
    worker = make_worker(concurrency=2)
    await registry.register(worker, ttl=TTL)

    await registry.heartbeat(worker.id, load=WorkerLoad(active_jobs=2), at=later(5), ttl=TTL)

    stored = await registry.get(worker.id)
    assert stored is not None
    assert stored.status is WorkerStatus.BUSY
    assert list(await registry.list_available()) == []


async def test_heartbeat_from_an_unknown_worker_is_rejected(
    registry: WorkerRegistry,
) -> None:
    """The worker must learn it has to register again, not be silently accepted."""
    accepted = await registry.heartbeat(
        WorkerId.generate(), load=WorkerLoad(), at=later(1), ttl=TTL
    )

    assert accepted is False


async def test_a_heartbeat_readmits_a_worker_declared_unavailable(
    registry: WorkerRegistry,
) -> None:
    """A network partition must not permanently remove a healthy GPU."""
    worker = make_worker()
    await registry.register(worker, ttl=TTL)
    assert await registry.reap_stale(now=later(120), heartbeat_timeout=TIMEOUT)

    accepted = await registry.heartbeat(worker.id, load=WorkerLoad(), at=later(130), ttl=TTL)

    assert accepted is True
    stored = await registry.get(worker.id)
    assert stored is not None
    assert stored.status is WorkerStatus.READY
    assert {available.id for available in await registry.list_available()} == {worker.id}


# -- reaping ------------------------------------------------------------
async def test_a_silent_worker_is_reaped(registry: WorkerRegistry) -> None:
    silent = make_worker()
    alive = make_worker()
    await registry.register(silent, ttl=TTL)
    await registry.register(alive, ttl=TTL)
    await registry.heartbeat(alive.id, load=WorkerLoad(), at=later(50), ttl=TTL)

    reaped = await registry.reap_stale(now=later(60), heartbeat_timeout=TIMEOUT)

    assert [worker.id for worker in reaped] == [silent.id]
    stored = await registry.get(silent.id)
    assert stored is not None
    assert stored.status is WorkerStatus.OFFLINE
    assert {worker.id for worker in await registry.list_available()} == {alive.id}


async def test_reaping_twice_reports_a_worker_only_once(registry: WorkerRegistry) -> None:
    """Several orchestrators may reap concurrently; only a transition is news."""
    worker = make_worker()
    await registry.register(worker, ttl=TTL)

    first = await registry.reap_stale(now=later(60), heartbeat_timeout=TIMEOUT)
    second = await registry.reap_stale(now=later(61), heartbeat_timeout=TIMEOUT)

    assert [reaped.id for reaped in first] == [worker.id]
    assert list(second) == []


async def test_a_reaped_worker_carries_the_events_it_produced(
    registry: WorkerRegistry,
) -> None:
    """Going offline is news (spec section 15): the caller has to be able to
    publish it, so the buffer must reach it undrained."""
    worker = make_worker()
    await registry.register(worker, ttl=TTL)

    reaped = await registry.reap_stale(now=later(60), heartbeat_timeout=TIMEOUT)

    assert len(reaped) == 1
    events = reaped[0].pending_events
    assert [type(event).name for event in events] == [
        "worker.status_changed",
        "worker.unavailable",
    ]
    assert isinstance(events[1], WorkerUnavailable)
    assert events[1].worker_id == worker.id
    assert events[1].reason == "heartbeat timeout"
    assert events[1].occurred_at == later(60)
    # What the registry keeps is state, never an event waiting to be published
    # twice by whoever reads next.
    stored = await registry.get(worker.id)
    assert stored is not None
    assert stored.pending_events == ()


async def test_reaping_spares_a_worker_within_its_timeout(registry: WorkerRegistry) -> None:
    worker = make_worker()
    await registry.register(worker, ttl=TTL)

    assert list(await registry.reap_stale(now=later(44), heartbeat_timeout=TIMEOUT)) == []
    stored = await registry.get(worker.id)
    assert stored is not None
    assert stored.status is WorkerStatus.READY
