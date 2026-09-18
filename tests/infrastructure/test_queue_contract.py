"""One contract suite, run against both ``JobQueue`` implementations.

The point of writing it once and parameterising it is that "the in-memory
adapter behaves like Redis" stops being a claim and becomes something the build
checks. Every test here states a property the orchestrator depends on — exactly
one consumer per job, a lease that lapses, priority that is respected — and
both adapters have to satisfy it identically.

The Redis variants carry the ``integration`` marker and are skipped, never
failed, when no Redis is reachable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from test_redis_support import (
    BACKENDS,
    MEMORY,
    T0,
    connect,
    later,
    make_job,
)

# A fixture is only visible to pytest through the module that names it, so the
# shared Redis endpoint is re-exported here rather than imported for its value.
from test_redis_support import redis_url as redis_url  # noqa: PLC0414

from domain.entities.job import Job
from domain.enums import JobStatus, JobType, Priority
from domain.exceptions import JobLeaseExpiredError
from domain.ports.job_queue import JobQueue
from domain.value_objects.identifiers import RunId, WorkerId
from domain.value_objects.lease import Lease, LeaseToken
from infrastructure.queue import InMemoryJobQueue, RedisJobQueue

LEASE = timedelta(seconds=30)
CODE = [JobType.CODE]


@pytest.fixture(params=BACKENDS)
async def queue(request: pytest.FixtureRequest) -> AsyncIterator[JobQueue]:
    if request.param == MEMORY:
        yield InMemoryJobQueue()
        return
    client = connect(request.getfixturevalue("redis_url"))
    await client.flushdb()
    try:
        yield RedisJobQueue(client)
    finally:
        await client.flushdb()
        await client.aclose()


@pytest.fixture
def consumer() -> WorkerId:
    return WorkerId.generate()


async def claim(
    queue: JobQueue,
    consumer: WorkerId,
    *,
    at: float = 0.0,
    types: list[JobType] | None = None,
) -> tuple[Job, Lease] | None:
    """Claim with the suite's defaults, ``at`` seconds after ``T0``."""
    return await queue.claim(
        consumer=consumer,
        job_types=types or CODE,
        lease_duration=LEASE,
        now=later(at),
    )


# -- claiming -----------------------------------------------------------
async def test_claiming_an_empty_queue_yields_nothing(queue: JobQueue, consumer: WorkerId) -> None:
    assert await claim(queue, consumer) is None


async def test_a_claimed_job_comes_back_whole(queue: JobQueue, consumer: WorkerId) -> None:
    """Everything the worker needs must survive the queue, not just the id."""
    job = make_job(payload={"objective": "add retries", "files": ["a.py"]})
    await queue.enqueue(job)

    claimed = await claim(queue, consumer)

    assert claimed is not None
    leased, lease = claimed
    assert leased.id == job.id
    assert leased.run_id == job.run_id
    assert leased.project_id == job.project_id
    assert leased.payload == job.payload
    assert leased.requirements == job.requirements
    assert leased.priority is job.priority
    assert leased.max_attempts == job.max_attempts
    assert leased.status is JobStatus.LEASED
    assert leased.attempt == 1
    assert lease.holder == consumer
    assert lease.expires_at == T0 + LEASE
    assert leased.lease == lease


async def test_priority_wins_and_age_breaks_the_tie(queue: JobQueue, consumer: WorkerId) -> None:
    older_normal = make_job(priority=Priority.NORMAL, created_at=later(1))
    newer_normal = make_job(priority=Priority.NORMAL, created_at=later(2))
    low = make_job(priority=Priority.LOW, created_at=later(0))
    critical = make_job(priority=Priority.CRITICAL, created_at=later(3))
    high = make_job(priority=Priority.HIGH, created_at=later(4))
    for job in (low, newer_normal, critical, older_normal, high):
        await queue.enqueue(job)

    order = []
    while (claimed := await claim(queue, consumer, at=10)) is not None:
        order.append(claimed[0].id)

    assert order == [critical.id, high.id, older_normal.id, newer_normal.id, low.id]


async def test_claiming_only_takes_the_requested_types(queue: JobQueue, consumer: WorkerId) -> None:
    review = make_job(job_type=JobType.REVIEW)
    code = make_job(job_type=JobType.CODE)
    await queue.enqueue(code)
    await queue.enqueue(review)

    claimed = await claim(queue, consumer, types=[JobType.REVIEW])

    assert claimed is not None
    assert claimed[0].id == review.id
    assert await queue.depth(JobType.CODE) == 1


async def test_concurrent_consumers_never_share_a_job(queue: JobQueue) -> None:
    """The property the whole design exists for: one job, one consumer.

    Twice as many claims as jobs are issued at once, so every job is contested
    and the losers must come back empty rather than with a duplicate.
    """
    jobs = [make_job(created_at=later(index)) for index in range(12)]
    for job in jobs:
        await queue.enqueue(job)
    consumers = [WorkerId.generate() for _ in range(4)]

    results = await asyncio.gather(
        *(claim(queue, consumer, at=20) for consumer in consumers for _ in range(6))
    )

    handed_out = [claimed[0].id for claimed in results if claimed is not None]
    assert len(handed_out) == len(jobs)
    assert len(set(handed_out)) == len(jobs)


# -- idempotency --------------------------------------------------------
async def test_enqueuing_twice_does_not_duplicate_work(queue: JobQueue, consumer: WorkerId) -> None:
    job = make_job()
    await queue.enqueue(job)
    await queue.enqueue(job)

    assert await queue.depth() == 1
    assert await claim(queue, consumer) is not None
    assert await claim(queue, consumer) is None


async def test_a_redelivered_enqueue_loses_to_a_live_lease(
    queue: JobQueue, consumer: WorkerId
) -> None:
    job = make_job()
    await queue.enqueue(job)
    assert await claim(queue, consumer) is not None

    await queue.enqueue(job)

    assert await queue.depth() == 0
    assert await claim(queue, WorkerId.generate()) is None


# -- leases -------------------------------------------------------------
async def test_renewing_pushes_the_deadline_back(queue: JobQueue, consumer: WorkerId) -> None:
    job = make_job()
    await queue.enqueue(job)
    claimed = await claim(queue, consumer)
    assert claimed is not None

    renewed = await queue.renew(
        job_id=job.id, token=claimed[1].token, duration=LEASE, now=later(10)
    )

    assert renewed.expires_at == later(10) + LEASE
    assert renewed.token == claimed[1].token
    assert renewed.holder == consumer
    assert await queue.reclaim_expired(now=later(31)) == []


async def test_renewing_with_a_foreign_token_raises(queue: JobQueue, consumer: WorkerId) -> None:
    job = make_job()
    await queue.enqueue(job)
    assert await claim(queue, consumer) is not None

    with pytest.raises(JobLeaseExpiredError):
        await queue.renew(
            job_id=job.id, token=LeaseToken("not-the-token"), duration=LEASE, now=later(1)
        )


async def test_renewing_a_lapsed_lease_raises(queue: JobQueue, consumer: WorkerId) -> None:
    """A holder that overslept must not be able to resume: its job may already
    be running somewhere else."""
    job = make_job()
    await queue.enqueue(job)
    claimed = await claim(queue, consumer)
    assert claimed is not None

    with pytest.raises(JobLeaseExpiredError):
        await queue.renew(job_id=job.id, token=claimed[1].token, duration=LEASE, now=later(31))


async def test_a_lapsed_lease_makes_the_job_claimable_again(
    queue: JobQueue, consumer: WorkerId
) -> None:
    job = make_job()
    await queue.enqueue(job)
    first = await claim(queue, consumer)
    assert first is not None

    reclaimed = await queue.reclaim_expired(now=later(31))

    assert list(reclaimed) == [job.id]
    second = await claim(queue, WorkerId.generate(), at=32)
    assert second is not None
    assert second[0].id == job.id
    assert second[0].attempt == 2
    assert second[1].token != first[1].token


async def test_a_lapsed_lease_without_attempts_left_is_not_reoffered(
    queue: JobQueue, consumer: WorkerId
) -> None:
    job = make_job(max_attempts=1)
    await queue.enqueue(job)
    assert await claim(queue, consumer) is not None

    assert list(await queue.reclaim_expired(now=later(31))) == [job.id]

    assert await queue.depth() == 0
    assert await claim(queue, consumer, at=32) is None


async def test_reclaiming_ignores_live_leases(queue: JobQueue, consumer: WorkerId) -> None:
    job = make_job()
    await queue.enqueue(job)
    assert await claim(queue, consumer) is not None

    assert list(await queue.reclaim_expired(now=later(29))) == []


# -- completion ---------------------------------------------------------
async def test_acknowledging_removes_the_job_for_good(queue: JobQueue, consumer: WorkerId) -> None:
    job = make_job()
    await queue.enqueue(job)
    claimed = await claim(queue, consumer)
    assert claimed is not None

    await queue.acknowledge(job_id=job.id, token=claimed[1].token)
    # Replaying the acknowledgement, then the publication that preceded it:
    # at-least-once delivery makes both a matter of when, not if.
    await queue.acknowledge(job_id=job.id, token=claimed[1].token)
    await queue.enqueue(job)

    assert await queue.depth() == 0
    assert await claim(queue, consumer, at=1) is None


async def test_a_late_report_from_a_previous_holder_is_ignored(
    queue: JobQueue, consumer: WorkerId
) -> None:
    job = make_job()
    await queue.enqueue(job)
    first = await claim(queue, consumer)
    assert first is not None
    await queue.reclaim_expired(now=later(31))
    second = await claim(queue, WorkerId.generate(), at=32)
    assert second is not None

    await queue.acknowledge(job_id=job.id, token=first[1].token)

    # The job still belongs to its new holder, which can still report on it.
    renewed = await queue.renew(job_id=job.id, token=second[1].token, duration=LEASE, now=later(33))
    assert renewed.expires_at == later(33) + LEASE


async def test_releasing_with_requeue_offers_the_job_again(
    queue: JobQueue, consumer: WorkerId
) -> None:
    job = make_job()
    await queue.enqueue(job)
    claimed = await claim(queue, consumer)
    assert claimed is not None

    await queue.release(job_id=job.id, token=claimed[1].token, requeue=True)

    assert await queue.depth() == 1
    again = await claim(queue, WorkerId.generate(), at=1)
    assert again is not None
    assert again[0].attempt == 2


async def test_releasing_without_requeue_takes_the_job_out(
    queue: JobQueue, consumer: WorkerId
) -> None:
    job = make_job()
    await queue.enqueue(job)
    claimed = await claim(queue, consumer)
    assert claimed is not None

    await queue.release(job_id=job.id, token=claimed[1].token, requeue=False)

    assert await queue.depth() == 0
    assert await claim(queue, consumer, at=1) is None
    assert list(await queue.reclaim_expired(now=later(60))) == []


# -- cancellation -------------------------------------------------------
async def test_cancelling_a_run_leaves_other_runs_alone(
    queue: JobQueue, consumer: WorkerId
) -> None:
    doomed = RunId.generate()
    spared = RunId.generate()
    first = make_job(run_id=doomed, created_at=later(0))
    second = make_job(run_id=doomed, created_at=later(1))
    survivor = make_job(run_id=spared, created_at=later(2))
    for job in (first, second, survivor):
        await queue.enqueue(job)

    cancelled = await queue.cancel_run_jobs(doomed)

    assert set(cancelled) == {first.id, second.id}
    assert await queue.depth() == 1
    claimed = await claim(queue, consumer, at=3)
    assert claimed is not None
    assert claimed[0].id == survivor.id


async def test_cancelling_a_run_invalidates_an_in_flight_lease(
    queue: JobQueue, consumer: WorkerId
) -> None:
    run_id = RunId.generate()
    job = make_job(run_id=run_id)
    await queue.enqueue(job)
    claimed = await claim(queue, consumer)
    assert claimed is not None

    cancelled = await queue.cancel_run_jobs(run_id)

    assert list(cancelled) == [job.id]
    with pytest.raises(JobLeaseExpiredError):
        await queue.renew(job_id=job.id, token=claimed[1].token, duration=LEASE, now=later(1))


async def test_cancelling_twice_reports_nothing_the_second_time(queue: JobQueue) -> None:
    run_id = RunId.generate()
    await queue.enqueue(make_job(run_id=run_id))

    assert len(await queue.cancel_run_jobs(run_id)) == 1
    assert list(await queue.cancel_run_jobs(run_id)) == []


# -- metrics ------------------------------------------------------------
async def test_depth_counts_claimable_work_only(queue: JobQueue, consumer: WorkerId) -> None:
    await queue.enqueue(make_job(job_type=JobType.CODE, created_at=later(0)))
    await queue.enqueue(make_job(job_type=JobType.CODE, created_at=later(1)))
    await queue.enqueue(make_job(job_type=JobType.REVIEW, created_at=later(2)))

    assert await queue.depth() == 3
    assert await queue.depth(JobType.CODE) == 2
    assert await queue.depth(JobType.TEST) == 0

    assert await claim(queue, consumer, at=3) is not None
    assert await queue.depth(JobType.CODE) == 1
