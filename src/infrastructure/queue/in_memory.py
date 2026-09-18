"""In-memory twins of the three coordination ports.

They exist so the platform can be run and tested without Docker, which is only
useful if they behave *exactly* like the Redis adapters: same lease semantics,
same priority order down to the tie-break, same replay tolerance. The shared
contract suite in ``tests/infrastructure`` runs against both for that reason,
and the queue borrows ``queue_score`` from the Redis codecs rather than
inventing its own ordering — two formulas would eventually disagree, and the
disagreement would only show up in production.

They are asyncio-safe, not merely dictionaries: every operation that awaits
runs under an ``asyncio.Lock``, so two coroutines racing for the same job get
the same outcome they would get from a Lua script. What they deliberately do
not reproduce is Redis' garbage collection — nothing here expires on a timer,
because a process that will be restarted has nothing to collect.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from typing import Any

from domain.entities.job import Job
from domain.entities.worker import Worker
from domain.enums import FailureKind, JobStatus, JobType
from domain.events.base import DomainEvent
from domain.exceptions import JobLeaseExpiredError
from domain.value_objects.identifiers import JobId, RunId, WorkerId
from domain.value_objects.lease import Lease, LeaseToken
from domain.value_objects.worker import WorkerLoad
from infrastructure.redis.codecs import is_terminal_for_run, queue_score, run_id_of

__all__ = [
    "DEFAULT_SUBSCRIBER_BACKLOG",
    "InMemoryEventBus",
    "InMemoryJobQueue",
    "InMemoryRunEventStream",
    "InMemoryWorkerRegistry",
]

_log = logging.getLogger(__name__)

DEFAULT_SUBSCRIBER_BACKLOG = 10_000
"""Events a slow subscriber may fall behind by before the oldest are dropped.

The same trade-off ``MAXLEN`` makes on the Redis side: a subscriber that cannot
keep up must not be allowed to block the publisher, and the durable record of
what happened is PostgreSQL's, not this queue's.
"""


def _record(job: Job) -> dict[str, Any]:
    """Every constructor argument of a job, read through its public surface."""
    return {
        "job_id": job.id,
        "run_id": job.run_id,
        "project_id": job.project_id,
        "job_type": job.type,
        "created_at": job.created_at,
        "role": job.role,
        "candidate_id": job.candidate_id,
        "priority": job.priority,
        "status": job.status,
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "payload": job.payload,
        "requirements": job.requirements,
        "idempotency_key": job.idempotency_key,
        "lease": job.lease,
        "assigned_worker_id": job.assigned_worker_id,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": job.result,
        "failure_kind": job.failure_kind,
        "failure_reason": job.failure_reason,
    }


def _rebuild(job: Job, **overrides: Any) -> Job:
    """A copy of a job with some fields replaced.

    This is the in-memory stand-in for a Redis round trip: the queue never
    stores the caller's object, so a caller that keeps mutating its own entity
    cannot corrupt queue state — and the copy starts with an empty event
    buffer, exactly like a record read back out of Redis.
    """
    return Job(**{**_record(job), **overrides})


class InMemoryJobQueue:
    """Job queue with leases, in one process."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jobs: dict[JobId, Job] = {}
        self._ready: dict[JobType, dict[JobId, int]] = {}
        self._scores: dict[JobId, int] = {}
        self._leases: dict[JobId, datetime] = {}
        self._runs: dict[RunId, set[JobId]] = {}
        self._acknowledged: set[JobId] = set()

    # -- publication ----------------------------------------------------
    async def enqueue(self, job: Job) -> None:
        """Publish a job. Enqueuing the same job twice must not duplicate work."""
        async with self._lock:
            stored = self._jobs.get(job.id)
            if stored is not None and (stored.status.is_in_flight or job.id in self._acknowledged):
                # Someone is holding it, or it is already finished: a
                # redelivered publication must lose to both.
                return
            score = queue_score(job.priority, job.created_at)
            self._jobs[job.id] = _rebuild(job, status=JobStatus.QUEUED)
            self._scores[job.id] = score
            self._ready.setdefault(job.type, {})[job.id] = score
            self._runs.setdefault(job.run_id, set()).add(job.id)

    # -- consumption ----------------------------------------------------
    async def claim(
        self,
        *,
        consumer: WorkerId,
        job_types: Sequence[JobType],
        lease_duration: timedelta,
        now: datetime,
    ) -> tuple[Job, Lease] | None:
        """Atomically take the highest-priority eligible job, or ``None``.

        Like the Redis adapter, the transition runs through ``Job.lease_to``,
        so the returned job carries a ``JobLeased`` event while the copy kept
        here does not.
        """
        wanted = list(dict.fromkeys(job_types))
        if not wanted:
            return None
        async with self._lock:
            while True:
                candidate = self._next_ready(wanted)
                if candidate is None:
                    return None
                job_type, job_id = candidate
                del self._ready[job_type][job_id]
                stored = self._jobs.get(job_id)
                if stored is None or stored.status is not JobStatus.QUEUED:
                    # Cancelled or already taken: the entry was stale, look on.
                    continue
                claimed = _rebuild(stored)
                lease = claimed.lease_to(
                    worker_id=consumer,
                    now=now,
                    duration=lease_duration,
                    token=LeaseToken.generate(),
                )
                self._jobs[job_id] = _rebuild(claimed)
                self._leases[job_id] = lease.expires_at
                return claimed, lease

    def _next_ready(self, wanted: Sequence[JobType]) -> tuple[JobType, JobId] | None:
        """Lowest score across the requested types, ties broken by id.

        The tie-break matters: a sorted set orders equal scores
        lexicographically, and picking differently here would make the two
        adapters disagree about which of two jobs created in the same
        millisecond runs first.
        """
        best: tuple[tuple[int, str], JobType, JobId] | None = None
        for job_type in wanted:
            for job_id, score in self._ready.get(job_type, {}).items():
                key = (score, str(job_id))
                if best is None or key < best[0]:
                    best = (key, job_type, job_id)
        return None if best is None else (best[1], best[2])

    async def renew(
        self, *, job_id: JobId, token: LeaseToken, duration: timedelta, now: datetime
    ) -> Lease:
        """Extend a held lease. Raises ``JobLeaseExpiredError`` if it was lost."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobLeaseExpiredError(job_id)
            # The entity owns the rule: wrong token or lapsed deadline both mean
            # the work has moved on and this caller must stop.
            renewed = job.renew_lease(token=token, now=now, duration=duration)
            self._jobs[job_id] = _rebuild(job)
            self._leases[job_id] = renewed.expires_at
            return renewed

    async def acknowledge(self, *, job_id: JobId, token: LeaseToken) -> None:
        """Remove a finished job from the queue."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not self._holds(job, token):
                return
            self._leases.pop(job_id, None)
            self._runs.get(job.run_id, set()).discard(job_id)
            self._acknowledged.add(job_id)
            self._jobs[job_id] = _rebuild(job, lease=None, assigned_worker_id=None)

    async def release(self, *, job_id: JobId, token: LeaseToken, requeue: bool) -> None:
        """Give a job back, optionally making it immediately claimable again."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not self._holds(job, token):
                return
            self._leases.pop(job_id, None)
            status = JobStatus.QUEUED if requeue else JobStatus.PENDING
            self._jobs[job_id] = _rebuild(job, status=status, lease=None, assigned_worker_id=None)
            if requeue:
                self._ready.setdefault(job.type, {})[job_id] = self._scores[job_id]

    # -- recovery -------------------------------------------------------
    async def reclaim_expired(self, *, now: datetime, limit: int = 100) -> Sequence[JobId]:
        """Return jobs whose lease lapsed so the orchestrator can retry them."""
        async with self._lock:
            lapsed = sorted(
                (job_id for job_id, expiry in self._leases.items() if expiry <= now),
                key=lambda job_id: (self._leases[job_id], str(job_id)),
            )[:limit]
            reclaimed: list[JobId] = []
            for job_id in lapsed:
                del self._leases[job_id]
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.expire_lease(now):
                    job.requeue(now=now, reason="lease expired")
                    self._ready.setdefault(job.type, {})[job_id] = self._scores[job_id]
                self._jobs[job_id] = _rebuild(job)
                reclaimed.append(job_id)
            return reclaimed

    async def cancel_run_jobs(self, run_id: RunId) -> Sequence[JobId]:
        """Drop every pending job of a run; in-flight ones are cancelled by lease."""
        async with self._lock:
            cancelled: list[JobId] = []
            for job_id in sorted(self._runs.get(run_id, set()), key=str):
                job = self._jobs.get(job_id)
                if job is None or job.status.is_terminal:
                    continue
                self._ready.get(job.type, {}).pop(job_id, None)
                self._leases.pop(job_id, None)
                self._jobs[job_id] = _rebuild(
                    job,
                    status=JobStatus.CANCELLED,
                    failure_kind=FailureKind.CANCELLED,
                    lease=None,
                    assigned_worker_id=None,
                )
                cancelled.append(job_id)
            return cancelled

    # -- metrics --------------------------------------------------------
    async def depth(self, job_type: JobType | None = None) -> int:
        """Queue depth, exported as a metric and usable by future autoscaling."""
        async with self._lock:
            if job_type is not None:
                return len(self._ready.get(job_type, {}))
            return sum(len(ready) for ready in self._ready.values())

    @staticmethod
    def _holds(job: Job, token: LeaseToken) -> bool:
        """Whether the report comes from the worker that currently owns the job.

        A mismatch is ignored rather than raised: it is a late report from a
        previous holder, and the work has already been handed to someone else.
        """
        return job.lease is not None and job.lease.token == token


class InMemoryWorkerRegistry:
    """Worker registry for a single process.

    The heartbeat TTL is accepted and not used: on the Redis side it only
    governs when a forgotten record is collected, while *staleness* — the thing
    the orchestrator acts on — is decided from the stored heartbeat timestamp
    against an injected ``now``. Here there is nothing to collect, so the
    observable contract is identical.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._workers: dict[WorkerId, Worker] = {}

    async def register(self, worker: Worker, *, ttl: timedelta) -> None:
        """Add or refresh a worker registration with a heartbeat TTL."""
        async with self._lock:
            self._workers[worker.id] = _copy_worker(worker)

    async def heartbeat(
        self, worker_id: WorkerId, *, load: WorkerLoad, at: datetime, ttl: timedelta
    ) -> bool:
        """Refresh liveness. Returns False if the worker is no longer registered."""
        async with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False
            worker.heartbeat(now=at, load=load)
            worker.pull_events()
            return True

    async def get(self, worker_id: WorkerId) -> Worker | None:
        async with self._lock:
            worker = self._workers.get(worker_id)
            return _copy_worker(worker) if worker is not None else None

    async def list_all(self) -> Sequence[Worker]:
        """Every known worker, including draining and unavailable ones."""
        async with self._lock:
            return [_copy_worker(worker) for worker in self._sorted()]

    async def list_available(self) -> Sequence[Worker]:
        """Workers that may receive new jobs right now."""
        async with self._lock:
            return [
                _copy_worker(worker)
                for worker in self._sorted()
                if worker.status.accepts_new_jobs and worker.available_slots > 0
            ]

    async def update(self, worker: Worker) -> None:
        """Persist a state change made on the entity (drain, unhealthy, load)."""
        async with self._lock:
            if worker.id not in self._workers:
                # Never resurrects a deregistered worker, matching the Redis
                # adapter's update script.
                return
            self._workers[worker.id] = _copy_worker(worker)

    async def deregister(self, worker_id: WorkerId) -> None:
        async with self._lock:
            self._workers.pop(worker_id, None)

    async def reap_stale(self, *, now: datetime, heartbeat_timeout: timedelta) -> Sequence[Worker]:
        """Mark silent workers unavailable and return them, events included."""
        async with self._lock:
            reaped: list[Worker] = []
            for stored in self._sorted():
                if not stored.status.is_live or not stored.is_stale(now, heartbeat_timeout):
                    continue
                # The transition runs on the copy that leaves, so the events
                # reach the caller while the registry's own record stays as
                # event-free as everything else it holds.
                reaped_worker = _copy_worker(stored)
                reaped_worker.mark_unavailable(now=now, reason="heartbeat timeout")
                self._workers[reaped_worker.id] = _copy_worker(reaped_worker)
                reaped.append(reaped_worker)
            return reaped

    def _sorted(self) -> list[Worker]:
        """Stable order, so two callers see the pool the same way."""
        return [self._workers[worker_id] for worker_id in sorted(self._workers, key=str)]


def _copy_worker(worker: Worker) -> Worker:
    """Detached copy: the registry hands out state, never its own objects."""
    copy = Worker(
        worker_id=worker.id,
        endpoint=worker.endpoint,
        capabilities=worker.capabilities,
        registered_at=worker.registered_at,
        status=worker.status,
        load=worker.load,
        last_heartbeat_at=worker.last_heartbeat_at,
        metadata=worker.metadata,
    )
    return copy


class InMemoryEventBus:
    """Event bus for a single process.

    No lock: registering a subscriber and fanning an event out never await
    halfway through, so an asyncio task cannot observe either operation
    half-done. The per-subscriber ``asyncio.Queue`` is what makes the hand-off
    between publisher and consumer safe.
    """

    def __init__(self, *, backlog: int = DEFAULT_SUBSCRIBER_BACKLOG) -> None:
        self._backlog = backlog
        self._subscribers: dict[RunId, list[InMemoryRunEventStream]] = {}

    async def publish(self, events: Sequence[DomainEvent]) -> None:
        """Publish a batch. Called after the transaction that produced them."""
        for event in events:
            run_id = run_id_of(event)
            if run_id is None:
                continue
            for stream in list(self._subscribers.get(run_id, ())):
                stream.offer(event)

    def subscribe(self, run_id: RunId) -> AsyncIterator[DomainEvent]:
        """Stream events for one run, starting from now.

        Registration happens here rather than on first iteration, which closes
        the race a caller cannot otherwise close: publish immediately after
        ``subscribe`` and the event is already addressed to this subscriber.
        """
        stream = InMemoryRunEventStream(self, run_id, backlog=self._backlog)
        self._subscribers.setdefault(run_id, []).append(stream)
        return stream

    def _unsubscribe(self, stream: InMemoryRunEventStream) -> None:
        streams = self._subscribers.get(stream.run_id)
        if streams is None:
            return
        if stream in streams:
            streams.remove(stream)
        if not streams:
            del self._subscribers[stream.run_id]


class InMemoryRunEventStream:
    """One subscriber's view of a run."""

    def __init__(self, bus: InMemoryEventBus, run_id: RunId, *, backlog: int) -> None:
        self.run_id = run_id
        self._bus = bus
        self._queue: asyncio.Queue[DomainEvent | None] = asyncio.Queue(maxsize=backlog)
        self._closed = False

    async def ready(self) -> None:
        """Already positioned; present so callers can treat both buses alike."""

    def offer(self, event: DomainEvent) -> None:
        if self._closed:
            return
        self._put(event)
        if is_terminal_for_run(event):
            # Deliver it, then end the stream: nothing else will ever come.
            self._closed = True
            self._put(None)
            self._bus._unsubscribe(self)

    def _put(self, item: DomainEvent | None) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop the oldest, like a trimmed stream: a stalled subscriber must
            # never become back-pressure on the publisher.
            _log.warning("subscriber backlog full for run %s, dropping oldest event", self.run_id)
            self._queue.get_nowait()
            self._queue.put_nowait(item)

    def __aiter__(self) -> AsyncIterator[DomainEvent]:
        return self

    async def __anext__(self) -> DomainEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        """Stop the subscription. The caller's disconnect handler calls this."""
        self._closed = True
        self._bus._unsubscribe(self)
        self._put(None)
