"""Distributed job queue with leases (spec sections 12 and 36).

At-least-once delivery is assumed; consumers are idempotent. Claiming returns a
lease that must be renewed, which is how a vanished consumer releases its work
automatically.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from domain.entities.job import Job
from domain.enums import JobType
from domain.value_objects.identifiers import JobId, RunId, WorkerId
from domain.value_objects.lease import Lease, LeaseToken

__all__ = ["JobQueue"]


@runtime_checkable
class JobQueue(Protocol):
    """Transport for schedulable work. Durable state lives in PostgreSQL."""

    async def enqueue(self, job: Job) -> None:
        """Publish a job. Enqueuing the same job twice must not duplicate work."""
        ...

    async def claim(
        self,
        *,
        consumer: WorkerId,
        job_types: Sequence[JobType],
        lease_duration: timedelta,
        now: datetime,
    ) -> tuple[Job, Lease] | None:
        """Atomically take the highest-priority eligible job, or ``None``."""
        ...

    async def renew(
        self, *, job_id: JobId, token: LeaseToken, duration: timedelta, now: datetime
    ) -> Lease:
        """Extend a held lease. Raises ``JobLeaseExpiredError`` if it was lost."""
        ...

    async def acknowledge(self, *, job_id: JobId, token: LeaseToken) -> None:
        """Remove a finished job from the queue."""
        ...

    async def release(self, *, job_id: JobId, token: LeaseToken, requeue: bool) -> None:
        """Give a job back, optionally making it immediately claimable again."""
        ...

    async def reclaim_expired(self, *, now: datetime, limit: int = 100) -> Sequence[JobId]:
        """Return jobs whose lease lapsed so the orchestrator can retry them."""
        ...

    async def cancel_run_jobs(self, run_id: RunId) -> Sequence[JobId]:
        """Drop every pending job of a run; in-flight ones are cancelled by lease."""
        ...

    async def depth(self, job_type: JobType | None = None) -> int:
        """Queue depth, exported as a metric and usable by future autoscaling."""
        ...
