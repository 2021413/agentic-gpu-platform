"""Events describing distributed job execution (spec sections 12 and 36)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from domain.enums import FailureKind, JobType
from domain.events.base import DomainEvent, EventName
from domain.value_objects.identifiers import JobId, RunId, WorkerId

__all__ = [
    "JobCompleted",
    "JobEnqueued",
    "JobFailed",
    "JobLeaseExpired",
    "JobLeased",
    "JobRequeued",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class JobEnqueued(DomainEvent):
    name: ClassVar[EventName] = "job.enqueued"

    job_id: JobId
    run_id: RunId
    job_type: JobType
    attempt: int


@dataclass(frozen=True, slots=True, kw_only=True)
class JobLeased(DomainEvent):
    name: ClassVar[EventName] = "job.leased"

    job_id: JobId
    run_id: RunId
    worker_id: WorkerId
    attempt: int


@dataclass(frozen=True, slots=True, kw_only=True)
class JobCompleted(DomainEvent):
    name: ClassVar[EventName] = "job.completed"

    job_id: JobId
    run_id: RunId
    job_type: JobType
    duration_ms: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class JobFailed(DomainEvent):
    name: ClassVar[EventName] = "job.failed"

    job_id: JobId
    run_id: RunId
    failure_kind: FailureKind
    reason: str
    attempt: int
    retryable: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class JobLeaseExpired(DomainEvent):
    name: ClassVar[EventName] = "job.lease_expired"

    job_id: JobId
    run_id: RunId
    holder: WorkerId


@dataclass(frozen=True, slots=True, kw_only=True)
class JobRequeued(DomainEvent):
    name: ClassVar[EventName] = "job.requeued"

    job_id: JobId
    run_id: RunId
    attempt: int
    reason: str
