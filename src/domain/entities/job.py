"""Schedulable unit of work (spec sections 12, 35 and 36).

Both inference and deterministic work are jobs. A job is claimed under a
time-bounded lease, and every externally triggered mutation is idempotent,
because delivery is assumed to be at-least-once.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from domain.entities.base import Entity
from domain.enums import AgentRole, FailureKind, JobStatus, JobType, Priority
from domain.events.job import (
    JobCompleted,
    JobEnqueued,
    JobFailed,
    JobLeased,
    JobLeaseExpired,
    JobRequeued,
)
from domain.exceptions import (
    InvalidStateTransitionError,
    JobLeaseExpiredError,
)
from domain.value_objects.identifiers import (
    CandidateId,
    IdempotencyKey,
    JobId,
    ProjectId,
    RunId,
    WorkerId,
)
from domain.value_objects.lease import Lease, LeaseToken
from domain.value_objects.worker import JobRequirements

__all__ = ["Job"]

DEFAULT_MAX_ATTEMPTS = 3


class Job(Entity):
    """One attemptable unit of work with an explicit retry budget."""

    __slots__ = (
        "_assigned_worker_id",
        "_attempt",
        "_candidate_id",
        "_completed_at",
        "_created_at",
        "_failure_kind",
        "_failure_reason",
        "_id",
        "_idempotency_key",
        "_lease",
        "_max_attempts",
        "_payload",
        "_priority",
        "_project_id",
        "_requirements",
        "_result",
        "_role",
        "_run_id",
        "_started_at",
        "_status",
        "_type",
    )

    def __init__(
        self,
        *,
        job_id: JobId,
        run_id: RunId,
        project_id: ProjectId,
        job_type: JobType,
        created_at: datetime,
        role: AgentRole | None = None,
        candidate_id: CandidateId | None = None,
        priority: Priority = Priority.NORMAL,
        status: JobStatus = JobStatus.PENDING,
        attempt: int = 0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        payload: Mapping[str, Any] | None = None,
        requirements: JobRequirements | None = None,
        idempotency_key: IdempotencyKey | None = None,
        lease: Lease | None = None,
        assigned_worker_id: WorkerId | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        result: Mapping[str, Any] | None = None,
        failure_kind: FailureKind | None = None,
        failure_reason: str | None = None,
    ) -> None:
        super().__init__()
        if job_type.requires_inference and role is None:
            raise ValueError(f"{job_type} is an inference job and requires an agent role")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._id = job_id
        self._run_id = run_id
        self._project_id = project_id
        self._type = job_type
        self._role = role
        self._candidate_id = candidate_id
        self._priority = priority
        self._status = status
        self._attempt = attempt
        self._max_attempts = max_attempts
        self._payload: dict[str, Any] = dict(payload or {})
        self._requirements = requirements
        self._idempotency_key = idempotency_key
        self._lease = lease
        self._assigned_worker_id = assigned_worker_id
        self._created_at = created_at
        self._started_at = started_at
        self._completed_at = completed_at
        self._result: dict[str, Any] | None = dict(result) if result is not None else None
        self._failure_kind = failure_kind
        self._failure_reason = failure_reason

    # -- construction ---------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        job_id: JobId,
        run_id: RunId,
        project_id: ProjectId,
        job_type: JobType,
        now: datetime,
        role: AgentRole | None = None,
        candidate_id: CandidateId | None = None,
        priority: Priority = Priority.NORMAL,
        payload: Mapping[str, Any] | None = None,
        requirements: JobRequirements | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        idempotency_key: IdempotencyKey | None = None,
    ) -> Job:
        return cls(
            job_id=job_id,
            run_id=run_id,
            project_id=project_id,
            job_type=job_type,
            created_at=now,
            role=role,
            candidate_id=candidate_id,
            priority=priority,
            payload=payload,
            requirements=requirements,
            max_attempts=max_attempts,
            idempotency_key=idempotency_key,
        )

    # -- accessors ------------------------------------------------------
    @property
    def identity(self) -> JobId:
        return self._id

    @property
    def id(self) -> JobId:
        return self._id

    @property
    def run_id(self) -> RunId:
        return self._run_id

    @property
    def project_id(self) -> ProjectId:
        return self._project_id

    @property
    def type(self) -> JobType:
        return self._type

    @property
    def role(self) -> AgentRole | None:
        return self._role

    @property
    def candidate_id(self) -> CandidateId | None:
        return self._candidate_id

    @property
    def priority(self) -> Priority:
        return self._priority

    @property
    def status(self) -> JobStatus:
        return self._status

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def payload(self) -> Mapping[str, Any]:
        return dict(self._payload)

    @property
    def requirements(self) -> JobRequirements | None:
        return self._requirements

    @property
    def idempotency_key(self) -> IdempotencyKey | None:
        return self._idempotency_key

    @property
    def lease(self) -> Lease | None:
        return self._lease

    @property
    def assigned_worker_id(self) -> WorkerId | None:
        return self._assigned_worker_id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def completed_at(self) -> datetime | None:
        return self._completed_at

    @property
    def result(self) -> Mapping[str, Any] | None:
        return dict(self._result) if self._result is not None else None

    @property
    def failure_kind(self) -> FailureKind | None:
        return self._failure_kind

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def attempts_remaining(self) -> int:
        return max(self._max_attempts - self._attempt, 0)

    @property
    def duration_ms(self) -> int:
        if self._started_at is None or self._completed_at is None:
            return 0
        return int((self._completed_at - self._started_at).total_seconds() * 1000)

    # -- lifecycle ------------------------------------------------------
    def enqueue(self, now: datetime) -> None:
        """Publish to the queue. Idempotent: re-enqueuing a queued job is a no-op."""
        if self._status is JobStatus.QUEUED:
            return
        if self._status not in (JobStatus.PENDING, JobStatus.FAILED):
            raise InvalidStateTransitionError("Job", self._status, JobStatus.QUEUED)
        self._status = JobStatus.QUEUED
        self._lease = None
        self._assigned_worker_id = None
        self.record(
            JobEnqueued(
                occurred_at=now,
                job_id=self._id,
                run_id=self._run_id,
                job_type=self._type,
                attempt=self._attempt,
            )
        )

    def lease_to(
        self,
        *,
        worker_id: WorkerId,
        now: datetime,
        duration: timedelta,
        token: LeaseToken | None = None,
    ) -> Lease:
        """Claim the job for a worker and start an attempt."""
        if self._status is not JobStatus.QUEUED:
            raise InvalidStateTransitionError("Job", self._status, JobStatus.LEASED)
        self._attempt += 1
        self._status = JobStatus.LEASED
        self._assigned_worker_id = worker_id
        self._started_at = self._started_at or now
        self._lease = Lease.granted(
            job_id=self._id, holder=worker_id, now=now, duration=duration, token=token
        )
        self.record(
            JobLeased(
                occurred_at=now,
                job_id=self._id,
                run_id=self._run_id,
                worker_id=worker_id,
                attempt=self._attempt,
            )
        )
        return self._lease

    def mark_running(self, *, token: LeaseToken, now: datetime) -> None:
        self._assert_lease(token, now)
        if self._status is JobStatus.RUNNING:
            return
        self._status = JobStatus.RUNNING
        self._started_at = self._started_at or now

    def renew_lease(self, *, token: LeaseToken, now: datetime, duration: timedelta) -> Lease:
        lease = self._assert_lease(token, now)
        self._lease = lease.renewed(now=now, duration=duration)
        return self._lease

    def complete(
        self,
        *,
        token: LeaseToken | None,
        now: datetime,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        """Record success. Replaying a completion is a no-op, not an error."""
        if self._status is JobStatus.SUCCEEDED:
            return
        if token is not None:
            self._assert_lease(token, now)
        self._status = JobStatus.SUCCEEDED
        self._completed_at = now
        self._result = dict(result) if result is not None else None
        self._lease = None
        self.record(
            JobCompleted(
                occurred_at=now,
                job_id=self._id,
                run_id=self._run_id,
                job_type=self._type,
                duration_ms=self.duration_ms,
            )
        )

    def fail(
        self,
        *,
        token: LeaseToken | None,
        now: datetime,
        kind: FailureKind,
        reason: str,
        retryable: bool | None = None,
    ) -> bool:
        """Record a failure and return whether another attempt is allowed.

        ``retryable`` defaults to the domain rule: infrastructure and inference
        problems deserve another worker, produced-code defects do not belong to
        the job's retry budget — they go back into the agentic repair loop.
        """
        if self._status.is_terminal:
            return False
        if token is not None:
            self._assert_lease(token, now)
        if retryable is None:
            retryable = kind.is_infrastructure or kind is FailureKind.INVALID_STRUCTURED_OUTPUT
        may_retry = retryable and self.attempts_remaining > 0
        self._status = JobStatus.FAILED if may_retry else JobStatus.DEAD
        self._failure_kind = kind
        self._failure_reason = reason
        self._completed_at = now
        self._lease = None
        self._assigned_worker_id = None
        self.record(
            JobFailed(
                occurred_at=now,
                job_id=self._id,
                run_id=self._run_id,
                failure_kind=kind,
                reason=reason,
                attempt=self._attempt,
                retryable=may_retry,
            )
        )
        return may_retry

    def expire_lease(self, now: datetime) -> bool:
        """Reclaim a job whose holder went silent. Returns True if it may retry.

        This is what prevents permanently stuck jobs when a worker disappears
        mid-flight.
        """
        lease = self._lease
        if lease is None or not lease.is_expired(now):
            return False
        holder = lease.holder
        self._lease = None
        self._assigned_worker_id = None
        self.record(
            JobLeaseExpired(occurred_at=now, job_id=self._id, run_id=self._run_id, holder=holder)
        )
        if self.attempts_remaining > 0:
            self._status = JobStatus.FAILED
            return True
        self._status = JobStatus.DEAD
        self._failure_kind = FailureKind.INFRASTRUCTURE
        self._failure_reason = "lease expired and retry budget exhausted"
        self._completed_at = now
        return False

    def requeue(self, *, now: datetime, reason: str) -> None:
        """Return a failed-but-retryable job to the queue."""
        if self._status is not JobStatus.FAILED:
            raise InvalidStateTransitionError("Job", self._status, JobStatus.QUEUED)
        if self.attempts_remaining <= 0:
            raise InvalidStateTransitionError("Job", self._status, JobStatus.QUEUED)
        self._status = JobStatus.QUEUED
        self._completed_at = None
        self._lease = None
        self._assigned_worker_id = None
        self.record(
            JobRequeued(
                occurred_at=now,
                job_id=self._id,
                run_id=self._run_id,
                attempt=self._attempt,
                reason=reason,
            )
        )

    def cancel(self, now: datetime) -> None:
        """Cancellation is idempotent and never fails on a terminal job."""
        if self._status.is_terminal:
            return
        self._status = JobStatus.CANCELLED
        self._completed_at = now
        self._lease = None
        self._assigned_worker_id = None
        self._failure_kind = FailureKind.CANCELLED

    # -- internals ------------------------------------------------------
    def _assert_lease(self, token: LeaseToken, now: datetime) -> Lease:
        lease = self._lease
        if lease is None or lease.token != token:
            raise JobLeaseExpiredError(self._id, holder=self._assigned_worker_id)
        if lease.is_expired(now):
            raise JobLeaseExpiredError(self._id, holder=lease.holder)
        return lease

    def __repr__(self) -> str:
        return (
            f"Job(id={self._id}, type={self._type}, status={self._status}, "
            f"attempt={self._attempt}/{self._max_attempts})"
        )
