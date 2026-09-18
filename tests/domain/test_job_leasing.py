"""Leasing is what prevents permanently stuck jobs (spec section 36)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from domain.entities.job import Job
from domain.enums import AgentRole, FailureKind, JobStatus, JobType
from domain.exceptions import InvalidStateTransitionError, JobLeaseExpiredError
from domain.value_objects.identifiers import JobId, ProjectId, RunId, WorkerId
from domain.value_objects.lease import LeaseToken

LEASE = timedelta(seconds=30)


def make_job(now: datetime, *, max_attempts: int = 3) -> Job:
    return Job.create(
        job_id=JobId.generate(),
        run_id=RunId.generate(),
        project_id=ProjectId.generate(),
        job_type=JobType.PLAN,
        role=AgentRole.PLANNER,
        now=now,
        max_attempts=max_attempts,
    )


def test_inference_jobs_require_a_role(now: datetime) -> None:
    with pytest.raises(ValueError, match="agent role"):
        Job.create(
            job_id=JobId.generate(),
            run_id=RunId.generate(),
            project_id=ProjectId.generate(),
            job_type=JobType.CODE,
            now=now,
        )


def test_leasing_starts_an_attempt(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    assert job.status is JobStatus.LEASED
    assert job.attempt == 1
    assert not lease.is_expired(now)


def test_enqueue_is_idempotent(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    job.enqueue(now)
    assert sum(1 for e in job.pending_events if e.name == "job.enqueued") == 1


def test_a_stale_holder_cannot_report(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    with pytest.raises(JobLeaseExpiredError):
        job.complete(token=LeaseToken("someone-else"), now=now)


def test_reporting_after_expiry_is_rejected(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    with pytest.raises(JobLeaseExpiredError):
        job.complete(token=lease.token, now=now + timedelta(seconds=31))


def test_expired_lease_makes_the_job_retryable(now: datetime) -> None:
    job = make_job(now, max_attempts=2)
    job.enqueue(now)
    job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)

    assert job.expire_lease(now + timedelta(seconds=31)) is True
    assert job.status is JobStatus.FAILED
    assert job.assigned_worker_id is None

    job.requeue(now=now, reason="worker vanished")
    job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    assert job.attempt == 2
    assert job.expire_lease(now + timedelta(seconds=31)) is False
    assert job.status is JobStatus.DEAD


def test_a_live_lease_does_not_expire(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    assert job.expire_lease(now + timedelta(seconds=5)) is False
    assert job.status is JobStatus.LEASED


def test_renewal_extends_the_lease(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    later = now + timedelta(seconds=20)
    renewed = job.renew_lease(token=lease.token, now=later, duration=LEASE)
    assert renewed.expires_at == later + LEASE


def test_completion_is_idempotent(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    job.complete(token=lease.token, now=now, result={"plan": "ok"})
    job.complete(token=None, now=now)
    assert sum(1 for e in job.pending_events if e.name == "job.completed") == 1
    assert job.result == {"plan": "ok"}


def test_code_defects_do_not_consume_the_job_retry_budget(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    retryable = job.fail(token=lease.token, now=now, kind=FailureKind.TEST, reason="2 tests failed")
    assert retryable is False
    assert job.status is JobStatus.DEAD


def test_infrastructure_failures_are_retried(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    assert (
        job.fail(
            token=lease.token,
            now=now,
            kind=FailureKind.INFRASTRUCTURE,
            reason="connection refused",
        )
        is True
    )
    assert job.status is JobStatus.FAILED


def test_requeue_requires_remaining_attempts(now: datetime) -> None:
    job = make_job(now, max_attempts=1)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    job.fail(token=lease.token, now=now, kind=FailureKind.INFRASTRUCTURE, reason="boom")
    with pytest.raises(InvalidStateTransitionError):
        job.requeue(now=now, reason="retry")


def test_cancelling_a_finished_job_is_a_no_op(now: datetime) -> None:
    job = make_job(now)
    job.enqueue(now)
    lease = job.lease_to(worker_id=WorkerId.generate(), now=now, duration=LEASE)
    job.complete(token=lease.token, now=now)
    job.cancel(now)
    assert job.status is JobStatus.SUCCEEDED
