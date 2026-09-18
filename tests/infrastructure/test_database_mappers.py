"""Mapper round-trips, without a database.

A row is only useful if the aggregate that comes back out of it is the one that
went in — same counters, same budgets, same failure. These tests exercise the
translation alone, so they run everywhere, Docker or not.

Domain entities compare by identity, so equality would pass even if every
counter were lost. Each test therefore compares an explicit state snapshot.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from domain import events as domain_events
from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan
from domain.entities.project import Project
from domain.entities.review import Review
from domain.entities.run import Run
from domain.entities.worker import Worker
from domain.enums import FailureKind, JobStatus, ReviewVerdict, RunStatus
from domain.events import RunCreated, RunStateChanged, WorkerRegistered
from domain.events.base import DomainEvent
from domain.value_objects.identifiers import (
    CandidateId,
    PlanId,
    ProjectId,
    RunId,
    WorkerId,
    WorkspaceId,
)
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ToolResult
from domain.value_objects.validation import ValidationReport
from infrastructure.database.event_codec import EVENT_TYPES, UnknownDomainEventError, load_event
from infrastructure.database.mappers import (
    candidate_to_domain,
    candidate_to_model,
    job_to_domain,
    job_to_model,
    plan_to_domain,
    plan_to_model,
    project_to_domain,
    project_to_model,
    review_to_domain,
    review_to_model,
    run_to_domain,
    run_to_model,
    tool_result_to_domain,
    tool_result_to_model,
    worker_to_domain,
    worker_to_model,
)


def run_state(run: Run) -> dict[str, Any]:
    """Everything a resumed orchestrator reads off a run."""
    return {
        "id": run.id,
        "project_id": run.project_id,
        "objective": run.objective,
        "status": run.status,
        "candidate_count": run.candidate_count,
        "limits": run.limits,
        "plan_id": run.plan_id,
        "plan_revisions": run.plan_revisions,
        "repair_iterations": run.repair_iterations,
        "review_iterations": run.review_iterations,
        "selected_candidate_id": run.selected_candidate_id,
        "token_usage": run.token_usage,
        "idempotency_key": run.idempotency_key,
        "metadata": dict(run.metadata),
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "updated_at": run.updated_at,
        "failure_kind": run.failure_kind,
        "failure_reason": run.failure_reason,
    }


def job_state(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "run_id": job.run_id,
        "project_id": job.project_id,
        "type": job.type,
        "role": job.role,
        "candidate_id": job.candidate_id,
        "priority": job.priority,
        "status": job.status,
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "payload": dict(job.payload),
        "requirements": job.requirements,
        "idempotency_key": job.idempotency_key,
        "lease": job.lease,
        "assigned_worker_id": job.assigned_worker_id,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": job.result,
        "failure_kind": job.failure_kind,
        "failure_reason": job.failure_reason,
    }


def candidate_state(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "run_id": candidate.run_id,
        "index": candidate.index,
        "status": candidate.status,
        "workspace_id": candidate.workspace_id,
        "patch": candidate.patch,
        "validation": candidate.validation,
        "summary": candidate.summary,
        "uncertainties": candidate.uncertainties,
        "coder_iterations": candidate.coder_iterations,
        "repair_iterations": candidate.repair_iterations,
        "worker_id": candidate.worker_id,
        "last_job_id": candidate.last_job_id,
        "review_verdict": candidate.review_verdict,
        "created_at": candidate.created_at,
        "started_at": candidate.started_at,
        "completed_at": candidate.completed_at,
        "failure_kind": candidate.failure_kind,
        "failure_reason": candidate.failure_reason,
    }


def worker_state(worker: Worker) -> dict[str, Any]:
    return {
        "id": worker.id,
        "endpoint": worker.endpoint,
        "capabilities": worker.capabilities,
        "status": worker.status,
        "load": worker.load,
        "registered_at": worker.registered_at,
        "last_heartbeat_at": worker.last_heartbeat_at,
        "metadata": dict(worker.metadata),
    }


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------
def test_project_round_trip(project: Project) -> None:
    restored = project_to_domain(project_to_model(project))

    assert restored == project
    assert restored.toolchain == project.toolchain
    assert restored.metadata == project.metadata


def test_run_round_trip_keeps_budgets_counters_and_failure(
    new_run: Callable[..., Run], now: datetime
) -> None:
    run = new_run(idempotency_key="create-run-42")
    run.start_planning(now)
    run.plan_ready(plan_id=PlanId.generate(), task_count=2, now=now)
    run.start_coding(now)
    run.start_validating(now)
    candidate_id = CandidateId.generate()
    run.start_reviewing(candidate_id=candidate_id, now=now)
    assert run.request_repair(candidate_id=candidate_id, reason="tests fail", now=now)
    run.fail(now=now, kind=FailureKind.TEST, reason="the suite never went green")

    restored = run_to_domain(run_to_model(run))

    assert run_state(restored) == run_state(run)
    # Spelt out because these are exactly the values a silent mapping bug loses.
    assert restored.limits.max_repair_iterations == 2
    assert restored.repair_iterations == 1
    assert restored.plan_revisions == 1
    assert restored.review_iterations == 1
    assert restored.failure_kind is FailureKind.TEST


def test_run_round_trip_of_a_fresh_run(new_run: Callable[..., Run]) -> None:
    run = new_run()

    restored = run_to_domain(run_to_model(run))

    assert run_state(restored) == run_state(run)
    assert restored.idempotency_key is None


def test_job_round_trip_keeps_the_lease(
    new_run: Callable[..., Run],
    new_job: Callable[..., Job],
    now: datetime,
    lease_duration: timedelta,
) -> None:
    job = new_job(run=new_run(), idempotency_key="enqueue-code-1")
    job.enqueue(now)
    worker_id = WorkerId.generate()
    lease = job.lease_to(worker_id=worker_id, now=now, duration=lease_duration)
    job.mark_running(token=lease.token, now=now)

    restored = job_to_domain(job_to_model(job))

    assert job_state(restored) == job_state(job)
    assert restored.lease is not None
    assert restored.lease.token == lease.token
    assert restored.lease.expires_at == now + lease_duration
    assert restored.status is JobStatus.RUNNING


def test_job_round_trip_of_a_dead_job(
    new_run: Callable[..., Run], new_job: Callable[..., Job], now: datetime
) -> None:
    job = new_job(run=new_run(), candidate_id=CandidateId.generate())
    job.enqueue(now)
    job.lease_to(worker_id=WorkerId.generate(), now=now, duration=timedelta(minutes=1))
    job.fail(token=None, now=now, kind=FailureKind.INFERENCE, reason="context overflow")

    restored = job_to_domain(job_to_model(job))

    assert job_state(restored) == job_state(job)
    assert restored.lease is None
    assert restored.assigned_worker_id is None
    assert restored.attempt == 1


def test_candidate_round_trip_keeps_patch_and_validation(
    new_run: Callable[..., Run],
    new_candidate: Callable[..., Candidate],
    candidate_patch: Patch,
    tool_results: tuple[ToolResult, ...],
    now: datetime,
) -> None:
    candidate = new_candidate(run=new_run(), index=1)
    candidate.attach_workspace(WorkspaceId.generate())
    candidate.record_worker(WorkerId.generate())
    candidate.start_coding(now=now)
    candidate.submit_patch(
        patch=candidate_patch, now=now, summary="added the endpoint", uncertainties=("naming",)
    )
    candidate.start_validation(now)
    candidate.record_validation(
        report=ValidationReport(results=tool_results, static_analysis_is_blocking=True), now=now
    )
    candidate.record_review(ReviewVerdict.FAIL)

    restored = candidate_to_domain(candidate_to_model(candidate))

    assert candidate_state(restored) == candidate_state(candidate)
    assert restored.patch is not None
    assert restored.patch.changed_paths == candidate_patch.changed_paths
    assert restored.patch.total_churn == candidate_patch.total_churn
    assert restored.validation.tests_passed is False
    assert restored.validation.is_viable is False


def test_candidate_round_trip_without_a_patch(
    new_run: Callable[..., Run], new_candidate: Callable[..., Candidate]
) -> None:
    candidate = new_candidate(run=new_run())

    restored = candidate_to_domain(candidate_to_model(candidate))

    assert candidate_state(restored) == candidate_state(candidate)
    assert restored.patch is None
    assert restored.validation == ValidationReport()


def test_plan_round_trip_keeps_task_order_and_dependencies(
    new_run: Callable[..., Run], new_plan: Callable[..., Plan]
) -> None:
    plan = new_plan(run=new_run(), revision=2)

    restored = plan_to_domain(plan_to_model(plan))

    assert restored == plan
    assert [task.key for task in restored.tasks] == [task.key for task in plan.tasks]
    assert restored.task("tests").depends_on == ("route",)
    assert restored.execution_layers() == plan.execution_layers()


def test_review_round_trip_keeps_findings_in_order(
    new_run: Callable[..., Run],
    new_candidate: Callable[..., Candidate],
    new_review: Callable[..., Review],
) -> None:
    run = new_run()
    review = new_review(run=run, candidate=new_candidate(run=run))

    restored = review_to_domain(review_to_model(review))

    assert restored == review
    assert restored.findings == review.findings
    assert restored.repair_brief() == review.repair_brief()


def test_worker_round_trip(worker: Worker, now: datetime) -> None:
    worker.heartbeat(now=now + timedelta(seconds=30))

    restored = worker_to_domain(worker_to_model(worker))

    assert worker_state(restored) == worker_state(worker)
    assert restored.capabilities.supported_roles == worker.capabilities.supported_roles


def test_tool_result_round_trip(tool_results: tuple[ToolResult, ...]) -> None:
    for position, result in enumerate(tool_results):
        model = tool_result_to_model(
            result, run_id=RunId.generate(), candidate_id=None, position=position
        )
        assert tool_result_to_domain(model) == result


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def test_every_event_class_is_registered_under_a_unique_name() -> None:
    exported = {
        name
        for name in domain_events.__all__
        if isinstance(getattr(domain_events, name), type)
        and issubclass(getattr(domain_events, name), DomainEvent)
        and getattr(domain_events, name) is not DomainEvent
    }
    assert len(EVENT_TYPES) == len(exported), "two events share a `name`, rows would collide"


@pytest.mark.parametrize(
    "event",
    [
        RunCreated(
            occurred_at=datetime(2026, 2, 1, 9, 0, tzinfo=UTC),
            run_id=RunId.generate(),
            project_id=ProjectId.generate(),
            objective="ship it",
            candidate_count=3,
        ),
        RunStateChanged(
            occurred_at=datetime(2026, 2, 1, 9, 1, tzinfo=UTC),
            run_id=RunId.generate(),
            previous=RunStatus.CODING,
            current=RunStatus.VALIDATING,
            reason=None,
        ),
        WorkerRegistered(
            occurred_at=datetime(2026, 2, 1, 9, 2, tzinfo=UTC),
            worker_id=WorkerId.generate(),
            model_id="Qwen3-Coder-30B-A3B",
            endpoint="http://worker:8000",
            max_concurrency=4,
        ),
    ],
    ids=["run_created", "run_state_changed", "worker_registered"],
)
def test_event_round_trip_restores_typed_fields(event: DomainEvent) -> None:
    restored = load_event(
        name=type(event).name,
        payload=event.payload(),
        occurred_at=event.occurred_at,
        event_id=event.event_id,
    )

    assert restored == event
    assert type(restored) is type(event)


def test_loading_an_unknown_event_name_is_loud() -> None:
    with pytest.raises(UnknownDomainEventError):
        load_event(
            name="run.invented",
            payload={},
            occurred_at=datetime.now(tz=UTC),
            event_id=uuid4(),
        )
