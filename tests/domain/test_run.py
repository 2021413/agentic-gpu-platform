"""Run aggregate: budgets, events and cancellation semantics."""

from __future__ import annotations

from datetime import datetime

import pytest

from domain.entities.run import Run
from domain.enums import FailureKind, RunStatus
from domain.events.run import RunCreated, RunStateChanged
from domain.exceptions import InvalidStateTransitionError, RunNotModifiableError
from domain.value_objects.identifiers import CandidateId, PlanId, ProjectId, RunId
from domain.value_objects.limits import RunLimits
from domain.value_objects.llm import TokenUsage


def make_run(now: datetime, **kwargs: object) -> Run:
    defaults: dict[str, object] = {
        "run_id": RunId.generate(),
        "project_id": ProjectId.generate(),
        "objective": "Implement the packet parser and add tests",
        "now": now,
    }
    defaults.update(kwargs)
    return Run.create(**defaults)  # type: ignore[arg-type]


def test_creation_emits_run_created(now: datetime) -> None:
    run = make_run(now)
    (event,) = run.pull_events()
    assert isinstance(event, RunCreated)
    assert event.objective == "Implement the packet parser and add tests"
    assert run.status is RunStatus.CREATED


def test_blank_objective_is_rejected(now: datetime) -> None:
    with pytest.raises(ValueError, match="objective"):
        make_run(now, objective="   ")


def test_candidate_count_cannot_exceed_the_limit(now: datetime) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        make_run(now, candidate_count=4, limits=RunLimits(max_parallel_candidates=3))


def test_every_transition_emits_a_state_change(now: datetime) -> None:
    run = make_run(now)
    run.pull_events()
    run.start_planning(now)
    names = [type(e).__name__ for e in run.pull_events()]
    assert "RunStateChanged" in names
    assert "PlanRequested" in names


def test_repair_budget_is_bounded(now: datetime) -> None:
    run = make_run(now, limits=RunLimits(max_repair_iterations=1))
    candidate = CandidateId.generate()
    run.start_planning(now)
    run.plan_ready(plan_id=PlanId.generate(), task_count=1, now=now)
    run.start_coding(now)
    run.start_validating(now)
    run.start_reviewing(candidate_id=candidate, now=now)

    assert run.request_repair(candidate_id=candidate, reason="tests fail", now=now) is True

    run.start_coding(now)
    run.start_validating(now)
    run.start_reviewing(candidate_id=candidate, now=now)
    assert run.request_repair(candidate_id=candidate, reason="tests fail", now=now) is False
    assert run.status is RunStatus.REVIEWING


def test_plan_revision_budget_is_bounded(now: datetime) -> None:
    run = make_run(now, limits=RunLimits(max_plan_revisions=1))
    run.start_planning(now)
    run.plan_ready(plan_id=PlanId.generate(), task_count=1, now=now)
    assert run.request_plan_revision(now) is True
    run.plan_ready(plan_id=PlanId.generate(), task_count=1, now=now)
    assert run.request_plan_revision(now) is False


def test_cancellation_is_idempotent_and_terminal(now: datetime) -> None:
    run = make_run(now)
    assert run.request_cancellation(now=now, reason="user asked") is True
    assert run.request_cancellation(now=now) is False
    run.confirm_cancelled(now=now)
    run.confirm_cancelled(now=now)
    assert run.status is RunStatus.CANCELLED
    assert run.failure_kind is FailureKind.CANCELLED


def test_a_completed_run_cannot_be_cancelled(now: datetime) -> None:
    run = make_run(now)
    run.start_planning(now)
    run.plan_ready(plan_id=PlanId.generate(), task_count=1, now=now)
    run.start_coding(now)
    run.start_validating(now)
    run.start_reviewing(candidate_id=CandidateId.generate(), now=now)
    run.complete(now=now)
    assert run.request_cancellation(now=now) is False
    with pytest.raises(RunNotModifiableError):
        run.ensure_modifiable()


def test_failing_twice_keeps_the_first_reason(now: datetime) -> None:
    run = make_run(now)
    run.fail(now=now, kind=FailureKind.INFRASTRUCTURE, reason="no worker ever appeared")
    run.fail(now=now, kind=FailureKind.TEST, reason="second call")
    assert run.failure_reason == "no worker ever appeared"


def test_illegal_transition_raises(now: datetime) -> None:
    run = make_run(now)
    with pytest.raises(InvalidStateTransitionError):
        run.start_validating(now)


def test_token_usage_accumulates(now: datetime) -> None:
    run = make_run(now)
    run.add_token_usage(TokenUsage(100, 20), now)
    run.add_token_usage(TokenUsage(5, 1), now)
    assert run.token_usage == TokenUsage(105, 21)
    assert run.token_usage.total_tokens == 126


def test_state_change_carries_previous_and_current(now: datetime) -> None:
    run = make_run(now)
    run.pull_events()
    run.start_planning(now)
    change = next(e for e in run.pull_events() if isinstance(e, RunStateChanged))
    assert change.previous is RunStatus.CREATED
    assert change.current is RunStatus.PLANNING
