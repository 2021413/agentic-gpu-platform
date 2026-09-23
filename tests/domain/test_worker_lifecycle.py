"""Worker lifecycle and eligibility (spec sections 5 and 25)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from tests.conftest import make_worker

from domain.entities.worker import Worker
from domain.enums import AgentRole, WorkerStatus
from domain.exceptions import InvalidStateTransitionError, WorkerUnavailableError
from domain.value_objects.worker import JobRequirements, WorkerLoad

CODE = JobRequirements(role=AgentRole.CODER)


def test_registration_emits_an_event(worker: Worker) -> None:
    assert [e.name for e in worker.pending_events] == ["worker.registered"]
    assert worker.status is WorkerStatus.READY


def test_capacity_is_a_hard_ceiling(now: datetime) -> None:
    worker = make_worker(now=now, concurrency=1)
    worker.reserve_slot(CODE)
    assert worker.status is WorkerStatus.BUSY
    assert worker.can_accept(CODE) is False
    with pytest.raises(WorkerUnavailableError):
        worker.reserve_slot(CODE)


def test_a_draining_worker_receives_no_new_jobs(worker: Worker, now: datetime) -> None:
    worker.start_draining(now)
    assert worker.can_accept(CODE) is False
    assert worker.rejection_reason(CODE) == "status=DRAINING"


def test_a_draining_worker_finishes_its_work_then_is_drained(now: datetime) -> None:
    worker = make_worker(now=now, concurrency=2)
    worker.reserve_slot(CODE)
    worker.start_draining(now)
    assert worker.is_drained is False
    worker.release_slot()
    assert worker.is_drained is True


def test_draining_twice_is_a_no_op(worker: Worker, now: datetime) -> None:
    worker.start_draining(now)
    worker.pull_events()
    worker.start_draining(now)
    assert worker.pending_events == ()


def test_an_offline_worker_cannot_start_draining(worker: Worker, now: datetime) -> None:
    worker.mark_unavailable(now=now, reason="heartbeat timeout")
    with pytest.raises(InvalidStateTransitionError):
        worker.start_draining(now)


def test_role_incompatibility_is_reported(now: datetime) -> None:
    worker = make_worker(now=now, roles=frozenset({AgentRole.PLANNER}))
    assert worker.can_accept(CODE) is False
    assert worker.rejection_reason(CODE) == "role CODER unsupported"


def test_model_pinning_is_honoured(worker: Worker) -> None:
    pinned = JobRequirements(role=AgentRole.CODER, model_id="some-other-model")
    assert worker.can_accept(pinned) is False
    assert "some-other-model" in (worker.rejection_reason(pinned) or "")


def test_a_prompt_larger_than_the_context_window_is_refused(now: datetime) -> None:
    worker = make_worker(now=now, context_length=1000)
    huge = JobRequirements(role=AgentRole.CODER, estimated_prompt_tokens=4000)
    assert worker.can_accept(huge) is False
    reason = worker.rejection_reason(huge) or ""
    # The numbers matter: "too big" sends an operator looking at the wrong
    # knob, while the window and the reserve say which one to turn.
    assert "4000" in reason and "1000" in reason, reason


def test_a_prompt_that_fits_the_window_but_not_the_reply_is_refused(now: datetime) -> None:
    """The half of the rule that was missing.

    900 tokens fit a 1000-token window, but not with 256 kept back for the
    answer — and an answer with nowhere to go is a truncated one.
    """
    worker = make_worker(now=now, context_length=1000)
    tight = JobRequirements(
        role=AgentRole.CODER, estimated_prompt_tokens=900, reserved_output_tokens=256
    )

    assert worker.can_accept(tight) is False
    assert "reserved for the reply" in (worker.rejection_reason(tight) or "")


def test_staleness_is_measured_from_the_last_heartbeat(worker: Worker, now: datetime) -> None:
    timeout = timedelta(seconds=30)
    assert worker.is_stale(now + timedelta(seconds=10), timeout) is False
    assert worker.is_stale(now + timedelta(seconds=31), timeout) is True

    worker.heartbeat(now=now + timedelta(seconds=25))
    assert worker.is_stale(now + timedelta(seconds=31), timeout) is False


def test_a_heartbeat_readmits_a_worker_declared_offline(worker: Worker, now: datetime) -> None:
    worker.mark_unavailable(now=now, reason="heartbeat timeout")
    assert worker.status is WorkerStatus.OFFLINE

    worker.heartbeat(now=now + timedelta(seconds=1))
    assert worker.status is WorkerStatus.READY
    assert worker.can_accept(CODE) is True


def test_heartbeat_refreshes_self_reported_load(worker: Worker, now: datetime) -> None:
    worker.heartbeat(now=now, load=WorkerLoad(active_jobs=2))
    assert worker.active_jobs == 2
    assert worker.status is WorkerStatus.BUSY


def test_deregistration_is_recorded(worker: Worker, now: datetime) -> None:
    worker.deregister(now=now, graceful=True)
    assert worker.status is WorkerStatus.OFFLINE
    assert "worker.deregistered" in [e.name for e in worker.pending_events]
