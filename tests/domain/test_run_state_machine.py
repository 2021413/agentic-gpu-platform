"""The transition table is the contract; these tests pin it down."""

from __future__ import annotations

from itertools import pairwise

import pytest

from domain.enums import RunStatus
from domain.exceptions import InvalidStateTransitionError
from domain.services.run_state_machine import RunStateMachine

TERMINAL = (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


def test_happy_path_is_reachable() -> None:
    path = [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.PLAN_READY,
        RunStatus.CODING,
        RunStatus.VALIDATING,
        RunStatus.REVIEWING,
        RunStatus.COMPLETED,
    ]
    for current, nxt in pairwise(path):
        assert RunStateMachine.can(current, nxt), f"{current} -> {nxt} must be legal"


def test_planner_can_be_bypassed_for_trivial_work() -> None:
    assert RunStateMachine.can(RunStatus.CREATED, RunStatus.CODING)


def test_repair_loops_back_into_coding() -> None:
    assert RunStateMachine.can(RunStatus.REVIEWING, RunStatus.REPAIRING)
    assert RunStateMachine.can(RunStatus.REPAIRING, RunStatus.CODING)


@pytest.mark.parametrize("state", [s for s in RunStatus if s not in TERMINAL])
def test_any_live_state_can_start_cancelling(state: RunStatus) -> None:
    expected = state is not RunStatus.CANCELLING
    assert RunStateMachine.can(state, RunStatus.CANCELLING) is expected


@pytest.mark.parametrize("state", TERMINAL)
def test_terminal_states_are_absorbing(state: RunStatus) -> None:
    assert RunStateMachine.allowed_from(state) == frozenset()


def test_shortcuts_are_rejected() -> None:
    assert not RunStateMachine.can(RunStatus.CREATED, RunStatus.COMPLETED)
    assert not RunStateMachine.can(RunStatus.PLANNING, RunStatus.REVIEWING)


def test_self_transition_is_rejected() -> None:
    assert not RunStateMachine.can(RunStatus.CODING, RunStatus.CODING)


def test_ensure_raises_with_both_states() -> None:
    with pytest.raises(InvalidStateTransitionError) as excinfo:
        RunStateMachine.ensure(RunStatus.CREATED, RunStatus.COMPLETED)
    assert excinfo.value.current is RunStatus.CREATED
    assert excinfo.value.requested is RunStatus.COMPLETED


def test_cancelling_only_settles_or_fails() -> None:
    assert RunStateMachine.allowed_from(RunStatus.CANCELLING) == frozenset(
        {RunStatus.CANCELLED, RunStatus.FAILED}
    )
