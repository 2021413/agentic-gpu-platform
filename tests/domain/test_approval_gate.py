"""Human approval before a patch lands (spec section 34, extension).

Integration is a write into someone's repository. Until now it happened the
moment the reviewer said PASS, with no human in the loop: the run went from
REVIEWING straight to COMPLETED and the merge was already done.

The gate is opt-in. A deployment that wants the old behaviour changes nothing,
because a run that waits for an approval nobody is watching is a run that
never finishes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from domain.enums import RunStatus
from domain.exceptions import InvalidStateTransitionError
from domain.services.run_state_machine import RunStateMachine

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_review_can_lead_to_a_wait_for_approval() -> None:
    assert RunStateMachine.can(RunStatus.REVIEWING, RunStatus.AWAITING_APPROVAL)


def test_an_approved_run_completes() -> None:
    assert RunStateMachine.can(RunStatus.AWAITING_APPROVAL, RunStatus.COMPLETED)


def test_a_rejected_run_can_go_back_to_the_coder() -> None:
    """Rejection is feedback, not a verdict: the reviewer passed it, a human
    did not, and the coder is the one who can act on that."""
    assert RunStateMachine.can(RunStatus.AWAITING_APPROVAL, RunStatus.REPAIRING)


def test_a_waiting_run_can_still_be_cancelled_or_failed() -> None:
    assert RunStateMachine.can(RunStatus.AWAITING_APPROVAL, RunStatus.CANCELLING)
    assert RunStateMachine.can(RunStatus.AWAITING_APPROVAL, RunStatus.FAILED)


def test_waiting_is_not_terminal() -> None:
    """It must keep appearing in the active list, or the sweep would forget it
    and nobody could ever approve it."""
    assert RunStatus.AWAITING_APPROVAL.is_terminal is False


@pytest.mark.parametrize(
    "origin", [RunStatus.CREATED, RunStatus.PLANNING, RunStatus.CODING, RunStatus.VALIDATING]
)
def test_approval_cannot_be_reached_before_a_review(origin: RunStatus) -> None:
    """Otherwise a human would be asked to approve something nobody reviewed."""
    assert not RunStateMachine.can(origin, RunStatus.AWAITING_APPROVAL)

    with pytest.raises(InvalidStateTransitionError):
        RunStateMachine.ensure(origin, RunStatus.AWAITING_APPROVAL)


def test_a_completed_run_cannot_be_sent_back_for_approval() -> None:
    assert not RunStateMachine.can(RunStatus.COMPLETED, RunStatus.AWAITING_APPROVAL)
