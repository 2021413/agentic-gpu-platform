"""A truncated answer must not be re-asked (spec section 11).

The structured parser already refuses to repair a truncated answer, and says
why: "retrying with the same budget would truncate identically". One level up,
the retry policy saw an INVALID_STRUCTURED_OUTPUT like any other and answered
REPAIR_PROMPT — so the same oversized answer was generated again, twice, on
billed hardware, before the job finally failed for a reason nobody could act
on.

Running out of room is not a schema violation. It is a budget problem, and the
only useful thing to do with it is to say so.
"""

from __future__ import annotations

from domain.enums import FailureKind
from domain.services.retry_policy import RetryAction, RetryPolicy


def policy() -> RetryPolicy:
    return RetryPolicy()


def test_a_truncated_answer_is_never_re_asked() -> None:
    decision = policy().decide(kind=FailureKind.OUTPUT_TRUNCATED, attempt=1)

    assert decision.action is RetryAction.FAIL


def test_it_fails_on_the_very_first_attempt() -> None:
    """Not after the budget: the second generation is already waste."""
    for attempt in (1, 2, 3):
        assert policy().decide(kind=FailureKind.OUTPUT_TRUNCATED, attempt=attempt).action is (
            RetryAction.FAIL
        )


def test_the_reason_points_at_the_budget_not_at_the_schema() -> None:
    """An operator reading this must reach for MAX_MODEL_LEN, not for the prompt."""
    reason = policy().decide(kind=FailureKind.OUTPUT_TRUNCATED, attempt=1).reason

    assert "room" in reason or "budget" in reason or "token" in reason, reason


def test_a_genuine_schema_violation_is_still_repaired() -> None:
    """The narrow fix must not disarm the repair loop it sits next to."""
    decision = policy().decide(kind=FailureKind.INVALID_STRUCTURED_OUTPUT, attempt=1)

    assert decision.action is RetryAction.REPAIR_PROMPT
