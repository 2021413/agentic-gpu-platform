"""Failures are not interchangeable (spec section 11)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from domain.enums import FailureKind
from domain.services.retry_policy import RetryAction, RetryPolicy


@pytest.fixture
def policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=3, base_delay=timedelta(seconds=1))


@pytest.mark.parametrize("kind", [FailureKind.INFRASTRUCTURE, FailureKind.INFERENCE])
def test_infrastructure_problems_move_to_another_worker(
    policy: RetryPolicy, kind: FailureKind
) -> None:
    decision = policy.decide(kind=kind, attempt=1)
    assert decision.action is RetryAction.RETRY_OTHER_WORKER
    assert decision.should_retry_job


def test_a_failed_tool_retries_on_the_same_worker(policy: RetryPolicy) -> None:
    assert policy.decide(kind=FailureKind.TOOL, attempt=1).action is RetryAction.RETRY_SAME_WORKER


@pytest.mark.parametrize("kind", [FailureKind.COMPILATION, FailureKind.TEST, FailureKind.REVIEW])
def test_code_defects_go_to_the_repair_loop_not_the_retry_budget(
    policy: RetryPolicy, kind: FailureKind
) -> None:
    decision = policy.decide(kind=kind, attempt=1)
    assert decision.action is RetryAction.AGENTIC_REPAIR
    assert not decision.should_retry_job


def test_invalid_structured_output_is_re_asked_then_fails(policy: RetryPolicy) -> None:
    assert (
        policy.decide(kind=FailureKind.INVALID_STRUCTURED_OUTPUT, attempt=1).action
        is RetryAction.REPAIR_PROMPT
    )
    assert (
        policy.decide(kind=FailureKind.INVALID_STRUCTURED_OUTPUT, attempt=2).action
        is RetryAction.FAIL
    )


def test_cancellation_never_retries(policy: RetryPolicy) -> None:
    assert policy.decide(kind=FailureKind.CANCELLED, attempt=1).action is RetryAction.FAIL


def test_the_budget_is_bounded(policy: RetryPolicy) -> None:
    assert policy.decide(kind=FailureKind.INFRASTRUCTURE, attempt=3).action is RetryAction.FAIL


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(base_delay=timedelta(seconds=1), max_delay=timedelta(seconds=4))
    assert policy.backoff(1) == timedelta(seconds=1)
    assert policy.backoff(2) == timedelta(seconds=2)
    assert policy.backoff(9) == timedelta(seconds=4)
