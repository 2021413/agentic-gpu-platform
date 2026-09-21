"""Retry decisions (spec section 11).

Failures are not interchangeable. An unreachable worker deserves another
worker; a failing test deserves the repair loop; malformed structured output
deserves one repair prompt, then honesty. Encoding that distinction here keeps
it out of every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum, unique

from domain.enums import FailureKind

__all__ = ["RetryAction", "RetryDecision", "RetryPolicy"]


@unique
class RetryAction(StrEnum):
    RETRY_SAME_WORKER = "RETRY_SAME_WORKER"
    RETRY_OTHER_WORKER = "RETRY_OTHER_WORKER"
    """Infrastructure failed: the job is fine, the machine was not."""

    REPAIR_PROMPT = "REPAIR_PROMPT"
    """The model answered badly; re-ask with the validation error attached."""

    AGENTIC_REPAIR = "AGENTIC_REPAIR"
    """The produced code is wrong; that is work for the coder, not a retry."""

    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    action: RetryAction
    delay: timedelta = timedelta(0)
    reason: str = ""

    @property
    def should_retry_job(self) -> bool:
        return self.action in (
            RetryAction.RETRY_SAME_WORKER,
            RetryAction.RETRY_OTHER_WORKER,
            RetryAction.REPAIR_PROMPT,
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded, kind-aware retry policy with exponential backoff."""

    max_attempts: int = 3
    base_delay: timedelta = timedelta(seconds=1)
    max_delay: timedelta = timedelta(seconds=30)
    max_structured_output_repairs: int = 2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def backoff(self, attempt: int) -> timedelta:
        """Exponential backoff, capped. ``attempt`` is 1-based."""
        seconds = self.base_delay.total_seconds() * (2 ** max(attempt - 1, 0))
        return timedelta(seconds=min(seconds, self.max_delay.total_seconds()))

    def decide(self, *, kind: FailureKind, attempt: int) -> RetryDecision:  # noqa: PLR0911
        """What to do after attempt ``attempt`` failed with ``kind``."""
        if kind is FailureKind.CANCELLED:
            return RetryDecision(RetryAction.FAIL, reason="run cancelled")

        if kind.is_code_defect:
            # Not a retry: the deterministic tools told us the code is wrong.
            return RetryDecision(
                RetryAction.AGENTIC_REPAIR, reason=f"{kind} is repaired by the coder, not retried"
            )

        if kind is FailureKind.OUTPUT_TRUNCATED:
            # Re-asking spends another full generation to hit the same wall.
            # The parser refuses to repair it for that reason; the job level
            # used to retry it anyway, twice, on billed hardware.
            return RetryDecision(
                RetryAction.FAIL,
                reason=(
                    "the answer ran out of room before it was finished; "
                    "raise the worker's token budget or narrow the prompt"
                ),
            )

        if kind is FailureKind.INVALID_STRUCTURED_OUTPUT:
            if attempt >= self.max_structured_output_repairs:
                return RetryDecision(
                    RetryAction.FAIL, reason="structured output still invalid after repairs"
                )
            return RetryDecision(
                RetryAction.REPAIR_PROMPT,
                delay=timedelta(0),
                reason="re-ask the model with the schema violation attached",
            )

        if attempt >= self.max_attempts:
            return RetryDecision(RetryAction.FAIL, reason="retry budget exhausted")

        if kind is FailureKind.INFRASTRUCTURE:
            return RetryDecision(
                RetryAction.RETRY_OTHER_WORKER,
                delay=self.backoff(attempt),
                reason="worker or transport failed",
            )
        if kind is FailureKind.INFERENCE:
            return RetryDecision(
                RetryAction.RETRY_OTHER_WORKER,
                delay=self.backoff(attempt),
                reason="inference failed; another worker may succeed",
            )
        if kind is FailureKind.TOOL:
            return RetryDecision(
                RetryAction.RETRY_SAME_WORKER,
                delay=self.backoff(attempt),
                reason="tool could not run; the workspace is unchanged",
            )
        return RetryDecision(RetryAction.FAIL, reason=f"unhandled failure kind {kind}")
