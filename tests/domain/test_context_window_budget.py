"""A prompt must leave room for the answer (spec section 5).

`fits` compared the prompt against the whole window, so a prompt of exactly
16384 tokens "fitted" a 16384-token worker with nothing left to reply with.
vLLM answers that with a 400, or truncates mid-JSON — which the structured
parser then reports as malformed JSON, and the repair loop spends three more
generations on a cause it was never told about.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from domain.entities.worker import Worker
from domain.enums import AgentRole
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import JobRequirements, WorkerCapabilities, WorkerEndpoint

SERVED = 16_384


def caps(context_length: int = SERVED) -> WorkerCapabilities:
    return WorkerCapabilities(model_id="qwen3-coder", context_length=context_length)


def test_a_prompt_that_fills_the_window_leaves_no_room_to_answer() -> None:
    needed = JobRequirements(
        role=AgentRole.CODER,
        estimated_prompt_tokens=SERVED,
        reserved_output_tokens=2_048,
    )

    assert caps().fits(needed) is False


def test_a_prompt_that_leaves_room_is_accepted() -> None:
    needed = JobRequirements(
        role=AgentRole.CODER,
        estimated_prompt_tokens=SERVED - 4_096,
        reserved_output_tokens=2_048,
    )

    assert caps().fits(needed) is True


def test_the_boundary_is_exact() -> None:
    """One token either side of the limit, so the comparison cannot drift."""
    exact = JobRequirements(
        role=AgentRole.CODER, estimated_prompt_tokens=SERVED - 2_048, reserved_output_tokens=2_048
    )
    over = JobRequirements(
        role=AgentRole.CODER,
        estimated_prompt_tokens=SERVED - 2_048 + 1,
        reserved_output_tokens=2_048,
    )

    assert caps().fits(exact) is True
    assert caps().fits(over) is False


def test_usable_prompt_tokens_says_what_is_actually_available() -> None:
    assert caps().usable_prompt_tokens(reserved_output_tokens=2_048) == SERVED - 2_048
    # Never negative: a reserve larger than the window means nothing fits,
    # which must read as zero rather than as a negative budget a caller
    # would happily pass to a slicing operation.
    assert caps(1_024).usable_prompt_tokens(reserved_output_tokens=2_048) == 0


def test_a_negative_reserve_is_refused() -> None:
    with pytest.raises(ValueError, match="reserved_output_tokens"):
        JobRequirements(role=AgentRole.CODER, reserved_output_tokens=-1)


def test_a_worker_that_cannot_hold_the_prompt_says_why() -> None:
    """The rejection must be diagnosable, not just False."""
    worker = Worker.register(
        worker_id=WorkerId.generate(),
        endpoint=WorkerEndpoint("http://gpu:8000"),
        capabilities=caps(),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    needed = JobRequirements(
        role=AgentRole.CODER, estimated_prompt_tokens=SERVED, reserved_output_tokens=2_048
    )

    reason = worker.rejection_reason(needed)
    assert reason is not None
    assert "context" in reason.lower()
