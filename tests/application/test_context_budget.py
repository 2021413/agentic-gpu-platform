"""The context is budgeted against the fleet, not against a constant.

`context_max_tokens` was a fixed 24000 while the engines were served with
MAX_MODEL_LEN=16384. The orchestrator therefore asked for a view of the
repository that no worker could hold, and the scheduler — which does check —
would have refused every job the moment the context stopped being empty.

Two numbers that must agree, living in two packages that never spoke.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from tests.application.fakes import FakeWorkerRegistry

from application.orchestration.worker_pool import WorkerPool
from domain.entities.worker import Worker
from domain.enums import AgentRole
from domain.services.scheduling import LeastLoadedCompatibleScheduler
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import JobRequirements, WorkerCapabilities, WorkerEndpoint

NOW = datetime(2026, 1, 1, tzinfo=UTC)
RESERVE = 4_096


def worker(context_length: int) -> Worker:
    return Worker.register(
        worker_id=WorkerId.generate(),
        endpoint=WorkerEndpoint("http://gpu.invalid:8000"),
        capabilities=WorkerCapabilities(model_id="qwen3-coder", context_length=context_length),
        now=NOW,
    )


async def pool_of(*workers: Worker) -> WorkerPool:
    registry = FakeWorkerRegistry()
    for w in workers:
        await registry.register(w, ttl=timedelta(seconds=60))
    return WorkerPool(
        registry=registry,
        scheduler=LeastLoadedCompatibleScheduler(),
        factory=None,  # type: ignore[arg-type]
    )


def wants(role: AgentRole = AgentRole.CODER) -> JobRequirements:
    return JobRequirements(role=role, reserved_output_tokens=RESERVE)


async def test_the_budget_is_what_the_fleet_can_actually_hold() -> None:
    pool = await pool_of(worker(16_384))

    assert await pool.prompt_budget(wants()) == 16_384 - RESERVE


async def test_the_roomiest_worker_sets_the_budget() -> None:
    """A prompt only has to fit *somewhere*, and the scheduler will route it
    there — so the narrowest worker must not cap what the others could take."""
    pool = await pool_of(worker(16_384), worker(65_536))

    assert await pool.prompt_budget(wants()) == 65_536 - RESERVE


async def test_an_empty_fleet_gives_no_budget_rather_than_a_comfortable_guess() -> None:
    """Nothing registered means nothing is known. Inventing a number here is
    how 262144 ended up being advertised against a 16384 engine."""
    pool = await pool_of()

    assert await pool.prompt_budget(wants()) is None


async def test_a_worker_that_cannot_serve_the_role_does_not_set_the_budget() -> None:
    reviewer_only = Worker.register(
        worker_id=WorkerId.generate(),
        endpoint=WorkerEndpoint("http://gpu.invalid:8001"),
        capabilities=WorkerCapabilities(
            model_id="qwen3-coder",
            context_length=262_144,
            supported_roles=frozenset({AgentRole.REVIEWER}),
        ),
        now=NOW,
    )
    pool = await pool_of(worker(16_384), reviewer_only)

    assert await pool.prompt_budget(wants(AgentRole.CODER)) == 16_384 - RESERVE


@pytest.mark.parametrize("reserve", [0, 4_096, 16_384, 99_999])
async def test_the_budget_is_never_negative(reserve: int) -> None:
    pool = await pool_of(worker(16_384))

    budget = await pool.prompt_budget(
        JobRequirements(role=AgentRole.CODER, reserved_output_tokens=reserve)
    )
    assert budget is not None and budget >= 0


# -- the excerpt is not the whole prompt -----------------------------------
#
# `prompt_budget` answers "how large a prompt can the fleet take". That number
# was handed to the context provider as the budget for the code excerpt, which
# assumed the excerpt was the entire prompt. It is not: the instructions, the
# objective, the plan, the accumulated review findings and the JSON schema ride
# in the same window and none of them were counted.
#
# The first real run against a 32768-token engine failed by exactly one token:
#
#     maximum context length is 32768 tokens. However, you requested 4096
#     output tokens and your prompt contains at least 28673 input tokens
#
# 28672 is 32768 - 4096 to the token. The excerpt had filled the whole prompt
# budget and the template pushed it over — after the GPU had been woken.

from application.orchestration.orchestrator import OrchestratorConfig  # noqa: E402

ROOMY = OrchestratorConfig(context_max_tokens=1_000_000, prompt_overhead_tokens=2_048)


def test_the_excerpt_never_fills_the_whole_prompt_budget() -> None:
    fleet = 32_768 - RESERVE  # the window that produced the failure above

    assert ROOMY.excerpt_budget(fleet) < fleet


def test_what_it_leaves_is_exactly_the_allowance() -> None:
    assert ROOMY.excerpt_budget(28_672) == 28_672 - 2_048


def test_a_fleet_too_narrow_for_the_prompt_itself_gets_no_excerpt() -> None:
    """Zero, not a negative budget the provider would treat as unbounded."""
    assert OrchestratorConfig(prompt_overhead_tokens=2_048).excerpt_budget(100) == 0


def test_an_empty_fleet_leaves_the_configured_ceiling_alone() -> None:
    """There is nothing to learn from, and guessing is what caused the
    24000-against-16384 mismatch this whole mechanism exists to prevent."""
    assert ROOMY.excerpt_budget(None) == 1_000_000


def test_the_fleet_only_ever_lowers_the_ceiling() -> None:
    tight = OrchestratorConfig(context_max_tokens=8_000, prompt_overhead_tokens=2_048)

    assert tight.excerpt_budget(1_000_000) == 8_000
