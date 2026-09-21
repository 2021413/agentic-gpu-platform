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
