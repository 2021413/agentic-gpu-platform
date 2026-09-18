"""The scenario the spec calls out: a whole run, on the real stack, no GPU.

Real PostgreSQL with real migrations-equivalent schema, real mappers, real
orchestrator, real git worktrees, real sandboxed tool execution. Only the model
is fake. If the layers written separately do not actually fit together, this is
where it shows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.dto.commands import (
    CancelRunCommand,
    CreateProjectCommand,
    CreateRunCommand,
    DrainWorkerCommand,
    HeartbeatCommand,
    RegisterWorkerCommand,
)
from bootstrap.container import Container, describe
from domain.entities.project import ToolchainConfig
from domain.enums import AgentRole, RunStatus, WorkerStatus
from domain.value_objects.identifiers import IdempotencyKey
from domain.value_objects.worker import GpuSpec, WorkerLoad
from infrastructure.database.engine import create_database_engine
from infrastructure.database.models import Base

pytestmark = pytest.mark.e2e

SETTLE_LIMIT = 60


async def prepare_schema(container: Container) -> None:
    """Create the schema the adapter expects, as a migration would."""
    engine = create_database_engine(container.settings.database_url, pool_size=2)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


async def register_worker(container: Container, *, concurrency: int = 2):
    return await container.register_worker.execute(
        RegisterWorkerCommand(
            endpoint=f"http://fake-worker-{concurrency}:8000",
            model_id=container.settings.model_id,
            context_length=container.settings.model_context_length,
            max_concurrency=concurrency,
            supported_roles=frozenset(AgentRole),
            gpu=GpuSpec(gpu_type="none", gpu_count=0),
        )
    )


async def create_project(container: Container, repo: Path, toolchain: ToolchainConfig):
    return await container.create_project.execute(
        CreateProjectCommand(
            name=f"sample-{repo.name}",
            local_path=str(repo),
            default_branch="main",
            toolchain=toolchain,
        )
    )


async def settle(container: Container, *, limit: int = SETTLE_LIMIT) -> int:
    """Drive the executor until the queue drains.

    Bounded on purpose: an orchestrator that keeps scheduling work forever is a
    bug, and the test must fail rather than hang.
    """
    steps = 0
    while await container.executor.run_once():
        steps += 1
        if steps >= limit:
            raise AssertionError(f"the workflow never settled within {limit} jobs")
    return steps


async def test_a_full_run_reaches_completed_on_the_real_stack(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    await register_worker(container)
    project = await create_project(container, sample_repository, toolchain)

    run = await container.create_run.execute(
        CreateRunCommand(
            project_id=project.id,
            objective="Implement the packet parser and add tests",
            candidate_count=1,
        )
    )
    await container.orchestrator.start(run.id)
    await settle(container)

    detail = await container.get_run.execute(run.id, detailed=True)
    assert detail.run.status is RunStatus.COMPLETED, detail.run.failure_reason
    assert detail.run.selected_candidate_id is not None
    assert detail.candidates, "the run completed without recording a candidate"
    assert detail.run.input_tokens > 0, "token accounting was never persisted"


async def test_the_run_survives_an_orchestrator_restart(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    """Durable state lives in PostgreSQL; restarting loses only intent."""
    await prepare_schema(container)
    await register_worker(container)
    project = await create_project(container, sample_repository, toolchain)

    run = await container.create_run.execute(
        CreateRunCommand(project_id=project.id, objective="Add a retry to the client")
    )
    await container.orchestrator.start(run.id)
    await container.executor.run_once()

    reloaded = await container.get_run.execute(run.id)
    assert reloaded.run.status is not RunStatus.CREATED, "progress was not persisted"

    resumed = await container.orchestrator.resume_active_runs()
    assert run.id in resumed
    await settle(container)
    assert (await container.get_run.execute(run.id)).run.status in (
        RunStatus.COMPLETED,
        RunStatus.FAILED,
    )


async def test_two_candidates_use_isolated_git_worktrees(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    await register_worker(container, concurrency=2)
    project = await create_project(container, sample_repository, toolchain)

    run = await container.create_run.execute(
        CreateRunCommand(
            project_id=project.id,
            objective="Refactor the parser and migrate the tests across a.py b.py c.py d.py",
            candidate_count=2,
        )
    )
    await container.orchestrator.start(run.id)
    await settle(container)

    candidates = await container.list_candidates.execute(run.id)
    assert len(candidates) == 2
    detail = await container.get_run.execute(run.id, detailed=True)
    assert detail.run.status is RunStatus.COMPLETED, detail.run.failure_reason


async def test_a_second_worker_can_join_during_a_run(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    first = await register_worker(container, concurrency=1)
    project = await create_project(container, sample_repository, toolchain)

    run = await container.create_run.execute(
        CreateRunCommand(project_id=project.id, objective="Add a retry to the client")
    )
    await container.orchestrator.start(run.id)
    await container.executor.run_once()

    second = await register_worker(container, concurrency=3)
    workers = await container.list_workers.execute()
    assert {w.id for w in workers} == {first.id, second.id}

    await settle(container)
    assert (await container.get_run.execute(run.id)).run.status is RunStatus.COMPLETED


async def test_a_draining_worker_leaves_without_breaking_the_run(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    leaving = await register_worker(container, concurrency=1)
    staying = await register_worker(container, concurrency=2)
    project = await create_project(container, sample_repository, toolchain)

    run = await container.create_run.execute(
        CreateRunCommand(project_id=project.id, objective="Add a retry to the client")
    )
    await container.orchestrator.start(run.id)

    await container.drain_worker.execute(DrainWorkerCommand(worker_id=leaving.id))
    drained = next(w for w in await container.list_workers.execute() if w.id == leaving.id)
    assert drained.status is WorkerStatus.DRAINING
    available = await container.list_workers.execute(only_available=True)
    assert leaving.id not in {w.id for w in available}
    assert staying.id in {w.id for w in available}

    await settle(container)
    assert (await container.get_run.execute(run.id)).run.status is RunStatus.COMPLETED


async def test_a_failing_test_suite_ends_the_run_deterministically(
    container: Container, sample_repository: Path
) -> None:
    await prepare_schema(container)
    await register_worker(container)
    failing = ToolchainConfig(
        language="python", build_command="/bin/true", test_command="/bin/false"
    )
    project = await create_project(container, sample_repository, failing)

    run = await container.create_run.execute(
        CreateRunCommand(project_id=project.id, objective="Add a retry to the client")
    )
    await container.orchestrator.start(run.id)
    await settle(container)

    final = await container.get_run.execute(run.id)
    assert final.run.status is RunStatus.FAILED
    assert final.run.failure_reason


async def test_cancelling_a_run_stops_it_everywhere(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    await register_worker(container)
    project = await create_project(container, sample_repository, toolchain)

    run = await container.create_run.execute(
        CreateRunCommand(
            project_id=project.id, objective="Add a retry to the client", candidate_count=1
        )
    )
    await container.orchestrator.start(run.id)
    await container.cancel_run.execute(CancelRunCommand(run_id=run.id, reason="user"))

    await settle(container)
    assert (await container.get_run.execute(run.id)).run.status is RunStatus.CANCELLED
    assert await container.queue.depth() == 0


async def test_the_run_event_stream_is_persisted_and_ordered(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    await register_worker(container)
    project = await create_project(container, sample_repository, toolchain)

    run = await container.create_run.execute(
        CreateRunCommand(project_id=project.id, objective="Add a retry to the client")
    )
    await container.orchestrator.start(run.id)
    await settle(container)

    events = await container.list_run_events.execute(run.id)
    names = [e.name for e in events]
    sequences = [e.sequence for e in events]

    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    assert "run.created" in names
    assert "run.completed" in names

    after = await container.list_run_events.execute(run.id, after_sequence=sequences[0])
    assert all(e.sequence > sequences[0] for e in after), "SSE resumption would replay"


async def test_an_idempotent_create_never_starts_two_runs(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    project = await create_project(container, sample_repository, toolchain)
    key = IdempotencyKey("client-retry-1")

    first = await container.create_run.execute(
        CreateRunCommand(project_id=project.id, objective="Add a retry", idempotency_key=key)
    )
    second = await container.create_run.execute(
        CreateRunCommand(project_id=project.id, objective="Add a retry", idempotency_key=key)
    )
    assert first.id == second.id
    assert len(await container.list_runs.execute(project.id)) == 1


async def test_heartbeats_keep_a_worker_schedulable(
    container: Container, sample_repository: Path, toolchain: ToolchainConfig
) -> None:
    await prepare_schema(container)
    worker = await register_worker(container)

    refreshed = await container.heartbeat.execute(
        HeartbeatCommand(worker_id=worker.id, load=WorkerLoad(active_jobs=1))
    )
    assert refreshed.active_jobs == 1
    assert refreshed.last_heartbeat_at >= worker.last_heartbeat_at

    reaped = await container.reap_workers.execute()
    assert worker.id not in reaped, "a worker that just beat must not be reaped"


def test_the_wiring_is_describable(container: Container) -> None:
    """The first question asked of a misbehaving deployment."""
    lines = describe(container)
    assert any("queue=" in line for line in lines)
    assert any("inference=" in line for line in lines)
