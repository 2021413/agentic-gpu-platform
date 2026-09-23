"""Failure recovery: the properties the spec lists as acceptance criteria.

A worker dies mid-job, a model answers garbage, the pool is empty, the
orchestrator restarts. None of these may lose a run or leave one hanging.
"""

from __future__ import annotations

import json
from datetime import timedelta

from tests.application.conftest import Platform
from tests.application.fakes import InMemoryUnitOfWork

from application.dto.commands import CreateRunCommand
from application.orchestration.maintenance import MaintenanceLoop
from application.use_cases.runs import CreateRunUseCase
from application.use_cases.workers import ReapStaleWorkersUseCase
from domain.enums import AgentRole, JobStatus, RunStatus
from domain.exceptions import LLMTimeoutError
from domain.services.task_complexity import HeuristicTaskComplexityPolicy
from domain.value_objects.identifiers import IdempotencyKey
from domain.value_objects.limits import RunLimits

SIMPLE = "Add a retry to the http client"


async def create_run(platform: Platform, project, **kwargs):
    use_case = CreateRunUseCase(
        uow_factory=platform.uow_factory,
        bus=platform.bus,
        clock=platform.clock,
        ids=platform.ids,
        complexity=HeuristicTaskComplexityPolicy(),
        limits=kwargs.pop("limits", RunLimits()),
    )
    return await use_case.execute(
        CreateRunCommand(project_id=project.id, objective=kwargs.pop("objective", SIMPLE), **kwargs)
    )


# ----------------------------------------------------------------------
# idempotency
# ----------------------------------------------------------------------
async def test_the_same_idempotency_key_never_starts_a_second_run(platform: Platform, project):
    key = IdempotencyKey("client-request-1")
    first = await create_run(platform, project, idempotency_key=key)
    second = await create_run(platform, project, idempotency_key=key)

    assert first.id == second.id
    assert len(platform.store.runs.items) == 1
    assert platform.bus.names().count("run.created") == 1


async def test_starting_a_run_twice_is_harmless(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)
    await platform.orchestrator.start(view.id)

    jobs = list(platform.store.jobs.items.values())
    assert len(jobs) == 1, "the second start must not duplicate the first job"


# ----------------------------------------------------------------------
# an empty or failing pool
# ----------------------------------------------------------------------
async def test_an_empty_pool_does_not_fail_the_run_immediately(platform: Platform, project):
    """No worker is a transient condition in an elastic pool, not an error."""
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)

    await platform.executor.run_once()

    run = platform.store.runs.items[view.id]
    assert run.status is not RunStatus.FAILED
    assert platform.queue.pending == 1, "the job must be waiting for a worker, not lost"


async def test_an_empty_pool_is_asked_again_later_not_immediately(
    platform: Platform, project
):
    """The retry policy computes a backoff and, until this, nothing carried it.

    With one worker — which scale-to-zero makes the ordinary case — an immediate
    requeue puts the same question to the same empty pool. A real run spent all
    three of its attempts in four seconds that way, having already planned,
    coded, built and tested two candidates, and died on an absence that lasted
    under a minute.
    """
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)

    await platform.executor.run_once()

    deferred = [when for when in platform.queue.deferred.values() if when is not None]
    assert deferred, "the job went straight back on the queue with no delay"


async def test_a_retry_that_is_worth_making_now_is_not_delayed(
    platform: Platform, project
):
    """The delay is opt-in per failure kind. A decision that carries none must
    still requeue immediately, or every retry pays for this one."""
    await platform.add_worker()
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)

    await platform.executor.run_once()

    assert all(when is None for when in platform.queue.deferred.values())


async def test_a_run_fails_cleanly_once_the_retry_budget_is_spent(platform: Platform, project):
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)

    for _ in range(5):
        if not await platform.executor.run_once():
            break

    run = platform.store.runs.items[view.id]
    assert run.status is RunStatus.FAILED
    assert run.failure_reason
    assert "worker" in run.failure_reason.lower()


async def test_inference_timeouts_move_the_job_to_another_worker(platform: Platform, project):
    await platform.add_worker()
    platform.provider._fail_with = LLMTimeoutError(30.0, model="fake-coder")

    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)
    await platform.executor.run_once()

    job = next(iter(platform.store.jobs.items.values()))
    assert job.attempt == 1
    assert job.status is JobStatus.QUEUED, "an inference timeout must be retried, not fatal"

    platform.provider._fail_with = None
    await platform.drain()
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


# ----------------------------------------------------------------------
# a worker that disappears
# ----------------------------------------------------------------------
async def test_a_dead_worker_releases_its_job_through_lease_expiry(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)

    claimed = await platform.executor.claim_one()
    assert claimed is not None
    job, _ = claimed
    await platform.store.jobs.add(job)

    # The worker never reports again.
    platform.clock.advance(timedelta(seconds=600))
    maintenance = MaintenanceLoop(
        queue=platform.queue,
        uow_factory=platform.uow_factory,
        bus=platform.bus,
        clock=platform.clock,
        reaper=ReapStaleWorkersUseCase(
            registry=platform.registry,
            clock=platform.clock,
            heartbeat_timeout=timedelta(seconds=30),
        ),
        orchestrator=platform.orchestrator,
    )
    requeued = await maintenance.tick()

    assert job.id in requeued, "the abandoned job must become retryable"
    assert "job.lease_expired" in platform.bus.names()
    assert all(not w.status.is_live for w in platform.registry.workers.values())


async def test_reaping_a_silent_worker_is_visible_on_the_event_stream(
    platform: Platform,
) -> None:
    """A fleet losing workers is exactly what an operator needs to see."""
    await platform.add_worker()
    reaper = ReapStaleWorkersUseCase(
        registry=platform.registry,
        clock=platform.clock,
        heartbeat_timeout=timedelta(seconds=30),
        bus=platform.bus,
    )

    platform.clock.advance(timedelta(seconds=600))
    reaped = await reaper.execute()

    assert len(reaped) == 1
    assert "worker.unavailable" in platform.bus.names()


async def test_an_exhausted_job_fails_its_run_instead_of_hanging(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)

    maintenance = MaintenanceLoop(
        queue=platform.queue,
        uow_factory=platform.uow_factory,
        bus=platform.bus,
        clock=platform.clock,
        reaper=ReapStaleWorkersUseCase(
            registry=platform.registry,
            clock=platform.clock,
            heartbeat_timeout=timedelta(seconds=30),
        ),
        orchestrator=platform.orchestrator,
    )

    for _ in range(4):
        claimed = await platform.executor.claim_one()
        if claimed is None:
            break
        job, _ = claimed
        await platform.store.jobs.add(job)
        platform.clock.advance(timedelta(seconds=600))
        await maintenance.tick()

    run = platform.store.runs.items[view.id]
    assert run.status is RunStatus.FAILED
    assert "retry budget" in (run.failure_reason or "")


# ----------------------------------------------------------------------
# malformed model output
# ----------------------------------------------------------------------
async def test_invalid_structured_output_is_repaired_then_accepted(platform: Platform, project):
    await platform.add_worker()
    platform.provider.script(
        AgentRole.CODER,
        "here is your diff, boss",  # not JSON at all
        json.dumps({"summary": "fixed", "diff": "diff --git a/x b/x\n+ok\n", "done": True}),
    )

    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)
    await platform.drain()

    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


async def test_persistently_invalid_output_fails_the_run_honestly(platform: Platform, project):
    await platform.add_worker()
    platform.provider.script(AgentRole.CODER, *["not json"] * 12)

    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)
    await platform.drain()

    run = platform.store.runs.items[view.id]
    assert run.status is RunStatus.FAILED
    assert run.failure_reason


# ----------------------------------------------------------------------
# restart
# ----------------------------------------------------------------------
async def test_an_orchestrator_restart_resumes_active_runs(platform: Platform, project):
    """Durable state lives in the database; a restart loses only intent."""
    await platform.add_worker()
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)
    await platform.drain()

    resumed = await platform.orchestrator.resume_active_runs()
    assert resumed == [], "a completed run must not be resumed"

    second = await create_run(platform, project, objective="Another objective entirely")
    await platform.orchestrator.start(second.id)
    assert second.id in await platform.orchestrator.resume_active_runs()


# ----------------------------------------------------------------------
# events and transactions
# ----------------------------------------------------------------------
async def test_events_are_only_published_after_the_commit(platform: Platform, project):
    uow = InMemoryUnitOfWork(platform.store)
    run = platform.store.runs.items.get((await create_run(platform, project)).id)
    assert run is not None
    published_before = len(platform.bus.published)

    async with uow:
        run.start_planning(platform.clock.now())
        uow.collect(run)
        # no commit: the transaction is abandoned

    assert len(platform.bus.published) == published_before
    assert run.pending_events == (), "abandoned events must not survive a rollback"


async def test_every_stored_event_is_addressable_for_sse_resumption(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project)
    await platform.orchestrator.start(view.id)
    await platform.drain()

    stored = await platform.store.events.list_by_run(view.id)
    sequences = [seq for seq, _ in stored]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences), "sequence numbers must be unique"
