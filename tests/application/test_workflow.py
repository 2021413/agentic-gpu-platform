"""The agentic workflow, end to end, with no GPU and no infrastructure.

These are the tests that decide whether the orchestrator is correct: plan, code,
validate, review, repair, and every way that sequence can go wrong.
"""

from __future__ import annotations

import json

import pytest
from tests.application.conftest import Platform

from application.dto.commands import CancelRunCommand, CreateRunCommand
from application.use_cases.runs import CancelRunUseCase, CreateRunUseCase
from domain.enums import AgentRole, CandidateStatus, FailureKind, RunStatus
from domain.services.task_complexity import HeuristicTaskComplexityPolicy
from domain.value_objects.limits import RunLimits

OBJECTIVE = "Refactor the packet parser, migrate the tests across modules a.py b.py c.py d.py"
SIMPLE_OBJECTIVE = "Add a retry to the http client"


async def create_run(
    platform: Platform,
    project,
    *,
    objective: str = OBJECTIVE,
    candidate_count: int | None = None,
    limits: RunLimits | None = None,
):
    use_case = CreateRunUseCase(
        uow_factory=platform.uow_factory,
        bus=platform.bus,
        clock=platform.clock,
        ids=platform.ids,
        complexity=HeuristicTaskComplexityPolicy(),
        limits=limits or RunLimits(),
    )
    return await use_case.execute(
        CreateRunCommand(
            project_id=project.id, objective=objective, candidate_count=candidate_count
        )
    )


async def test_a_run_completes_through_plan_code_validate_review(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=1)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    run = platform.store.runs.items[view.id]
    assert run.status is RunStatus.COMPLETED
    assert run.selected_candidate_id is not None

    names = platform.bus.names()
    for expected in (
        "run.created",
        "run.plan_requested",
        "run.plan_completed",
        "candidate.started",
        "candidate.validation_completed",
        "review.requested",
        "candidate.selected",
        "run.completed",
    ):
        assert expected in names, f"{expected} was never emitted; got {names}"


async def test_the_winning_patch_is_integrated_and_workspaces_released(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=1)
    await platform.orchestrator.start(view.id)
    await platform.drain()

    assert platform.workspaces.integrated, "the accepted candidate was never merged back"
    assert platform.workspaces.handles == {}, "workspaces leaked after the run completed"


async def test_two_candidates_run_in_parallel_on_isolated_workspaces(platform: Platform, project):
    await platform.add_worker(concurrency=2)
    view = await create_run(platform, project, candidate_count=2)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    candidates = list(platform.store.candidates.items.values())
    assert len(candidates) == 2
    paths = {
        platform.workspaces.handles.get(c.workspace_id).path
        if c.workspace_id in platform.workspaces.handles
        else str(c.workspace_id)
        for c in candidates
    }
    assert len(paths) == 2, "two candidates shared a workspace"
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


async def test_candidates_may_land_on_different_workers(platform: Platform, project):
    first = await platform.add_worker(concurrency=1)
    second = await platform.add_worker(concurrency=1)
    view = await create_run(platform, project, candidate_count=2)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    used = {c.worker_id for c in platform.store.candidates.items.values()}
    assert used <= {first.id, second.id}
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


async def test_a_single_worker_executes_candidates_sequentially(platform: Platform, project):
    """Correctness must never depend on parallelism being available."""
    await platform.add_worker(concurrency=1)
    view = await create_run(platform, project, candidate_count=2)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


async def test_a_worker_joining_mid_run_gets_the_next_job(platform: Platform, project):
    view = await create_run(platform, project, candidate_count=1)
    await platform.orchestrator.start(view.id)

    # No worker yet: the planner job cannot be served and is retried, not failed.
    await platform.executor.run_once()
    assert platform.store.runs.items[view.id].status is not RunStatus.FAILED

    await platform.add_worker()
    await platform.drain()
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


async def test_the_planner_is_skipped_for_a_trivial_objective(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project, objective="fix typo in README")
    await platform.orchestrator.start(view.id)
    await platform.drain()

    assert "run.plan_requested" not in platform.bus.names()
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


@pytest.mark.tool_exit_codes({"run_tests": 1})
async def test_a_failing_test_suite_triggers_repair(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(
        platform, project, candidate_count=1, limits=RunLimits(max_repair_iterations=1)
    )
    await platform.orchestrator.start(view.id)
    await platform.drain()

    run = platform.store.runs.items[view.id]
    assert "run.repair_requested" in platform.bus.names()
    assert run.status is RunStatus.FAILED
    assert run.repair_iterations == 1, "the repair budget must be respected exactly"


@pytest.mark.tool_exit_codes({"build": 1})
async def test_a_failing_build_skips_the_test_stage(platform: Platform, project):
    """Running tests against code that does not compile teaches nothing."""
    await platform.add_worker()
    view = await create_run(
        platform, project, candidate_count=1, limits=RunLimits(max_repair_iterations=0)
    )
    await platform.orchestrator.start(view.id)
    await platform.drain()

    ran = {i.tool for i in platform.tools.invocations}
    assert "build" in ran
    assert "run_tests" not in ran
    assert platform.store.runs.items[view.id].status is RunStatus.FAILED


async def test_a_failing_review_sends_the_candidate_back_to_the_coder(platform: Platform, project):
    await platform.add_worker()
    failing_review = json.dumps(
        {
            "verdict": "FAIL",
            "summary": "the parser ignores malformed input",
            "findings": [
                {
                    "summary": "no bounds check",
                    "severity": "BLOCKER",
                    "file": "parser.py",
                    "repair_instruction": "reject packets shorter than the header",
                }
            ],
        }
    )
    platform.provider.script(
        AgentRole.REVIEWER,
        failing_review,
        json.dumps({"verdict": "PASS", "summary": "fixed", "findings": []}),
    )

    view = await create_run(platform, project, candidate_count=1)
    await platform.orchestrator.start(view.id)
    await platform.drain()

    run = platform.store.runs.items[view.id]
    assert "run.repair_requested" in platform.bus.names()
    assert run.status is RunStatus.COMPLETED
    assert run.repair_iterations == 1


async def test_a_run_created_through_the_use_case_actually_starts(
    platform: Platform, project
) -> None:
    """Creating a run and scheduling it are two different things.

    The HTTP layer only creates: it persists the run and answers, so a client is
    never held on a GPU. Nothing then called start(), and _advance ignored
    CREATED, so a run created over the API sat there forever while the API
    reported success. The maintenance sweep is what closes the gap, and this
    test fails if it is ever removed.
    """
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=1)

    # Deliberately no orchestrator.start(): that is what the API does not do.
    assert platform.store.runs.items[view.id].status is RunStatus.CREATED

    started = await platform.orchestrator.advance_stalled_runs()

    assert view.id in started
    assert platform.store.runs.items[view.id].status is not RunStatus.CREATED
    assert platform.queue.pending > 0, "no job was scheduled for the run"

    await platform.drain()
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


async def test_starting_pending_runs_twice_schedules_nothing_extra(
    platform: Platform, project
) -> None:
    """The sweep runs on every maintenance tick, so it must be idempotent."""
    await platform.add_worker()
    await create_run(platform, project, candidate_count=1)

    await platform.orchestrator.advance_stalled_runs()
    scheduled = len(platform.store.jobs.items)

    assert await platform.orchestrator.advance_stalled_runs() == []
    assert len(platform.store.jobs.items) == scheduled


async def test_a_cancelled_run_stops_scheduling(platform: Platform, project):
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=2)
    await platform.orchestrator.start(view.id)

    cancel = CancelRunUseCase(
        uow_factory=platform.uow_factory,
        bus=platform.bus,
        queue=platform.queue,
        clock=platform.clock,
    )
    await cancel.execute(CancelRunCommand(run_id=view.id, reason="user changed their mind"))

    await platform.drain()
    run = platform.store.runs.items[view.id]
    assert run.status is RunStatus.CANCELLED
    assert run.failure_kind is FailureKind.CANCELLED
    assert platform.queue.pending == 0

    candidates = platform.store.candidates.items.values()
    assert all(c.status is CandidateStatus.CANCELLED for c in candidates)


async def test_cancelling_twice_is_idempotent(platform: Platform, project):
    view = await create_run(platform, project)
    cancel = CancelRunUseCase(
        uow_factory=platform.uow_factory,
        bus=platform.bus,
        queue=platform.queue,
        clock=platform.clock,
    )
    first = await cancel.execute(CancelRunCommand(run_id=view.id))
    second = await cancel.execute(CancelRunCommand(run_id=view.id))
    assert first.status is second.status is RunStatus.CANCELLED
    assert platform.bus.names().count("run.cancelled") == 1
