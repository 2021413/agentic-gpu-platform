"""Repository behaviour against a real PostgreSQL.

These tests exist for what only a server can prove: unique indexes actually
rejecting a replayed idempotency key, the lease reclaim query selecting exactly
the stuck jobs, and the event sequence staying monotonic per run across
transactions.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan
from domain.entities.project import Project, ToolchainConfig
from domain.entities.review import Review
from domain.entities.run import Run
from domain.enums import FailureKind, JobStatus, JobType, ReviewVerdict, RunStatus
from domain.events.worker import WorkerRegistered
from domain.exceptions import EntityNotFoundError
from domain.value_objects.identifiers import CandidateId, IdempotencyKey, RunId, WorkerId
from domain.value_objects.llm import TokenUsage
from domain.value_objects.patch import Patch
from domain.value_objects.tools import ToolResult
from domain.value_objects.validation import ValidationReport
from infrastructure.database.models import RunEventModel
from infrastructure.database.unit_of_work import SqlAlchemyUnitOfWork

pytestmark = pytest.mark.integration

Factory = async_sessionmaker[AsyncSession]


def unit(session_factory: Factory) -> SqlAlchemyUnitOfWork:
    """A fresh unit of work, i.e. a fresh transaction."""
    return SqlAlchemyUnitOfWork(session_factory)


async def store_project_and_run(session_factory: Factory, project: Project, run: Run) -> None:
    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.runs.add(run)
        await uow.commit()


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------
async def test_project_is_readable_by_id_and_by_name(
    session_factory: Factory, project: Project
) -> None:
    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.commit()

    async with unit(session_factory) as uow:
        assert await uow.projects.get(project.id) == project
        by_name = await uow.projects.get_by_name(project.name)
        assert by_name is not None
        assert by_name.toolchain == project.toolchain
        assert await uow.projects.get_by_name("absent") is None
        assert list(await uow.projects.list_all()) == [project]


async def test_correcting_a_toolchain_rewrites_one_column_of_the_same_row(
    session_factory: Factory, project: Project
) -> None:
    """The toolchain already has a column, so correcting it needs no migration.

    Checked against the server because that is the only thing that can prove
    the JSON column really took the new commands, and that the identity the
    project's runs were performed against — its path, branch, name and
    creation time — was left exactly as it was.
    """
    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.commit()

    async with unit(session_factory) as uow:
        stored = await uow.projects.get(project.id)
        assert stored is not None
        stored.replace_toolchain(ToolchainConfig(language="rust", test_command="cargo test"))
        await uow.projects.update_toolchain(stored)
        await uow.commit()

    async with unit(session_factory) as uow:
        reread = await uow.projects.get(project.id)
        assert reread is not None
        assert reread.toolchain.test_command == "cargo test"
        assert reread.toolchain.build_command is None
        assert reread.toolchain.language == "rust"
        assert reread.name == project.name
        assert reread.repository_url == project.repository_url
        assert reread.local_path == project.local_path
        assert reread.default_branch == project.default_branch
        assert reread.created_at == project.created_at
        # One row, not a second project silently inserted beside the first.
        assert len(await uow.projects.list_all()) == 1


async def test_correcting_the_toolchain_of_an_absent_project_is_refused(
    session_factory: Factory, project: Project
) -> None:
    async with unit(session_factory) as uow:
        with pytest.raises(EntityNotFoundError):
            await uow.projects.update_toolchain(project)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
async def test_run_survives_a_full_round_trip_through_the_database(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    run = new_run(idempotency_key="POST /runs#1")
    run.start_planning(now)
    run.add_token_usage(TokenUsage(input_tokens=1200, output_tokens=340), now)
    await store_project_and_run(session_factory, project, run)

    async with unit(session_factory) as uow:
        stored = await uow.runs.get(run.id)

    assert stored is not None
    assert stored.status is RunStatus.PLANNING
    assert stored.plan_revisions == 1
    assert stored.limits == run.limits
    assert stored.token_usage.total_tokens == 1540
    assert stored.metadata == run.metadata
    assert stored.idempotency_key == IdempotencyKey("POST /runs#1")


async def test_updating_a_run_keeps_the_same_row(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)

    run.start_planning(now)
    run.fail(now=now, kind=FailureKind.INFRASTRUCTURE, reason="no worker ever showed up")
    async with unit(session_factory) as uow:
        await uow.runs.update(run)
        await uow.commit()

    async with unit(session_factory) as uow:
        stored = await uow.runs.get(run.id)
        assert list(await uow.runs.list_by_project(project.id)) == [run]

    assert stored is not None
    assert stored.status is RunStatus.FAILED
    assert stored.failure_kind is FailureKind.INFRASTRUCTURE
    assert stored.failure_reason == "no worker ever showed up"


async def test_a_replayed_idempotency_key_is_rejected_by_the_database(
    session_factory: Factory, project: Project, new_run: Callable[..., Run]
) -> None:
    """Two runs, one key: the unique index is the last line of defence."""
    first = new_run(idempotency_key="POST /runs#same")
    await store_project_and_run(session_factory, project, first)

    duplicate = new_run(idempotency_key="POST /runs#same")
    with pytest.raises(IntegrityError):
        async with unit(session_factory) as uow:
            await uow.runs.add(duplicate)
            await uow.commit()

    async with unit(session_factory) as uow:
        found = await uow.runs.find_by_idempotency_key(IdempotencyKey("POST /runs#same"))
        assert found is not None
        assert found.id == first.id
        assert await uow.runs.get(duplicate.id) is None


async def test_list_active_excludes_terminal_runs_and_counts_by_status(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    planning = new_run()
    planning.start_planning(now)
    done = new_run()
    done.start_coding(now)
    done.start_validating(now)
    done.start_reviewing(candidate_id=CandidateId.generate(), now=now)
    done.complete(now=now)
    fresh = new_run()

    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        for run in (planning, done, fresh):
            await uow.runs.add(run)
        await uow.commit()

    async with unit(session_factory) as uow:
        active = await uow.runs.list_active()
        counts = await uow.runs.count_by_status()

    assert {run.id for run in active} == {planning.id, fresh.id}
    assert counts == {RunStatus.PLANNING: 1, RunStatus.CREATED: 1, RunStatus.COMPLETED: 1}


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
async def test_job_round_trip_and_listings(
    *,
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_job: Callable[..., Job],
    now: datetime,
    lease_duration: timedelta,
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)
    coding = new_job(run=run, idempotency_key="enqueue#1")
    planning = new_job(run=run, job_type=JobType.PLAN)
    coding.enqueue(now)
    lease = coding.lease_to(worker_id=WorkerId.generate(), now=now, duration=lease_duration)

    async with unit(session_factory) as uow:
        await uow.jobs.add(coding)
        await uow.jobs.add(planning)
        await uow.commit()

    async with unit(session_factory) as uow:
        stored = await uow.jobs.get(coding.id)
        queued = await uow.jobs.list_by_status(JobStatus.PENDING)
        by_run = await uow.jobs.list_by_run(run.id)
        by_key = await uow.jobs.find_by_idempotency_key(IdempotencyKey("enqueue#1"))

    assert stored is not None
    assert stored.lease is not None
    assert stored.lease.token == lease.token
    assert stored.requirements == coding.requirements
    assert stored.payload == coding.payload
    assert [job.id for job in queued] == [planning.id]
    assert {job.id for job in by_run} == {coding.id, planning.id}
    assert by_key is not None and by_key.id == coding.id


async def test_list_expired_leases_returns_only_stuck_in_flight_jobs(
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_job: Callable[..., Job],
    now: datetime,
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)

    stuck = new_job(run=run)
    stuck.enqueue(now)
    stuck.lease_to(worker_id=WorkerId.generate(), now=now, duration=timedelta(minutes=1))

    healthy = new_job(run=run)
    healthy.enqueue(now)
    healthy.lease_to(worker_id=WorkerId.generate(), now=now, duration=timedelta(hours=2))

    waiting = new_job(run=run)
    waiting.enqueue(now)

    async with unit(session_factory) as uow:
        for job in (stuck, healthy, waiting):
            await uow.jobs.add(job)
        await uow.commit()

    later = now + timedelta(minutes=30)
    async with unit(session_factory) as uow:
        expired = await uow.jobs.list_expired_leases(now=later)

    assert [job.id for job in expired] == [stuck.id]

    # Reclaiming it is the domain's job; persisting the outcome is ours.
    reclaimed = expired[0]
    assert reclaimed.expire_lease(later) is True
    async with unit(session_factory) as uow:
        await uow.jobs.update(reclaimed)
        await uow.commit()

    async with unit(session_factory) as uow:
        assert list(await uow.jobs.list_expired_leases(now=later)) == []
        stored = await uow.jobs.get(stuck.id)

    assert stored is not None
    assert stored.status is JobStatus.FAILED
    assert stored.lease is None


async def test_a_replayed_job_idempotency_key_is_rejected(
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_job: Callable[..., Job],
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)
    async with unit(session_factory) as uow:
        await uow.jobs.add(new_job(run=run, idempotency_key="worker-callback#7"))
        await uow.commit()

    with pytest.raises(IntegrityError):
        async with unit(session_factory) as uow:
            await uow.jobs.add(new_job(run=run, idempotency_key="worker-callback#7"))
            await uow.commit()


# ---------------------------------------------------------------------------
# plans, candidates, reviews, tool results
# ---------------------------------------------------------------------------
async def test_plan_revisions_are_append_only_and_the_latest_wins(
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_plan: Callable[..., Plan],
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)
    first = new_plan(run=run, revision=1)
    second = new_plan(run=run, revision=2)

    async with unit(session_factory) as uow:
        await uow.plans.add(first)
        await uow.plans.add(second)
        await uow.commit()

    async with unit(session_factory) as uow:
        latest = await uow.plans.latest_for_run(run.id)
        stored = await uow.plans.get(first.id)
        all_revisions = await uow.plans.list_by_run(run.id)

    assert latest is not None and latest.revision == 2
    assert stored == first
    assert stored is not None
    assert [task.key for task in stored.tasks] == ["route", "tests"]
    assert [plan.revision for plan in all_revisions] == [1, 2]


async def test_candidate_state_survives_an_update(
    *,
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_candidate: Callable[..., Candidate],
    candidate_patch: Patch,
    tool_results: tuple[ToolResult, ...],
    now: datetime,
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)
    first = new_candidate(run=run, index=0)
    second = new_candidate(run=run, index=1)

    async with unit(session_factory) as uow:
        await uow.candidates.add(first)
        await uow.candidates.add(second)
        await uow.commit()

    first.start_coding(now=now)
    first.submit_patch(patch=candidate_patch, now=now, summary="done", uncertainties=("naming",))
    first.start_validation(now)
    first.record_validation(report=ValidationReport(results=tool_results), now=now)
    async with unit(session_factory) as uow:
        await uow.candidates.update(first)
        await uow.commit()

    async with unit(session_factory) as uow:
        stored = await uow.candidates.get(first.id)
        listed = await uow.candidates.list_by_run(run.id)

    assert stored is not None
    assert stored.patch is not None
    assert stored.patch.diff == candidate_patch.diff
    assert stored.validation.tests_passed is False
    assert stored.coder_iterations == 1
    assert [candidate.index for candidate in listed] == [0, 1]


async def test_reviews_are_listed_in_iteration_order(
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_candidate: Callable[..., Candidate],
    new_review: Callable[..., Review],
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)
    candidate = new_candidate(run=run)
    async with unit(session_factory) as uow:
        await uow.candidates.add(candidate)
        await uow.commit()

    first = new_review(run=run, candidate=candidate, iteration=1)
    second = new_review(run=run, candidate=candidate, iteration=2)
    async with unit(session_factory) as uow:
        await uow.reviews.add(first)
        await uow.reviews.add(second)
        await uow.commit()

    async with unit(session_factory) as uow:
        listed = await uow.reviews.list_by_candidate(candidate.id)
        latest = await uow.reviews.latest_for_candidate(candidate.id)

    assert [review.iteration for review in listed] == [1, 2]
    assert latest is not None
    assert latest.id == second.id
    assert latest.verdict is ReviewVerdict.FAIL
    assert len(latest.findings) == 2
    assert latest.findings[0].repair_instruction == "add a test asserting a 200 response"


async def test_tool_results_keep_their_execution_order_across_batches(
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_candidate: Callable[..., Candidate],
    tool_results: tuple[ToolResult, ...],
) -> None:
    run = new_run()
    await store_project_and_run(session_factory, project, run)
    candidate = new_candidate(run=run)
    async with unit(session_factory) as uow:
        await uow.candidates.add(candidate)
        await uow.commit()

    async with unit(session_factory) as uow:
        await uow.tool_results.add_many(
            run_id=run.id, candidate_id=candidate.id, results=tool_results[:1]
        )
        await uow.commit()
    async with unit(session_factory) as uow:
        await uow.tool_results.add_many(
            run_id=run.id, candidate_id=candidate.id, results=tool_results[1:]
        )
        await uow.commit()

    async with unit(session_factory) as uow:
        stored = await uow.tool_results.list_by_candidate(candidate.id)

    assert list(stored) == list(tool_results)


# ---------------------------------------------------------------------------
# event store
# ---------------------------------------------------------------------------
async def test_event_sequence_is_monotonic_per_run_across_transactions(
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    now: datetime,
) -> None:
    first_run = new_run()
    second_run = new_run()
    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.runs.add(first_run)
        await uow.runs.add(second_run)
        await uow.events.append(first_run.pull_events())
        await uow.events.append(second_run.pull_events())
        await uow.commit()

    first_run.start_planning(now)
    async with unit(session_factory) as uow:
        await uow.events.append(first_run.pull_events())
        await uow.commit()

    async with unit(session_factory) as uow:
        first_events = await uow.events.list_by_run(first_run.id)
        second_events = await uow.events.list_by_run(second_run.id)
        tail = await uow.events.list_by_run(first_run.id, after_sequence=1)

    assert [sequence for sequence, _ in first_events] == [1, 2, 3]
    # Each run has its own counter: an SSE cursor on one run is unaffected by
    # the traffic of another.
    assert [sequence for sequence, _ in second_events] == [1]
    assert [type(event).name for _, event in first_events] == [
        "run.created",
        "run.state_changed",
        "run.plan_requested",
    ]
    assert [sequence for sequence, _ in tail] == [2, 3]


async def test_events_come_back_as_the_typed_objects_that_were_stored(
    session_factory: Factory, project: Project, new_run: Callable[..., Run]
) -> None:
    run = new_run()
    emitted = run.pull_events()
    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.runs.add(run)
        await uow.events.append(emitted)
        await uow.commit()

    async with unit(session_factory) as uow:
        stored = await uow.events.list_by_run(run.id)

    assert [event for _, event in stored] == list(emitted)


async def test_events_without_a_run_are_stored_outside_any_run_sequence(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    run = new_run()
    fleet_event = WorkerRegistered(
        occurred_at=now,
        worker_id=WorkerId.generate(),
        model_id="Qwen3-Coder-30B-A3B",
        endpoint="http://worker:8000",
        max_concurrency=2,
    )

    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.runs.add(run)
        await uow.events.append([*run.pull_events(), fleet_event])
        await uow.commit()
        rows = (
            await uow.session.execute(
                select(RunEventModel.name, RunEventModel.run_id, RunEventModel.sequence).order_by(
                    RunEventModel.name
                )
            )
        ).all()

    async with unit(session_factory) as uow:
        run_events = await uow.events.list_by_run(run.id)

    assert [name for name, _, _ in rows] == ["run.created", "worker.registered"]
    assert [(run_id is None, sequence) for _, run_id, sequence in rows] == [(False, 1), (True, 1)]
    assert [type(event).name for _, event in run_events] == ["run.created"]


async def test_reading_events_of_an_unknown_run_returns_nothing(
    session_factory: Factory,
) -> None:
    async with unit(session_factory) as uow:
        assert list(await uow.events.list_by_run(RunId.generate())) == []
