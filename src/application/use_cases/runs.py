"""Run use cases: creation, cancellation and reads.

Creating a run is where the platform decides how much machinery an objective
deserves: whether to plan at all, and how many candidates to race. Both come
from explicit policies, never from a hidden heuristic in a route handler.
"""

from __future__ import annotations

from collections.abc import Sequence

from application.dto.commands import CancelRunCommand, CreateRunCommand
from application.dto.views import (
    CandidatePatchView,
    CandidateView,
    EventView,
    PlanView,
    ReviewView,
    RunDetailView,
    RunView,
)
from application.ports import UnitOfWorkFactory
from application.services.event_publisher import commit_and_publish
from domain.entities.run import Run
from domain.enums import FailureKind
from domain.exceptions import EntityNotFoundError
from domain.ports.clock import Clock, IdGenerator
from domain.ports.event_bus import EventBus
from domain.ports.job_queue import JobQueue
from domain.services.task_complexity import TaskComplexityPolicy
from domain.value_objects.identifiers import CandidateId, ProjectId, RunId
from domain.value_objects.limits import RunLimits

__all__ = [
    "CancelRunUseCase",
    "CreateRunUseCase",
    "GetCandidatePatchUseCase",
    "GetRunUseCase",
    "ListReviewsUseCase",
    "ListRunEventsUseCase",
    "ListRunsUseCase",
]


class CreateRunUseCase:
    """Accept an objective and make the run durable before anything else happens.

    The run is persisted first and scheduled afterwards. A crash between the two
    leaves a CREATED run that the resume loop picks up — the opposite order
    would lose work that a client was already told about.
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        bus: EventBus,
        clock: Clock,
        ids: IdGenerator,
        complexity: TaskComplexityPolicy,
        limits: RunLimits | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._bus = bus
        self._clock = clock
        self._ids = ids
        self._complexity = complexity
        self._limits = limits or RunLimits()

    async def execute(self, command: CreateRunCommand) -> RunView:
        async with self._uow_factory() as uow:
            if command.idempotency_key is not None:
                replayed = await uow.runs.find_by_idempotency_key(command.idempotency_key)
                if replayed is not None:
                    # At-least-once delivery: the same key must never start a
                    # second run, it must return the first one.
                    return RunView.of(replayed)

            project = await uow.projects.get(command.project_id)
            if project is None:
                raise EntityNotFoundError("Project", command.project_id)

            assessment = self._complexity.assess(
                command.objective, max_candidates=self._limits.max_parallel_candidates
            )
            candidate_count = min(
                command.candidate_count or assessment.recommended_candidates,
                self._limits.max_parallel_candidates,
            )

            run = Run.create(
                run_id=self._ids.next_id(RunId),
                project_id=project.id,
                objective=command.objective,
                now=self._clock.now(),
                candidate_count=candidate_count,
                limits=self._limits,
                idempotency_key=command.idempotency_key,
                metadata={
                    **dict(command.metadata),
                    "complexity": str(assessment.complexity),
                    "complexity_rationale": assessment.rationale,
                    "requires_plan": assessment.complexity.requires_plan,
                },
            )
            await uow.runs.add(run)
            uow.collect(run)
            await commit_and_publish(uow, self._bus)
            return RunView.of(run)


class CancelRunUseCase:
    """Stop a run and everything it has in flight (spec section 34).

    Cancellation is idempotent and propagates: no new jobs are scheduled,
    pending jobs are dropped from the queue, and in-flight ones stop being
    renewable. Workspaces are released by the orchestrator once the jobs settle.
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        bus: EventBus,
        queue: JobQueue,
        clock: Clock,
    ) -> None:
        self._uow_factory = uow_factory
        self._bus = bus
        self._queue = queue
        self._clock = clock

    async def execute(self, command: CancelRunCommand) -> RunView:
        async with self._uow_factory() as uow:
            run = await uow.runs.get(command.run_id)
            if run is None:
                raise EntityNotFoundError("Run", command.run_id)

            now = self._clock.now()
            if not run.request_cancellation(now=now, reason=command.reason):
                return RunView.of(run)

            for job in await uow.jobs.list_by_run(run.id):
                job.cancel(now)
                await uow.jobs.update(job)

            for candidate in await uow.candidates.list_by_run(run.id):
                candidate.cancel(now)
                uow.collect(candidate)
                await uow.candidates.update(candidate)

            run.confirm_cancelled(now=now, reason=command.reason)
            await uow.runs.update(run)
            uow.collect(run)
            await commit_and_publish(uow, self._bus)

        # Dropping queued work after the commit keeps the durable state
        # authoritative; a crash in between only leaves jobs the workers will
        # refuse anyway, because the run is already CANCELLED.
        await self._queue.cancel_run_jobs(command.run_id)
        return RunView.of(run)


class GetRunUseCase:
    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, run_id: RunId, *, detailed: bool = False) -> RunDetailView:
        async with self._uow_factory() as uow:
            run = await uow.runs.get(run_id)
            if run is None:
                raise EntityNotFoundError("Run", run_id)
            if not detailed:
                return RunDetailView(run=RunView.of(run))

            plan = await uow.plans.latest_for_run(run_id)
            candidates = await uow.candidates.list_by_run(run_id)
            return RunDetailView(
                run=RunView.of(run),
                plan=PlanView.of(plan) if plan is not None else None,
                candidates=[CandidateView.of(c) for c in candidates],
            )


class ListRunsUseCase:
    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(
        self, project_id: ProjectId, *, limit: int = 50, offset: int = 0
    ) -> Sequence[RunView]:
        async with self._uow_factory() as uow:
            if await uow.projects.get(project_id) is None:
                # An unknown project must not read as "a project with no runs".
                raise EntityNotFoundError("Project", project_id)
            runs = await uow.runs.list_by_project(project_id, limit=limit, offset=offset)
            return [RunView.of(r) for r in runs]


class GetCandidatePatchUseCase:
    """The code a candidate actually wrote.

    Kept out of the candidate listing on purpose: a patch is unbounded, and a
    dashboard polling a list of candidates must not drag every diff with it.
    """

    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, run_id: RunId, candidate_id: CandidateId) -> CandidatePatchView:
        async with self._uow_factory() as uow:
            candidate = await uow.candidates.get(candidate_id)
            if candidate is None or candidate.run_id != run_id:
                # The run check matters: a candidate id from another run must
                # not be readable by guessing a run it does not belong to.
                raise EntityNotFoundError("Candidate", candidate_id)
            return CandidatePatchView.of(candidate)


class ListReviewsUseCase:
    """Every reviewer verdict of a run, with its findings.

    Reviews are append-only evidence: a repair loop produces one per round, and
    reading them in order is how you see what the reviewer kept objecting to.
    """

    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, run_id: RunId) -> Sequence[ReviewView]:
        async with self._uow_factory() as uow:
            if await uow.runs.get(run_id) is None:
                raise EntityNotFoundError("Run", run_id)
            reviews: list[ReviewView] = []
            for candidate in await uow.candidates.list_by_run(run_id):
                for review in await uow.reviews.list_by_candidate(candidate.id):
                    reviews.append(ReviewView.of(review))
        return sorted(reviews, key=lambda r: (r.created_at, r.iteration))


class ListCandidatesUseCase:
    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, run_id: RunId) -> Sequence[CandidateView]:
        async with self._uow_factory() as uow:
            return [CandidateView.of(c) for c in await uow.candidates.list_by_run(run_id)]


class ListRunEventsUseCase:
    """Historical events, used to backfill a client that connects mid-run."""

    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(
        self, run_id: RunId, *, after_sequence: int | None = None, limit: int = 500
    ) -> Sequence[EventView]:
        async with self._uow_factory() as uow:
            stored = await uow.events.list_by_run(
                run_id, after_sequence=after_sequence, limit=limit
            )
            return [
                EventView(
                    sequence=sequence,
                    name=event.name,
                    occurred_at=event.occurred_at,
                    payload=dict(event.payload()),
                )
                for sequence, event in stored
            ]


def terminal_failure_kind(run: Run) -> FailureKind | None:
    """Convenience for the API layer's problem details."""
    return run.failure_kind
