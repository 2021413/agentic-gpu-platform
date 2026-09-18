"""PostgreSQL implementations of the persistence ports (spec section 13).

Every class here implements one protocol from ``domain.ports.repositories``
and nothing more. They share a single ``AsyncSession``, which is what allows a
unit of work to group aggregate writes and event appends into one transaction.

None of these methods commits: committing is the unit of work's decision, not a
repository's, otherwise a half-written run could survive a failed step.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan
from domain.entities.project import Project
from domain.entities.review import Review
from domain.entities.run import Run
from domain.entities.worker import Worker
from domain.enums import JobStatus, RunStatus
from domain.events.base import DomainEvent
from domain.exceptions import EntityNotFoundError
from domain.value_objects.identifiers import (
    CandidateId,
    IdempotencyKey,
    JobId,
    PlanId,
    ProjectId,
    RunId,
    WorkerId,
)
from domain.value_objects.tools import ToolResult
from infrastructure.database.event_codec import event_run_id, load_event
from infrastructure.database.mappers import (
    apply_candidate,
    apply_job,
    apply_run,
    apply_worker,
    candidate_to_domain,
    candidate_to_model,
    job_to_domain,
    job_to_model,
    plan_to_domain,
    plan_to_model,
    project_to_domain,
    project_to_model,
    review_to_domain,
    review_to_model,
    run_to_domain,
    run_to_model,
    tool_result_to_domain,
    tool_results_to_models,
    worker_to_domain,
    worker_to_model,
)
from infrastructure.database.models import (
    CandidateModel,
    JobModel,
    PlanModel,
    ProjectModel,
    ReviewModel,
    RunEventModel,
    RunModel,
    ToolResultModel,
    WorkerModel,
)

__all__ = [
    "SqlAlchemyCandidateRepository",
    "SqlAlchemyEventStore",
    "SqlAlchemyJobRepository",
    "SqlAlchemyPlanRepository",
    "SqlAlchemyProjectRepository",
    "SqlAlchemyReviewRepository",
    "SqlAlchemyRunRepository",
    "SqlAlchemyToolResultRepository",
    "SqlAlchemyWorkerRepository",
]

_TERMINAL_RUN_STATUSES = tuple(status.value for status in RunStatus if status.is_terminal)
_IN_FLIGHT_JOB_STATUSES = (JobStatus.LEASED.value, JobStatus.RUNNING.value)

# Namespace for the advisory locks that serialize event sequence allocation.
# Any constant works as long as nothing else in the platform reuses it.
_EVENT_SEQUENCE_LOCK_NAMESPACE = 1_381_191_749


class _Repository:
    """Holds the session shared by every repository of a unit of work."""

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _scalars(self, statement: Select[Any]) -> Sequence[Any]:
        result = await self._session.scalars(statement)
        return result.all()


class SqlAlchemyProjectRepository(_Repository):
    """Implements ``domain.ports.repositories.ProjectRepository``."""

    async def add(self, project: Project) -> None:
        self._session.add(project_to_model(project))

    async def get(self, project_id: ProjectId) -> Project | None:
        model = await self._session.get(ProjectModel, project_id.value)
        return project_to_domain(model) if model else None

    async def get_by_name(self, name: str) -> Project | None:
        model = await self._session.scalar(select(ProjectModel).where(ProjectModel.name == name))
        return project_to_domain(model) if model else None

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> Sequence[Project]:
        statement = (
            select(ProjectModel)
            .order_by(ProjectModel.created_at, ProjectModel.id)
            .limit(limit)
            .offset(offset)
        )
        return [project_to_domain(model) for model in await self._scalars(statement)]


class SqlAlchemyRunRepository(_Repository):
    """Implements ``domain.ports.repositories.RunRepository``."""

    async def add(self, run: Run) -> None:
        self._session.add(run_to_model(run))

    async def get(self, run_id: RunId) -> Run | None:
        model = await self._session.get(RunModel, run_id.value)
        return run_to_domain(model) if model else None

    async def update(self, run: Run) -> None:
        """Copy the aggregate onto the row loaded in this session.

        The row is re-read rather than merged so the optimistic version counter
        compares against what this transaction actually saw.
        """
        model = await self._session.get(RunModel, run.id.value)
        if model is None:
            raise EntityNotFoundError("Run", run.id)
        apply_run(model, run)

    async def find_by_idempotency_key(self, key: IdempotencyKey) -> Run | None:
        model = await self._session.scalar(
            select(RunModel).where(RunModel.idempotency_key == str(key))
        )
        return run_to_domain(model) if model else None

    async def list_by_project(
        self, project_id: ProjectId, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Run]:
        statement = (
            select(RunModel)
            .where(RunModel.project_id == project_id.value)
            .order_by(RunModel.created_at.desc(), RunModel.id)
            .limit(limit)
            .offset(offset)
        )
        return [run_to_domain(model) for model in await self._scalars(statement)]

    async def list_active(self) -> Sequence[Run]:
        statement = (
            select(RunModel)
            .where(RunModel.status.not_in(_TERMINAL_RUN_STATUSES))
            .order_by(RunModel.created_at)
        )
        return [run_to_domain(model) for model in await self._scalars(statement)]

    async def count_by_status(self) -> dict[RunStatus, int]:
        """Only the statuses actually present are returned; absent means zero."""
        rows = await self._session.execute(
            select(RunModel.status, func.count()).group_by(RunModel.status)
        )
        return {RunStatus(status): count for status, count in rows.all()}


class SqlAlchemyJobRepository(_Repository):
    """Implements ``domain.ports.repositories.JobRepository``."""

    async def add(self, job: Job) -> None:
        self._session.add(job_to_model(job))

    async def get(self, job_id: JobId) -> Job | None:
        model = await self._session.get(JobModel, job_id.value)
        return job_to_domain(model) if model else None

    async def update(self, job: Job) -> None:
        model = await self._session.get(JobModel, job.id.value)
        if model is None:
            raise EntityNotFoundError("Job", job.id)
        apply_job(model, job)

    async def find_by_idempotency_key(self, key: IdempotencyKey) -> Job | None:
        model = await self._session.scalar(
            select(JobModel).where(JobModel.idempotency_key == str(key))
        )
        return job_to_domain(model) if model else None

    async def list_by_run(self, run_id: RunId) -> Sequence[Job]:
        statement = (
            select(JobModel)
            .where(JobModel.run_id == run_id.value)
            .order_by(JobModel.created_at, JobModel.id)
        )
        return [job_to_domain(model) for model in await self._scalars(statement)]

    async def list_by_status(self, status: JobStatus, *, limit: int = 100) -> Sequence[Job]:
        statement = (
            select(JobModel)
            .where(JobModel.status == status.value)
            .order_by(JobModel.created_at, JobModel.id)
            .limit(limit)
        )
        return [job_to_domain(model) for model in await self._scalars(statement)]

    async def list_expired_leases(self, *, now: datetime, limit: int = 100) -> Sequence[Job]:
        """Jobs whose holder went silent, oldest expiry first.

        Restricted to in-flight statuses: a job that already failed or finished
        may still carry lease columns, and reclaiming it would resurrect work
        nobody is waiting for.
        """
        statement = (
            select(JobModel)
            .where(
                JobModel.lease_expires_at.is_not(None),
                JobModel.lease_expires_at <= now,
                JobModel.status.in_(_IN_FLIGHT_JOB_STATUSES),
            )
            .order_by(JobModel.lease_expires_at)
            .limit(limit)
        )
        return [job_to_domain(model) for model in await self._scalars(statement)]


class SqlAlchemyWorkerRepository(_Repository):
    """Durable worker registrations.

    No port declares this repository yet — the live registry is Redis-backed
    (spec section 14) — but the spec requires worker metadata to survive in
    PostgreSQL, so the table and its adapter exist here, ready to back a
    ``WorkerRepository`` port when one is introduced.
    """

    async def add(self, worker: Worker) -> None:
        self._session.add(worker_to_model(worker))

    async def get(self, worker_id: WorkerId) -> Worker | None:
        model = await self._session.get(WorkerModel, worker_id.value)
        return worker_to_domain(model) if model else None

    async def update(self, worker: Worker) -> None:
        model = await self._session.get(WorkerModel, worker.id.value)
        if model is None:
            raise EntityNotFoundError("Worker", worker.id)
        apply_worker(model, worker)

    async def list_all(self) -> Sequence[Worker]:
        statement = select(WorkerModel).order_by(WorkerModel.registered_at, WorkerModel.id)
        return [worker_to_domain(model) for model in await self._scalars(statement)]


class SqlAlchemyPlanRepository(_Repository):
    """Implements ``domain.ports.repositories.PlanRepository``.

    Plans are immutable revisions, so there is no ``update``: a revised plan is
    a new row with a higher ``revision``.
    """

    async def add(self, plan: Plan) -> None:
        self._session.add(plan_to_model(plan))

    async def get(self, plan_id: PlanId) -> Plan | None:
        model = await self._session.get(PlanModel, plan_id.value)
        return plan_to_domain(model) if model else None

    async def latest_for_run(self, run_id: RunId) -> Plan | None:
        model = await self._session.scalar(
            select(PlanModel)
            .where(PlanModel.run_id == run_id.value)
            .order_by(PlanModel.revision.desc())
            .limit(1)
        )
        return plan_to_domain(model) if model else None

    async def list_by_run(self, run_id: RunId) -> Sequence[Plan]:
        statement = (
            select(PlanModel).where(PlanModel.run_id == run_id.value).order_by(PlanModel.revision)
        )
        return [plan_to_domain(model) for model in await self._scalars(statement)]


class SqlAlchemyCandidateRepository(_Repository):
    """Implements ``domain.ports.repositories.CandidateRepository``."""

    async def add(self, candidate: Candidate) -> None:
        self._session.add(candidate_to_model(candidate))

    async def get(self, candidate_id: CandidateId) -> Candidate | None:
        model = await self._session.get(CandidateModel, candidate_id.value)
        return candidate_to_domain(model) if model else None

    async def update(self, candidate: Candidate) -> None:
        model = await self._session.get(CandidateModel, candidate.id.value)
        if model is None:
            raise EntityNotFoundError("Candidate", candidate.id)
        apply_candidate(model, candidate)

    async def list_by_run(self, run_id: RunId) -> Sequence[Candidate]:
        statement = (
            select(CandidateModel)
            .where(CandidateModel.run_id == run_id.value)
            .order_by(CandidateModel.candidate_index)
        )
        return [candidate_to_domain(model) for model in await self._scalars(statement)]


class SqlAlchemyReviewRepository(_Repository):
    """Implements ``domain.ports.repositories.ReviewRepository``."""

    async def add(self, review: Review) -> None:
        self._session.add(review_to_model(review))

    async def list_by_candidate(self, candidate_id: CandidateId) -> Sequence[Review]:
        statement = (
            select(ReviewModel)
            .where(ReviewModel.candidate_id == candidate_id.value)
            .order_by(ReviewModel.iteration)
        )
        return [review_to_domain(model) for model in await self._scalars(statement)]

    async def latest_for_candidate(self, candidate_id: CandidateId) -> Review | None:
        model = await self._session.scalar(
            select(ReviewModel)
            .where(ReviewModel.candidate_id == candidate_id.value)
            .order_by(ReviewModel.iteration.desc())
            .limit(1)
        )
        return review_to_domain(model) if model else None


class SqlAlchemyToolResultRepository(_Repository):
    """Implements ``domain.ports.repositories.ToolResultRepository``."""

    async def add_many(
        self, *, run_id: RunId, candidate_id: CandidateId | None, results: Sequence[ToolResult]
    ) -> None:
        """Append results, continuing the candidate's execution order."""
        if not results:
            return
        next_position = await self._next_position(run_id, candidate_id)
        for model in tool_results_to_models(
            results, run_id=run_id, candidate_id=candidate_id, first_position=next_position
        ):
            self._session.add(model)

    async def list_by_candidate(self, candidate_id: CandidateId) -> Sequence[ToolResult]:
        statement = (
            select(ToolResultModel)
            .where(ToolResultModel.candidate_id == candidate_id.value)
            .order_by(ToolResultModel.position)
        )
        return [tool_result_to_domain(model) for model in await self._scalars(statement)]

    async def _next_position(self, run_id: RunId, candidate_id: CandidateId | None) -> int:
        scope = (
            ToolResultModel.candidate_id == candidate_id.value
            if candidate_id is not None
            else ToolResultModel.candidate_id.is_(None)
        )
        highest = await self._session.scalar(
            select(func.max(ToolResultModel.position)).where(
                ToolResultModel.run_id == run_id.value, scope
            )
        )
        return 0 if highest is None else highest + 1


class SqlAlchemyEventStore(_Repository):
    """Implements ``domain.ports.repositories.EventStore``.

    The sequence is allocated per run with ``MAX(sequence) + 1`` under a
    transaction-scoped advisory lock. A bare ``MAX + 1`` would let two
    concurrent transactions pick the same number and one of them would die on
    the unique index; the lock turns that race into a short wait instead, and it
    is released by PostgreSQL when the transaction ends, whatever the outcome.

    A per-run PostgreSQL sequence object would be the alternative, but sequences
    are not transactional: a rolled-back append would leave a permanent hole,
    and SSE clients use "give me everything after N" as a cursor.
    """

    async def append(self, events: Sequence[DomainEvent]) -> None:
        if not events:
            return
        # Grouped so the lock is taken once per run, in a stable order, which
        # also removes any chance of two appends deadlocking each other.
        grouped: dict[UUID | None, list[DomainEvent]] = {}
        for event in events:
            run_id = event_run_id(event)
            grouped.setdefault(run_id.value if run_id else None, []).append(event)

        for run_uuid in sorted(grouped, key=lambda value: (value is not None, str(value))):
            await self._lock_sequence(run_uuid)
            sequence = await self._next_sequence(run_uuid)
            for event in grouped[run_uuid]:
                self._session.add(
                    RunEventModel(
                        id=event.event_id,
                        run_id=run_uuid,
                        sequence=sequence,
                        name=type(event).name,
                        occurred_at=event.occurred_at,
                        payload=dict(event.payload()),
                    )
                )
                sequence += 1

    async def list_by_run(
        self, run_id: RunId, *, after_sequence: int | None = None, limit: int = 500
    ) -> Sequence[tuple[int, DomainEvent]]:
        statement = select(RunEventModel).where(RunEventModel.run_id == run_id.value)
        if after_sequence is not None:
            statement = statement.where(RunEventModel.sequence > after_sequence)
        statement = statement.order_by(RunEventModel.sequence).limit(limit)
        rows = await self._scalars(statement)
        return [
            (
                row.sequence,
                load_event(
                    name=row.name,
                    payload=row.payload,
                    occurred_at=row.occurred_at,
                    event_id=row.id,
                ),
            )
            for row in rows
        ]

    async def _lock_sequence(self, run_uuid: UUID | None) -> None:
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(:namespace, :key)"),
            {"namespace": _EVENT_SEQUENCE_LOCK_NAMESPACE, "key": _lock_key(run_uuid)},
        )

    async def _next_sequence(self, run_uuid: UUID | None) -> int:
        scope = (
            RunEventModel.run_id == run_uuid
            if run_uuid is not None
            else RunEventModel.run_id.is_(None)
        )
        highest = await self._session.scalar(select(func.max(RunEventModel.sequence)).where(scope))
        # Sequences are 1-based: an empty run starts at 1, so ``after_sequence=0``
        # is a valid "give me everything" cursor for an SSE client.
        return 1 if highest is None else highest + 1


def _lock_key(run_uuid: UUID | None) -> int:
    """A signed 32-bit key derived from the run id, as advisory locks require.

    Collisions between two runs are harmless: they only make two appends wait
    for each other.
    """
    if run_uuid is None:
        return 0
    return int.from_bytes(run_uuid.bytes[:4], "big", signed=True)
