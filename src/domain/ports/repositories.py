"""Persistence ports (spec section 13).

The domain states *what* must be storable and retrievable. Whether that is
PostgreSQL, and how, is infrastructure's business. Nothing here mentions
SQLAlchemy, sessions or SQL.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Protocol, runtime_checkable

from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan
from domain.entities.project import Project
from domain.entities.review import Review
from domain.entities.run import Run
from domain.enums import JobStatus, RunStatus
from domain.events.base import DomainEvent
from domain.value_objects.identifiers import (
    CandidateId,
    IdempotencyKey,
    JobId,
    PlanId,
    ProjectId,
    RunId,
)
from domain.value_objects.tools import ToolResult

__all__ = [
    "CandidateRepository",
    "EventStore",
    "JobRepository",
    "PlanRepository",
    "ProjectRepository",
    "ReviewRepository",
    "RunRepository",
    "ToolResultRepository",
    "UnitOfWork",
]


@runtime_checkable
class ProjectRepository(Protocol):
    async def add(self, project: Project) -> None: ...
    async def get(self, project_id: ProjectId) -> Project | None: ...
    async def get_by_name(self, name: str) -> Project | None: ...
    async def list_all(self, *, limit: int = 100, offset: int = 0) -> Sequence[Project]: ...


@runtime_checkable
class RunRepository(Protocol):
    async def add(self, run: Run) -> None: ...
    async def get(self, run_id: RunId) -> Run | None: ...
    async def update(self, run: Run) -> None: ...

    async def find_by_idempotency_key(self, key: IdempotencyKey) -> Run | None:
        """Replay support: the same key must return the original run."""
        ...

    async def list_by_project(
        self, project_id: ProjectId, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Run]: ...

    async def list_active(self) -> Sequence[Run]:
        """Non-terminal runs — what an orchestrator resumes after a restart."""
        ...

    async def count_by_status(self) -> dict[RunStatus, int]: ...


@runtime_checkable
class JobRepository(Protocol):
    async def add(self, job: Job) -> None: ...
    async def get(self, job_id: JobId) -> Job | None: ...
    async def update(self, job: Job) -> None: ...

    async def find_by_idempotency_key(self, key: IdempotencyKey) -> Job | None: ...

    async def list_by_run(self, run_id: RunId) -> Sequence[Job]: ...

    async def list_by_status(self, status: JobStatus, *, limit: int = 100) -> Sequence[Job]: ...

    async def list_expired_leases(self, *, now: datetime, limit: int = 100) -> Sequence[Job]:
        """Durable counterpart of the queue's reclaim, used on orchestrator restart."""
        ...


@runtime_checkable
class PlanRepository(Protocol):
    async def add(self, plan: Plan) -> None: ...
    async def get(self, plan_id: PlanId) -> Plan | None: ...
    async def latest_for_run(self, run_id: RunId) -> Plan | None: ...
    async def list_by_run(self, run_id: RunId) -> Sequence[Plan]: ...


@runtime_checkable
class CandidateRepository(Protocol):
    async def add(self, candidate: Candidate) -> None: ...
    async def get(self, candidate_id: CandidateId) -> Candidate | None: ...
    async def update(self, candidate: Candidate) -> None: ...
    async def list_by_run(self, run_id: RunId) -> Sequence[Candidate]: ...


@runtime_checkable
class ReviewRepository(Protocol):
    async def add(self, review: Review) -> None: ...
    async def list_by_candidate(self, candidate_id: CandidateId) -> Sequence[Review]: ...
    async def latest_for_candidate(self, candidate_id: CandidateId) -> Review | None: ...


@runtime_checkable
class ToolResultRepository(Protocol):
    """Deterministic evidence is persisted: it is what proves a run's outcome."""

    async def add_many(
        self, *, run_id: RunId, candidate_id: CandidateId | None, results: Sequence[ToolResult]
    ) -> None: ...

    async def list_by_candidate(self, candidate_id: CandidateId) -> Sequence[ToolResult]: ...


@runtime_checkable
class EventStore(Protocol):
    """Durable audit trail. Also backfills SSE subscribers that join late."""

    async def append(self, events: Sequence[DomainEvent]) -> None: ...

    async def list_by_run(
        self, run_id: RunId, *, after_sequence: int | None = None, limit: int = 500
    ) -> Sequence[tuple[int, DomainEvent]]:
        """Events with their monotonic sequence number, oldest first."""
        ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Transactional boundary grouping repository writes and event emission.

    Entering yields a unit with live repositories; leaving without ``commit``
    rolls everything back, including the events, so persisted state and
    published events can never disagree.
    """

    projects: ProjectRepository
    runs: RunRepository
    jobs: JobRepository
    plans: PlanRepository
    candidates: CandidateRepository
    reviews: ReviewRepository
    tool_results: ToolResultRepository
    events: EventStore

    async def __aenter__(self) -> UnitOfWork: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leaving without ``commit`` rolls back, events included."""
        ...

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...

    def collect(self, *entities: object) -> None:
        """Register aggregates whose buffered events must be drained on commit."""
        ...

    @property
    def collected_events(self) -> Sequence[DomainEvent]:
        """Events drained by the last successful commit.

        Publication happens *after* the transaction, so a rollback can never
        leave subscribers believing in something that was never stored.
        """
        ...
