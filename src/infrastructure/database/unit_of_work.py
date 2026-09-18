"""The transactional boundary (spec sections 13 and 15).

The whole point of this class is one guarantee: **state and events are written
by the same transaction**. Aggregates buffer their events instead of publishing
them; ``commit`` drains those buffers, appends them to the event store through
the *same* session, and only then commits. Either both land or neither does.

Publication on the ``EventBus`` happens afterwards, from
``collected_events``, and is therefore at-least-once: a crash between the commit
and the publish loses the notification, not the fact. The stored log is what
lets a subscriber catch up, which is why the event store is the source of truth
and the bus is only a delivery mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain.entities.base import Entity
from domain.events.base import DomainEvent
from domain.ports.repositories import (
    CandidateRepository,
    EventStore,
    JobRepository,
    PlanRepository,
    ProjectRepository,
    ReviewRepository,
    RunRepository,
    ToolResultRepository,
)
from infrastructure.database.repositories import (
    SqlAlchemyCandidateRepository,
    SqlAlchemyEventStore,
    SqlAlchemyJobRepository,
    SqlAlchemyPlanRepository,
    SqlAlchemyProjectRepository,
    SqlAlchemyReviewRepository,
    SqlAlchemyRunRepository,
    SqlAlchemyToolResultRepository,
    SqlAlchemyWorkerRepository,
)

__all__ = ["SqlAlchemyUnitOfWork"]


class SqlAlchemyUnitOfWork:
    """Implements ``domain.ports.repositories.UnitOfWork`` over one session.

    One ``async with`` block is one transaction: entering starts it, leaving
    rolls back whatever was not committed. Re-entering a live unit is refused
    rather than silently nesting, because a nested "transaction" that commits
    the outer one is the kind of bug that only shows up in production.
    """

    projects: ProjectRepository
    runs: RunRepository
    jobs: JobRepository
    plans: PlanRepository
    candidates: CandidateRepository
    reviews: ReviewRepository
    tool_results: ToolResultRepository
    events: EventStore
    workers: SqlAlchemyWorkerRepository

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        # The session and its repositories are built here rather than on entry
        # so the object satisfies the ``UnitOfWork`` protocol from the moment it
        # exists — a container injecting it must not have to enter it first. A
        # SQLAlchemy session opens no connection until it is actually used, so
        # this costs nothing.
        session = session_factory()
        self._session = session
        self._entered = False
        self._pending: list[Entity] = []
        self._collected_events: tuple[DomainEvent, ...] = ()
        self.projects = SqlAlchemyProjectRepository(session)
        self.runs = SqlAlchemyRunRepository(session)
        self.jobs = SqlAlchemyJobRepository(session)
        self.plans = SqlAlchemyPlanRepository(session)
        self.candidates = SqlAlchemyCandidateRepository(session)
        self.reviews = SqlAlchemyReviewRepository(session)
        self.tool_results = SqlAlchemyToolResultRepository(session)
        self.workers = SqlAlchemyWorkerRepository(session)
        self.events = SqlAlchemyEventStore(session)

    # -- context management ---------------------------------------------
    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        if self._entered:
            raise RuntimeError("this unit of work is already open")
        self._entered = True
        self._pending = []
        self._collected_events = ()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Roll back unconditionally, then release the connection.

        Rolling back after a successful commit is a no-op, so leaving without
        committing can only ever discard work — never publish half of it.
        """
        if not self._entered:
            return
        try:
            await self._session.rollback()
        finally:
            # Closing returns the connection to the pool and releases every
            # identity-mapped row; the session itself stays reusable, so the
            # same unit can serve a second transaction.
            await self._session.close()
            self._entered = False
            self._pending = []

    # -- transaction ----------------------------------------------------
    def collect(self, *entities: object) -> None:
        """Register aggregates whose buffered events must be drained on commit.

        Collecting the same aggregate twice is harmless: ``pull_events`` drains,
        so the second visit finds an empty buffer.
        """
        for entity in entities:
            if not isinstance(entity, Entity):
                raise TypeError(
                    f"only aggregates carrying domain events can be collected, "
                    f"got {type(entity).__name__}"
                )
            self._pending.append(entity)

    async def commit(self) -> None:
        """Append the collected events and commit, in that order.

        If the commit fails, the events have already been drained from their
        aggregates but nothing was stored: those in-memory objects are stale and
        the caller must reload them rather than retry with the same instances.
        """
        session = self._require_session()
        drained = self._drain()
        if drained:
            await self.events.append(drained)
        await session.commit()
        self._collected_events = drained

    async def rollback(self) -> None:
        await self._require_session().rollback()
        self._collected_events = ()

    @property
    def collected_events(self) -> Sequence[DomainEvent]:
        """Events drained by the last successful commit, in order.

        The caller publishes these on the event bus *after* the transaction, so
        a rollback can never leave subscribers believing in something that was
        never stored.
        """
        return self._collected_events

    @property
    def session(self) -> AsyncSession:
        """Escape hatch for adapters needing raw SQL inside this transaction."""
        return self._require_session()

    # -- internals ------------------------------------------------------
    def _drain(self) -> tuple[DomainEvent, ...]:
        drained: list[DomainEvent] = []
        for entity in self._pending:
            drained.extend(entity.pull_events())
        self._pending = []
        return tuple(drained)

    def _require_session(self) -> AsyncSession:
        """Transactional operations are only meaningful inside the context."""
        if not self._entered:
            raise RuntimeError("the unit of work must be entered before use")
        return self._session
