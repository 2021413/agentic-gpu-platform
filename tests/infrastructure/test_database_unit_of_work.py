"""The unit of work's central promise: state and events, or neither.

Every test here attacks the same seam — the moment where an aggregate's
buffered events become rows. If those two writes could ever be split, a run
could be persisted as COMPLETED while no ``run.completed`` event exists, and no
subscriber would ever learn about it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain.entities.project import Project
from domain.entities.run import Run
from domain.enums import RunStatus
from domain.ports.repositories import UnitOfWork
from infrastructure.database.models import RunEventModel
from infrastructure.database.unit_of_work import SqlAlchemyUnitOfWork

pytestmark = pytest.mark.integration

Factory = async_sessionmaker[AsyncSession]


def unit(session_factory: Factory) -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(session_factory)


async def count_events(session_factory: Factory) -> int:
    async with unit(session_factory) as uow:
        total = await uow.session.scalar(select(func.count()).select_from(RunEventModel))
    return int(total or 0)


def test_the_adapter_satisfies_the_port(session_factory: Factory) -> None:
    """A structural check: the container will inject it as a ``UnitOfWork``."""
    uow: UnitOfWork = SqlAlchemyUnitOfWork(session_factory)
    assert isinstance(uow, UnitOfWork)


async def test_commit_writes_state_and_drained_events_together(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    run = new_run()
    run.start_planning(now)

    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.runs.add(run)
        uow.collect(run)
        await uow.commit()
        published = list(uow.collected_events)

    # Draining is what the aggregate hands over; nothing is left behind to be
    # emitted twice by a later commit.
    assert run.pending_events == ()
    assert [type(event).name for event in published] == [
        "run.created",
        "run.state_changed",
        "run.plan_requested",
    ]

    async with unit(session_factory) as uow:
        stored_run = await uow.runs.get(run.id)
        stored_events = await uow.events.list_by_run(run.id)

    assert stored_run is not None
    assert stored_run.status is RunStatus.PLANNING
    assert [event for _, event in stored_events] == published


async def test_leaving_without_committing_stores_neither_state_nor_events(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    async with unit(session_factory) as seed:
        await seed.projects.add(project)
        await seed.commit()

    run = new_run()
    run.start_planning(now)
    async with unit(session_factory) as uow:
        await uow.runs.add(run)
        uow.collect(run)
        # No commit: leaving the block must undo everything.

    async with unit(session_factory) as uow:
        assert await uow.runs.get(run.id) is None
        assert list(await uow.events.list_by_run(run.id)) == []
    assert await count_events(session_factory) == 0


async def test_an_explicit_rollback_discards_the_whole_transaction(
    session_factory: Factory, project: Project, new_run: Callable[..., Run]
) -> None:
    async with unit(session_factory) as seed:
        await seed.projects.add(project)
        await seed.commit()

    run = new_run()
    async with unit(session_factory) as uow:
        await uow.runs.add(run)
        uow.collect(run)
        await uow.rollback()
        assert list(uow.collected_events) == []

    async with unit(session_factory) as uow:
        assert await uow.runs.get(run.id) is None


async def test_a_failing_commit_publishes_nothing_and_stores_nothing(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    """The events are written by the *same* transaction, so they die with it."""
    first = new_run(idempotency_key="POST /runs#dup")
    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.runs.add(first)
        uow.collect(first)
        await uow.commit()

    events_before = await count_events(session_factory)

    duplicate = new_run(idempotency_key="POST /runs#dup")
    duplicate.start_planning(now)
    failed = unit(session_factory)
    with pytest.raises(IntegrityError):
        async with failed as uow:
            await uow.runs.add(duplicate)
            uow.collect(duplicate)
            await uow.commit()

    assert list(failed.collected_events) == []
    async with unit(session_factory) as uow:
        assert await uow.runs.get(duplicate.id) is None
        assert list(await uow.events.list_by_run(duplicate.id)) == []
    assert await count_events(session_factory) == events_before


async def test_collect_accepts_only_aggregates(session_factory: Factory) -> None:
    async with unit(session_factory) as uow:
        with pytest.raises(TypeError):
            uow.collect("not an aggregate")


async def test_collecting_several_aggregates_drains_them_in_order(
    session_factory: Factory,
    project: Project,
    new_run: Callable[..., Run],
    new_candidate: Callable[..., object],
    now: datetime,
) -> None:
    run = new_run()
    candidate = new_candidate(run=run)
    candidate.start_coding(now=now)

    async with unit(session_factory) as uow:
        await uow.projects.add(project)
        await uow.runs.add(run)
        await uow.candidates.add(candidate)
        uow.collect(run, candidate)
        await uow.commit()
        published = [type(event).name for event in uow.collected_events]

    assert published == ["run.created", "candidate.started"]

    async with unit(session_factory) as uow:
        sequences = [sequence for sequence, _ in await uow.events.list_by_run(run.id)]

    assert sequences == [1, 2]


async def test_a_unit_of_work_can_be_reused_for_successive_transactions(
    session_factory: Factory, project: Project, new_run: Callable[..., Run], now: datetime
) -> None:
    uow = unit(session_factory)
    run = new_run()

    async with uow:
        await uow.projects.add(project)
        await uow.runs.add(run)
        uow.collect(run)
        await uow.commit()

    run.start_planning(now)
    async with uow:
        await uow.runs.update(run)
        uow.collect(run)
        await uow.commit()
        second_batch = [type(event).name for event in uow.collected_events]

    assert second_batch == ["run.state_changed", "run.plan_requested"]

    async with uow:
        stored = await uow.runs.get(run.id)
        events = await uow.events.list_by_run(run.id)

    assert stored is not None
    assert stored.status is RunStatus.PLANNING
    assert [sequence for sequence, _ in events] == [1, 2, 3]


async def test_using_a_unit_of_work_outside_its_context_is_refused(
    session_factory: Factory,
) -> None:
    uow = unit(session_factory)
    with pytest.raises(RuntimeError):
        await uow.commit()
