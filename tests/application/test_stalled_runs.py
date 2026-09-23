"""A run whose work is finished must not sit there (spec section 11).

Two runs died the same way in production. One never left CREATED, because
nothing in the deployed system called `start`. The other reached REVIEWING with
every job SUCCEEDED, a validated candidate and a PASS verdict — and stopped, on
a real stack, with no error anywhere:

    statut : REVIEWING | aucun échec
    PLAN|SUCCEEDED CODE|SUCCEEDED BUILD|SUCCEEDED TEST|SUCCEEDED REVIEW|SUCCEEDED

`_advance` is called once, right after a job finishes. Miss that one call — a
crash, a restart, a lost race against the job's own status write — and nothing
ever calls it again. The sweep exists so that a missed call costs a tick rather
than the whole run.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.application.conftest import Platform
from tests.application.test_workflow import create_run

from domain.enums import RunStatus
from domain.value_objects.identifiers import RunId


def swallow_the_advance(platform: Platform, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce the failure exactly: the job succeeds, nothing follows it."""

    async def never_advances(run_id: RunId) -> None:
        return None

    monkeypatch.setattr(platform.orchestrator, "_advance", never_advances)


async def test_a_run_left_mid_flight_is_picked_up_again(
    platform: Platform, project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=1)
    await platform.orchestrator.start(view.id)

    swallow_the_advance(platform, monkeypatch)
    await platform.drain()
    monkeypatch.undo()

    run = platform.store.runs.items[view.id]
    assert not run.is_terminal, "the test needs a run that actually stalled"

    swept = await platform.orchestrator.advance_stalled_runs()
    await platform.drain()

    assert view.id in swept
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED


async def test_the_sweep_leaves_a_run_that_is_still_working_alone(
    platform: Platform, project: Any
) -> None:
    """Idempotence is the whole safety of a sweep: it must never restart work
    that is under way, or two workers get handed the same candidate."""
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=1)
    await platform.orchestrator.start(view.id)

    before = list(platform.queue.enqueued)
    swept = await platform.orchestrator.advance_stalled_runs()

    assert view.id not in swept
    assert platform.queue.enqueued == before


async def test_a_finished_run_is_never_swept(platform: Platform, project: Any) -> None:
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=1)
    await platform.orchestrator.start(view.id)
    await platform.drain()
    assert platform.store.runs.items[view.id].status is RunStatus.COMPLETED

    assert await platform.orchestrator.advance_stalled_runs() == []
