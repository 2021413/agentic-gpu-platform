"""One unrecoverable run must not take the control plane with it.

Seen on a real stack, in a crash loop: an api container that would not boot,
restarting every 28 seconds, because a single run left over from before the
restart referred to a workspace the restart had destroyed.

    WorkspaceError: workspace no longer exists (workspace_id='6befe3bc-...')
    Application startup failed. Exiting.

The whole point of resuming on startup is resilience. A resume that can refuse
to start the process is worse than no resume at all: nothing is served, no run
progresses, and the cause is a single row.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.application.conftest import Platform
from tests.application.test_workflow import create_run

from domain.enums import RunStatus
from domain.exceptions import WorkspaceError
from domain.value_objects.identifiers import RunId


async def test_a_run_that_cannot_be_resumed_does_not_stop_the_others(
    platform: Platform, project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await platform.add_worker()
    doomed = await create_run(platform, project, candidate_count=1)
    healthy = await create_run(platform, project, candidate_count=1)

    real_advance = platform.orchestrator._advance

    async def explode_for_one(run_id: RunId) -> None:
        if run_id == doomed.id:
            raise WorkspaceError("workspace no longer exists", workspace_id="gone")
        await real_advance(run_id)

    monkeypatch.setattr(platform.orchestrator, "_advance", explode_for_one)

    resumed = await platform.orchestrator.resume_active_runs()

    assert healthy.id in resumed
    assert doomed.id not in resumed, "a run that could not be resumed must not be reported as one"


async def test_the_unrecoverable_run_is_failed_rather_than_left_active(
    platform: Platform, project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise it is retried on every boot, forever, and stays 'active' to
    anyone reading the API."""
    await platform.add_worker()
    doomed = await create_run(platform, project, candidate_count=1)

    async def always_explodes(run_id: RunId) -> None:
        raise WorkspaceError("workspace no longer exists", workspace_id="gone")

    monkeypatch.setattr(platform.orchestrator, "_advance", always_explodes)

    await platform.orchestrator.resume_active_runs()

    run = platform.store.runs.items[doomed.id]
    assert run.status is RunStatus.FAILED
    reason = (run.failure_reason or "").lower()
    assert "resume" in reason, reason
    # The wording must say what was actually attempted: an earlier version
    # blamed "a restart" for a failure during a periodic sweep.
    assert "workspace no longer exists" in reason


async def test_the_sweep_is_just_as_isolated(
    platform: Platform, project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It runs on a timer against the same rows, so it has the same exposure."""
    await platform.add_worker()
    await create_run(platform, project, candidate_count=1)

    async def always_explodes(run_id: RunId) -> None:
        raise WorkspaceError("workspace no longer exists", workspace_id="gone")

    monkeypatch.setattr(platform.orchestrator, "_advance", always_explodes)

    assert await platform.orchestrator.advance_stalled_runs() == []
