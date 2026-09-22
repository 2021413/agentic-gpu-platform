"""What the coder is told when it is asked to try again (spec section 11).

Three real runs against a real model ended the same way:

    no candidate survived deterministic validation; repair budget exhausted
    the reviewer rejected every candidate; repair budget exhausted

The repair loop ran its three rounds and converged on nothing, because the
coder was never told what went wrong. `tool_output` was a prompt variable that
nothing ever filled, and `repair_brief` only exists once a reviewer has spoken
— so a candidate whose tests failed before any review went back to the coder
with the same prompt it had the first time, and produced the same code.

The reviewer, which cannot change a line, received the full command output.
The coder, which is the only thing that can fix it, received none.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.application.conftest import Platform
from tests.application.test_workflow import create_run

from domain.enums import RunStatus


class _Recorder:
    """Delegates to the real coder and keeps what it was asked.

    Wraps the agent rather than patching a method on it: CoderAgent is a frozen
    slotted dataclass, and a test that has to defeat the design to observe it
    is usually observing the wrong thing.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    async def code(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return await self.inner.code(**kwargs)


def spy_on_coder(platform: Platform, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    recorder = _Recorder(platform.orchestrator._coder)
    monkeypatch.setattr(platform.orchestrator, "_coder", recorder)
    return recorder.calls


@pytest.mark.tool_exit_codes({"run_tests": 1})
async def test_a_failing_test_suite_reaches_the_coder(
    platform: Platform, project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect, stated plainly: the coder must be told what failed."""
    await platform.add_worker()
    calls = spy_on_coder(platform, monkeypatch)
    view = await create_run(platform, project, candidate_count=1)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    assert len(calls) > 1, "no repair round happened; this test needs one"
    repair = calls[-1]
    evidence = str(repair.get("tool_output") or "")
    assert evidence, "the coder was asked to repair with no idea what failed"
    assert "exit=1" in evidence or "exit code 1" in evidence, evidence


@pytest.mark.tool_exit_codes({"run_tests": 1})
async def test_the_first_attempt_carries_no_evidence(
    platform: Platform, project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing has run yet, so there is nothing honest to show."""
    await platform.add_worker()
    calls = spy_on_coder(platform, monkeypatch)
    view = await create_run(platform, project, candidate_count=1)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    assert not (calls[0].get("tool_output") or "")


class _RequestRecorder:
    """Delegates to the real provider and keeps the requests it was given."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.requests: list[Any] = []

    async def build(self, *, workspace: Any, request: Any) -> Any:
        self.requests.append(request)
        return await self.inner.build(workspace=workspace, request=request)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@pytest.mark.tool_exit_codes({"run_tests": 1})
async def test_the_coder_is_shown_the_files_it_already_changed(
    platform: Platform, project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a repair is asked for without the code being in the prompt.

    Context selection searches for terms taken from the objective; whether any
    of them happens to match the file the candidate wrote is luck. Asserted on
    the request rather than on what came back, because the wiring is what was
    missing and the provider here is a double that answers the same thing
    either way — which is how this went unnoticed the first time.
    """
    await platform.add_worker()
    recorder = _RequestRecorder(platform.orchestrator._context)
    monkeypatch.setattr(platform.orchestrator, "_context", recorder)
    view = await create_run(platform, project, candidate_count=1)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    with_paths = [r for r in recorder.requests if r.paths]
    assert with_paths, "no request ever named the files the candidate had changed"
    assert "parser.py" in with_paths[-1].paths, with_paths[-1].paths


@pytest.mark.tool_exit_codes({"run_tests": 1})
async def test_the_run_still_ends_when_the_evidence_does_not_help(
    platform: Platform, project: Any
) -> None:
    """Better information must not turn a bounded loop into an endless one."""
    await platform.add_worker()
    view = await create_run(platform, project, candidate_count=1)

    await platform.orchestrator.start(view.id)
    await platform.drain()

    assert platform.store.runs.items[view.id].status is RunStatus.FAILED
