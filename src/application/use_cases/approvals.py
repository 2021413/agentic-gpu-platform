"""Human approval of a reviewed patch (spec section 34, extension).

The orchestrator owns the decision because approving is not a read: it merges
into the project's repository, releases the workspaces and completes the run,
and rejecting schedules another coder job. The use case exists so the HTTP
layer talks to the application and never to the orchestrator directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from application.dto.views import RunView
from domain.entities.run import Run
from domain.value_objects.identifiers import RunId

__all__ = ["ApproveRunUseCase", "RunApprovalService"]


@runtime_checkable
class RunApprovalService(Protocol):
    """What approving needs, and nothing more.

    A narrow port rather than the orchestrator itself: approving is two
    operations, while the orchestrator is the whole workflow, and depending on
    all of it would force every caller that composes this use case to build
    a queue, a worker pool and a workspace manager first.
    """

    async def approve(self, run_id: RunId) -> Run: ...

    async def reject(self, run_id: RunId, *, reason: str) -> Run: ...


class ApproveRunUseCase:
    """Let a waiting run land, or send it back with a reason."""

    __slots__ = ("_orchestrator",)

    def __init__(self, *, orchestrator: RunApprovalService) -> None:
        self._orchestrator = orchestrator

    async def approve(self, run_id: RunId) -> RunView:
        return RunView.of(await self._orchestrator.approve(run_id))

    async def reject(self, run_id: RunId, *, reason: str) -> RunView:
        """``reason`` is not decoration: it is what the coder is given to work
        from on the next round, so an empty one would waste a repair."""
        return RunView.of(await self._orchestrator.reject(run_id, reason=reason))
