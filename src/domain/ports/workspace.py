"""Workspace isolation port (spec section 8)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.entities.project import Project
from domain.value_objects.identifiers import CandidateId, RunId, WorkspaceId
from domain.value_objects.patch import Patch
from domain.value_objects.workspace import WorkspaceHandle, WorkspaceRole

__all__ = ["WorkspaceManager"]


@runtime_checkable
class WorkspaceManager(Protocol):
    """Creates, isolates and releases per-candidate checkouts.

    Implementations must guarantee that two writable workspaces never share a
    filesystem path: that guarantee is what makes parallel candidates safe.
    """

    async def create(
        self,
        *,
        project: Project,
        run_id: RunId,
        role: WorkspaceRole,
        candidate_id: CandidateId | None = None,
        base_revision: str | None = None,
    ) -> WorkspaceHandle: ...

    async def get(self, workspace_id: WorkspaceId) -> WorkspaceHandle | None: ...

    async def diff(self, handle: WorkspaceHandle) -> Patch:
        """Uncommitted changes of the workspace, as a patch."""
        ...

    async def apply_patch(self, handle: WorkspaceHandle, patch: Patch) -> None:
        """Apply a patch to a writable workspace. Fails cleanly on conflict."""
        ...

    async def commit(self, handle: WorkspaceHandle, *, message: str) -> str:
        """Commit the workspace and return the resulting revision."""
        ...

    async def integrate(self, *, project: Project, handle: WorkspaceHandle, message: str) -> str:
        """Controlled merge of the accepted candidate back into the project.

        This is the only path by which agent work reaches the base repository.
        """
        ...

    async def release(self, workspace_id: WorkspaceId) -> None:
        """Destroy a workspace. Must be safe to call twice."""
        ...

    async def release_run(self, run_id: RunId) -> Sequence[WorkspaceId]:
        """Release every workspace of a run (cancellation, completion)."""
        ...
