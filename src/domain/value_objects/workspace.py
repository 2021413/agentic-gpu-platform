"""Isolated workspaces (spec section 8).

Two coders must never mutate the same path. A workspace is a handle to an
isolated checkout — a git worktree, a clone, or a snapshot — owned by exactly
one candidate for its lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

from domain.value_objects.identifiers import CandidateId, RunId, WorkspaceId

__all__ = ["WorkspaceHandle", "WorkspaceKind", "WorkspaceRole"]


@unique
class WorkspaceKind(StrEnum):
    GIT_WORKTREE = "GIT_WORKTREE"
    CLONE = "CLONE"
    SNAPSHOT = "SNAPSHOT"
    READONLY = "READONLY"


@unique
class WorkspaceRole(StrEnum):
    """What the workspace is for; read-only roles must never be written to."""

    PLANNER = "PLANNER"
    CANDIDATE = "CANDIDATE"
    REVIEWER = "REVIEWER"
    MERGE = "MERGE"

    @property
    def is_writable(self) -> bool:
        return self in (WorkspaceRole.CANDIDATE, WorkspaceRole.MERGE)


@dataclass(frozen=True, slots=True)
class WorkspaceHandle:
    """A leased, isolated filesystem location.

    ``path`` is meaningful only to the infrastructure that created it; the
    domain treats it as opaque and never touches the filesystem.
    """

    id: WorkspaceId
    run_id: RunId
    role: WorkspaceRole
    kind: WorkspaceKind
    path: str
    base_revision: str | None = None
    branch: str | None = None
    candidate_id: CandidateId | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("workspace path must not be empty")
        if self.role is WorkspaceRole.CANDIDATE and self.candidate_id is None:
            raise ValueError("a candidate workspace must name its candidate")

    @property
    def is_writable(self) -> bool:
        return self.role.is_writable
