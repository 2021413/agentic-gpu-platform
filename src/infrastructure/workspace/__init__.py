"""Workspace isolation adapters (spec section 8).

Parallel candidates are only safe because each of them owns a filesystem
location nobody else may touch. Everything in this package exists to defend
that single invariant.
"""

from __future__ import annotations

from infrastructure.workspace.git_cli import GitCommandRunner, GitResult
from infrastructure.workspace.git_worktree import GitWorktreeWorkspaceManager

__all__ = [
    "GitCommandRunner",
    "GitResult",
    "GitWorktreeWorkspaceManager",
]
