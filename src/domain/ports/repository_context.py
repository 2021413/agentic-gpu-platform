"""Code context selection (spec section 28).

Never send a repository blindly to a model. This port selects the slice that is
actually relevant; v1 does it with listing, filtering and ripgrep, and later
implementations may add tree-sitter, language servers or embeddings without any
change upstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from domain.value_objects.workspace import WorkspaceHandle

__all__ = ["ContextRequest", "FileExcerpt", "RepositoryContext", "RepositoryContextProvider"]


@dataclass(frozen=True, slots=True)
class FileExcerpt:
    """A bounded slice of a file, with the line numbers it came from."""

    path: str
    content: str
    start_line: int = 1
    end_line: int | None = None
    reason: str = ""

    @property
    def line_count(self) -> int:
        return self.content.count("\n") + 1


@dataclass(frozen=True, slots=True)
class RepositoryContext:
    """What the agent is allowed to see, plus how it was chosen."""

    excerpts: tuple[FileExcerpt, ...] = ()
    file_tree: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    estimated_tokens: int = 0

    def render(self) -> str:
        """Prompt-ready rendering; deliberately explicit about truncation."""
        blocks = [f"### {e.path} (from line {e.start_line})\n{e.content}" for e in self.excerpts]
        if self.file_tree:
            blocks.insert(0, "### Files\n" + "\n".join(self.file_tree))
        return "\n\n".join(blocks)


@dataclass(frozen=True, slots=True)
class ContextRequest:
    """What the caller needs context for, and how much it can afford."""

    objective: str
    queries: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    max_files: int = 40
    max_tokens: int = 24_000
    include_tree: bool = True
    exclude_globs: tuple[str, ...] = field(default=())


@runtime_checkable
class RepositoryContextProvider(Protocol):
    """Selects relevant code for a prompt."""

    async def build(
        self, *, workspace: WorkspaceHandle, request: ContextRequest
    ) -> RepositoryContext: ...

    async def search(
        self, *, workspace: WorkspaceHandle, pattern: str, limit: int = 100
    ) -> Sequence[FileExcerpt]: ...

    async def read_file(
        self, *, workspace: WorkspaceHandle, path: str, max_bytes: int = 200_000
    ) -> FileExcerpt | None: ...
