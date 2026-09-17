"""Patches produced by coder candidates.

A candidate never mutates the base repository: it produces a patch in its own
isolated workspace, which a controlled apply step merges once selected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["FileChange", "Patch"]

_DIFF_GIT_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+?)$", re.MULTILINE)
_NEW_FILE_RE = re.compile(r"^new file mode ", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class FileChange:
    """Per-file summary, used for review context and candidate comparison."""

    path: str
    added_lines: int = 0
    removed_lines: int = 0

    @property
    def churn(self) -> int:
        return self.added_lines + self.removed_lines


@dataclass(frozen=True, slots=True)
class Patch:
    """A unified diff plus its derived summary.

    ``diff`` is the authoritative content; the file list is derived from it so
    the two cannot disagree.
    """

    diff: str
    base_revision: str | None = None
    files: tuple[FileChange, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.diff.strip()

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)

    @property
    def total_churn(self) -> int:
        return sum(f.churn for f in self.files)

    @classmethod
    def from_unified_diff(cls, diff: str, *, base_revision: str | None = None) -> Patch:
        """Build a patch and derive its per-file statistics from the diff text."""
        return cls(diff=diff, base_revision=base_revision, files=_parse_stats(diff))


def _parse_stats(diff: str) -> tuple[FileChange, ...]:
    """Count added/removed lines per file in a git-style unified diff."""
    changes: list[FileChange] = []
    current_path: str | None = None
    added = removed = 0

    def flush() -> None:
        nonlocal current_path, added, removed
        if current_path is not None:
            changes.append(FileChange(current_path, added, removed))
        current_path, added, removed = None, 0, 0

    for line in diff.splitlines():
        header = _DIFF_GIT_RE.match(line)
        if header is not None:
            flush()
            current_path = header.group("b")
            continue
        if current_path is None or line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    flush()
    return tuple(changes)
