"""Reading and editing files inside a workspace.

These two tools do their work in-process rather than through the sandbox. That
is a deliberate exception to "everything runs sandboxed": the sandbox exists to
contain *command execution* — arbitrary code chosen by a model. Reading and
writing one path that has already been proven to live inside the workspace is a
bounded operation with no interpreter behind it, and routing it through ``cat``
or ``sed`` would add a shell-quoting surface without adding any safety.

Blocking file access is pushed to a worker thread so that one slow filesystem
cannot stall every other candidate sharing the event loop.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from domain.exceptions import ToolExecutionError
from domain.value_objects.tools import ToolInvocation, ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle
from infrastructure.tools.base import (
    Timer,
    bool_argument,
    int_argument,
    json_schema,
    local_result,
    resolve_in_workspace,
    string_argument,
)

__all__ = ["EditFileTool", "ReadFileTool"]

_FAILURE = 1
_MISSING = -1
"""Sentinel for "the file is not there", distinct from "zero occurrences"."""
_EDIT_MODES = ("replace", "create", "append")


class ReadFileTool:
    """Read a bounded slice of one file."""

    name = "read_file"
    kind = ToolKind.READ
    description = (
        "Read a file from the workspace, optionally a line range. Returns the raw "
        "text without line-number prefixes, so it can be quoted verbatim in a patch."
    )

    __slots__ = ("_default_max_bytes",)

    def __init__(self, *, default_max_bytes: int = 200_000) -> None:
        self._default_max_bytes = default_max_bytes

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "start_line": {"type": "integer", "default": 1, "minimum": 1},
                "max_lines": {"type": "integer", "default": 0, "minimum": 0},
                "max_bytes": {"type": "integer", "default": 0, "minimum": 0},
            },
            required=["path"],
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        timer = Timer()
        arguments = invocation.arguments
        relative = string_argument(self.name, arguments, "path")
        target = resolve_in_workspace(self.name, workspace, relative)
        start_line = int_argument(self.name, arguments, "start_line", default=1, minimum=1)
        max_lines = int_argument(self.name, arguments, "max_lines", default=0)
        max_bytes = int_argument(self.name, arguments, "max_bytes", default=0)
        limit = max_bytes or self._default_max_bytes

        command = f"read_file {relative}"
        try:
            raw = await asyncio.to_thread(_read_bytes, target, limit + 1)
        except FileNotFoundError:
            return local_result(
                tool=self.name,
                kind=self.kind,
                command=command,
                exit_code=_FAILURE,
                timer=timer,
                stderr=f"{relative}: no such file",
            )
        except IsADirectoryError:
            return local_result(
                tool=self.name,
                kind=self.kind,
                command=command,
                exit_code=_FAILURE,
                timer=timer,
                stderr=f"{relative}: is a directory",
            )
        except OSError as exc:
            raise ToolExecutionError(self.name, "file could not be read", reason=str(exc)) from exc

        if b"\x00" in raw[:8192]:
            return local_result(
                tool=self.name,
                kind=self.kind,
                command=command,
                exit_code=_FAILURE,
                timer=timer,
                stderr=f"{relative}: binary file, refusing to read",
            )

        truncated = len(raw) > limit
        text = raw[:limit].decode("utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[start_line - 1 :]
        if max_lines:
            truncated = truncated or len(selected) > max_lines
            selected = selected[:max_lines]
        end_line = start_line + len(selected) - 1

        return local_result(
            tool=self.name,
            kind=self.kind,
            command=command,
            exit_code=0,
            timer=timer,
            stdout="\n".join(selected),
            truncated=truncated,
            metadata={
                "path": relative,
                "start_line": start_line,
                "end_line": max(end_line, start_line - 1),
                "total_lines": len(lines),
            },
        )


class EditFileTool:
    """Create, overwrite, append to, or substitute inside a file.

    ``replace`` requires the old text to occur exactly once. An ambiguous edit is
    refused rather than applied to the first match: guessing which of three
    identical blocks the model meant is precisely the kind of silent corruption
    a deterministic tool layer exists to prevent.
    """

    name = "edit_file"
    kind = ToolKind.EDIT
    description = (
        "Edit a file in the workspace. mode='replace' substitutes a unique snippet, "
        "mode='create' writes a new file, mode='append' adds to the end."
    )

    __slots__ = ()

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "mode": {"type": "string", "enum": list(_EDIT_MODES), "default": "replace"},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace; must occur exactly once (mode=replace).",
                },
                "new_string": {"type": "string", "description": "Replacement text (mode=replace)."},
                "content": {"type": "string", "description": "File content (mode=create/append)."},
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "Allow mode=create to replace an existing file.",
                },
            },
            required=["path"],
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        timer = Timer()
        if not workspace.is_writable:
            # Defence in depth: the role allow-list should already have stopped
            # this, so reaching here means an orchestration bug worth shouting about.
            raise ToolExecutionError(
                self.name,
                "workspace is read-only",
                workspace_id=str(workspace.id),
                role=workspace.role.value,
            )

        arguments = invocation.arguments
        relative = string_argument(self.name, arguments, "path")
        target = resolve_in_workspace(self.name, workspace, relative)
        mode = string_argument(self.name, arguments, "mode", default="replace")
        if mode not in _EDIT_MODES:
            raise ToolExecutionError(self.name, "unknown edit mode", mode=mode)
        command = f"edit_file --mode {mode} {relative}"

        try:
            if mode == "replace":
                return await self._replace(arguments, target, relative, command, timer)
            return await self._write(arguments, target, relative, command, timer, mode=mode)
        except OSError as exc:
            raise ToolExecutionError(
                self.name, "file could not be written", path=relative, reason=str(exc)
            ) from exc

    async def _replace(
        self,
        arguments: Mapping[str, object],
        target: Path,
        relative: str,
        command: str,
        timer: Timer,
    ) -> ToolResult:
        old = string_argument(self.name, arguments, "old_string")
        new = string_argument(self.name, arguments, "new_string", default="")
        if not old:
            raise ToolExecutionError(self.name, "old_string must not be empty")

        # Read, count and write happen in one thread call: checking first and
        # writing later would leave a window in which the file changed.
        occurrences, written = await asyncio.to_thread(_substitute, target, old, new)
        if occurrences == _MISSING:
            return self._failed(command, timer, f"{relative}: no such file")
        if occurrences != 1:
            return self._failed(
                command,
                timer,
                f"{relative}: old_string occurs {occurrences} times, expected exactly 1",
                metadata={"occurrences": occurrences},
            )
        return local_result(
            tool=self.name,
            kind=self.kind,
            command=command,
            exit_code=0,
            timer=timer,
            stdout=f"{relative}: replaced 1 occurrence",
            metadata={"path": relative, "bytes_written": written},
        )

    async def _write(
        self,
        arguments: Mapping[str, object],
        target: Path,
        relative: str,
        command: str,
        timer: Timer,
        *,
        mode: str,
    ) -> ToolResult:
        content = string_argument(self.name, arguments, "content")
        if mode == "create":
            overwrite = bool_argument(self.name, arguments, "overwrite", default=False)
            written = await asyncio.to_thread(_create, target, content, overwrite)
            if written is None:
                return self._failed(
                    command, timer, f"{relative}: already exists (pass overwrite=true)"
                )
        else:
            written = await asyncio.to_thread(_append, target, content)

        return local_result(
            tool=self.name,
            kind=self.kind,
            command=command,
            exit_code=0,
            timer=timer,
            stdout=f"{relative}: wrote {written} bytes",
            metadata={"path": relative, "bytes_written": written, "mode": mode},
        )

    def _failed(
        self,
        command: str,
        timer: Timer,
        message: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> ToolResult:
        return local_result(
            tool=self.name,
            kind=self.kind,
            command=command,
            exit_code=_FAILURE,
            timer=timer,
            stderr=message,
            metadata=metadata,
        )


def _read_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _substitute(path: Path, old: str, new: str) -> tuple[int, int]:
    """Replace a unique occurrence; returns (occurrences, bytes written)."""
    try:
        original = path.read_text("utf-8", "replace")
    except (FileNotFoundError, IsADirectoryError):
        return _MISSING, 0
    occurrences = original.count(old)
    if occurrences != 1:
        return occurrences, 0
    updated = original.replace(old, new, 1)
    _atomic_write(path, updated)
    return 1, len(updated.encode())


def _create(path: Path, content: str, overwrite: bool) -> int | None:
    """Write a new file; ``None`` means it already existed and was left alone."""
    if path.exists() and not overwrite:
        return None
    path.parent.mkdir(0o755, parents=True, exist_ok=True)
    _atomic_write(path, content)
    return len(content.encode())


def _append(path: Path, content: str) -> int:
    existing = path.read_text("utf-8", "replace") if path.is_file() else ""
    merged = existing + content
    _atomic_write(path, merged)
    return len(merged.encode())


def _atomic_write(path: Path, content: str) -> None:
    """Write through a temporary file, so an interrupted edit cannot truncate."""
    directory = path.parent
    handle, temporary = tempfile.mkstemp(dir=str(directory), prefix=".agentic-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
