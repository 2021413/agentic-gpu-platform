"""Shared plumbing for deterministic tools.

Two conventions are enforced here, because getting them wrong is how a tool
layer becomes dangerous or useless:

* **Malformed arguments raise, failures return.** A missing ``path`` argument
  means the tool could not run at all — ``ToolExecutionError``. A file that does
  not exist, a pattern that matches nothing, a compilation that fails: those are
  ``ToolResult`` values with a non-zero exit code, because the agentic loop is
  supposed to read them and react.
* **Every path is resolved inside the workspace.** Model-authored arguments are
  untrusted input; ``../../etc/passwd`` is refused rather than followed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from domain.exceptions import ToolExecutionError
from domain.value_objects.tools import ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle

__all__ = [
    "Timer",
    "bool_argument",
    "int_argument",
    "json_schema",
    "local_result",
    "resolve_in_workspace",
    "string_argument",
    "string_list_argument",
]


def json_schema(
    properties: Mapping[str, Mapping[str, object]], required: Sequence[str] = ()
) -> Mapping[str, object]:
    """A JSON schema for tool calling.

    ``additionalProperties`` is false on purpose: a model that invents an
    argument should be corrected by the schema, not silently ignored.
    """
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


class Timer:
    """Wall-clock duration of an in-process tool, in milliseconds."""

    __slots__ = ("_started",)

    def __init__(self) -> None:
        self._started = time.monotonic()

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)


def string_argument(
    tool: str, arguments: Mapping[str, Any], name: str, *, default: str | None = None
) -> str:
    value = arguments.get(name, default)
    if value is None:
        raise ToolExecutionError(tool, f"missing required argument {name!r}")
    if not isinstance(value, str):
        raise ToolExecutionError(
            tool, f"argument {name!r} must be a string", got=type(value).__name__
        )
    return value


def int_argument(
    tool: str, arguments: Mapping[str, Any], name: str, *, default: int, minimum: int = 0
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolExecutionError(
            tool, f"argument {name!r} must be an integer", got=type(value).__name__
        )
    if value < minimum:
        raise ToolExecutionError(tool, f"argument {name!r} must be >= {minimum}", got=value)
    return value


def bool_argument(
    tool: str, arguments: Mapping[str, Any], name: str, *, default: bool = False
) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ToolExecutionError(
            tool, f"argument {name!r} must be a boolean", got=type(value).__name__
        )
    return value


def string_list_argument(
    tool: str, arguments: Mapping[str, Any], name: str, *, default: Sequence[str] = ()
) -> tuple[str, ...]:
    value = arguments.get(name, default)
    if isinstance(value, str):
        raise ToolExecutionError(tool, f"argument {name!r} must be a list of strings, not a string")
    if not isinstance(value, Sequence):
        raise ToolExecutionError(
            tool, f"argument {name!r} must be a list of strings", got=type(value).__name__
        )
    items = tuple(value)
    if any(not isinstance(item, str) for item in items):
        raise ToolExecutionError(tool, f"argument {name!r} must contain only strings")
    return items


def resolve_in_workspace(tool: str, workspace: WorkspaceHandle, relative: str) -> Path:
    """Map a model-provided path to a real one, or refuse it.

    Symlinks are resolved before the containment check, so a symlink planted
    inside the workspace cannot be used as a door out of it.
    """
    root = Path(workspace.path).resolve()
    candidate = (root / relative).resolve() if not Path(relative).is_absolute() else Path(relative)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolExecutionError(
            tool,
            "path escapes the workspace",
            path=relative,
            workspace_id=str(workspace.id),
        ) from exc
    return candidate


def local_result(
    *,
    tool: str,
    kind: ToolKind,
    command: str,
    exit_code: int,
    timer: Timer,
    stdout: str = "",
    stderr: str = "",
    truncated: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ToolResult:
    """Build the result of a tool that ran in-process rather than in a sandbox."""
    return ToolResult(
        tool=tool,
        kind=kind,
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=timer.elapsed_ms,
        truncated=truncated,
        metadata=dict(metadata or {}),
    )
