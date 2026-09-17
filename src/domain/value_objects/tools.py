"""Deterministic tool execution (spec section 7).

Tools are not LLMs. Their results are the *only* admissible evidence that a
build or a test suite passed: a model claiming "tests passed" establishes
nothing. ``ToolResult`` is therefore a first-class domain value, persisted and
carried into review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, unique
from types import MappingProxyType
from typing import Any

__all__ = [
    "ExecutionLimits",
    "ToolInvocation",
    "ToolKind",
    "ToolResult",
]

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@unique
class ToolKind(StrEnum):
    """Coarse classification driving which failures feed the repair loop."""

    SEARCH = "SEARCH"
    READ = "READ"
    EDIT = "EDIT"
    PATCH = "PATCH"
    COMMAND = "COMMAND"
    BUILD = "BUILD"
    TEST = "TEST"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    VCS = "VCS"


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """Sandbox bounds (spec section 31). Absent limits are not acceptable."""

    timeout_seconds: float = 300.0
    max_output_bytes: int = 1_000_000
    memory_mb: int | None = 4096
    cpu_count: float | None = 2.0
    network_enabled: bool = False

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A request to run one tool, addressed to a workspace."""

    tool: str
    kind: ToolKind
    arguments: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)

    def __post_init__(self) -> None:
        if not self.tool:
            raise ValueError("tool name must not be empty")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured outcome of one tool execution.

    A non-zero ``exit_code`` is a legitimate result, not an exception: a failing
    build is information the agentic loop consumes.
    """

    tool: str
    kind: ToolKind
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    truncated: bool = False
    artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    def tail(self, max_chars: int = 4000) -> str:
        """The end of the combined output — where compilers put what matters."""
        combined = "\n".join(part for part in (self.stdout, self.stderr) if part)
        if len(combined) <= max_chars:
            return combined
        return "...\n" + combined[-max_chars:]


def summarize(results: Sequence[ToolResult]) -> str:
    """One compact line per result, for prompts and events."""
    return "\n".join(
        f"[{r.kind}] {r.tool}: exit={r.exit_code} ({r.duration_ms}ms)" for r in results
    )
