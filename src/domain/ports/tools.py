"""Deterministic tooling and sandboxing ports (spec sections 7 and 31).

Agent tool execution is treated as hostile: it never runs on the orchestrator
host, always under explicit limits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from domain.value_objects.tools import ExecutionLimits, ToolInvocation, ToolResult
from domain.value_objects.workspace import WorkspaceHandle

__all__ = ["SandboxExecutor", "Tool", "ToolExecutor", "ToolRegistry"]


@runtime_checkable
class SandboxExecutor(Protocol):
    """Runs a command under isolation. The only way commands ever execute."""

    async def run(
        self,
        *,
        command: Sequence[str],
        workspace: WorkspaceHandle,
        limits: ExecutionLimits,
        environment: Mapping[str, str] | None = None,
    ) -> ToolResult:
        """Execute and always return a structured result.

        A non-zero exit code is a result, not an exception; only an inability to
        run at all raises ``ToolExecutionError``.
        """
        ...


@runtime_checkable
class Tool(Protocol):
    """One deterministic capability exposed to agents."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        """JSON schema advertised to the model for tool calling."""
        ...

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult: ...


@runtime_checkable
class ToolRegistry(Protocol):
    """The set of tools a given role is allowed to use."""

    def get(self, name: str) -> Tool | None: ...
    def names(self) -> Sequence[str]: ...
    def all(self) -> Sequence[Tool]: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Executes tool invocations against a workspace, enforcing the allow-list."""

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult: ...

    async def execute_many(
        self, *, invocations: Sequence[ToolInvocation], workspace: WorkspaceHandle
    ) -> Sequence[ToolResult]:
        """Run several tools, preserving order. Stops at the first hard failure."""
        ...
