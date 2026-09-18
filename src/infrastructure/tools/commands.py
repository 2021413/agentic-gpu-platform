"""Command execution tools: free-form, and the project's own build/test/analysis.

Nothing here ever reaches a shell. Commands are argument vectors executed
directly, so a model cannot smuggle ``; rm -rf /`` or a pipe through a string —
and, equally important, a command that appears to work in a terminal but relies
on shell syntax fails immediately and visibly instead of half-working.

Build, test and analysis commands come from the project's ``ToolchainConfig``.
When a project has not configured one, the corresponding tool is simply not
built: an agent must never see a ``run_tests`` tool that would run a guessed
command, because its failure would be reported as a defect in the code.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import replace

from domain.entities.project import ToolchainConfig
from domain.exceptions import ToolExecutionError
from domain.ports.tools import SandboxExecutor, Tool
from domain.value_objects.tools import ExecutionLimits, ToolInvocation, ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle
from infrastructure.tools.base import (
    int_argument,
    json_schema,
    resolve_in_workspace,
    string_list_argument,
)

__all__ = ["RunCommandTool", "ToolchainCommandTool", "build_toolchain_tools"]


class RunCommandTool:
    """Run an arbitrary command inside the sandbox.

    The most dangerous tool in the layer, hence the two guards: an optional
    executable allow-list, and the impossibility of raising the timeout a
    caller configured (a model may ask for less, never for more).
    """

    name = "run_command"
    kind = ToolKind.COMMAND
    description = (
        "Run a command in the workspace sandbox. Provide the argument vector "
        "(e.g. ['python', '-m', 'pytest', '-x']); no shell is used, so pipes, "
        "redirections and ';' are not available."
    )

    __slots__ = ("_allowed", "_sandbox")

    def __init__(
        self, *, sandbox: SandboxExecutor, allowed_executables: Sequence[str] | None = None
    ) -> None:
        self._sandbox = sandbox
        self._allowed = frozenset(allowed_executables) if allowed_executables else None

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Argument vector; the first item is the executable.",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Directory to run in, relative to the workspace root.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Lower the timeout for this call; it can never be raised.",
                },
            },
            required=["command"],
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        argv = string_list_argument(self.name, invocation.arguments, "command")
        if not argv:
            raise ToolExecutionError(self.name, "command must not be empty")
        if self._allowed is not None and argv[0] not in self._allowed:
            raise ToolExecutionError(
                self.name,
                "executable is not allowed",
                executable=argv[0],
                allowed=sorted(self._allowed),
            )

        target = _working_directory(self.name, workspace, invocation.arguments)
        limits = _narrowed_limits(self.name, invocation)
        result = await self._sandbox.run(
            command=argv, workspace=target, limits=limits, environment=None
        )
        return replace(result, tool=self.name, kind=self.kind)


class ToolchainCommandTool:
    """One command the project declared: build, tests or static analysis.

    Extra arguments from the model are appended rather than interpolated, so
    ``run_tests`` can be pointed at a single test without the model being able to
    rewrite the configured command.
    """

    __slots__ = (
        "_command",
        "_description",
        "_environment",
        "_kind",
        "_name",
        "_sandbox",
        "_subdirectory",
    )

    def __init__(
        self,
        *,
        name: str,
        kind: ToolKind,
        description: str,
        command: str,
        sandbox: SandboxExecutor,
        working_subdirectory: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        argv = shlex.split(command)
        if not argv:
            raise ValueError(f"tool {name!r} was configured with an empty command")
        self._name = name
        self._kind = kind
        self._description = description
        self._command = tuple(argv)
        self._sandbox = sandbox
        self._subdirectory = working_subdirectory
        self._environment = dict(environment or {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> ToolKind:
        return self._kind

    @property
    def description(self) -> str:
        return f"{self._description} Runs: {shlex.join(self._command)}"

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "extra_arguments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments appended to the configured command.",
                },
                "timeout_seconds": {"type": "integer", "minimum": 1},
            }
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        extra = string_list_argument(self._name, invocation.arguments, "extra_arguments")
        target = workspace
        if self._subdirectory:
            # The sandbox port takes a workspace, not a directory; a derived
            # handle keeps the contract intact while running one level down.
            path = resolve_in_workspace(self._name, workspace, self._subdirectory)
            target = replace(workspace, path=str(path))

        limits = _narrowed_limits(self._name, invocation)
        result = await self._sandbox.run(
            command=(*self._command, *extra),
            workspace=target,
            limits=limits,
            environment=self._environment or None,
        )
        return replace(result, tool=self._name, kind=self._kind)


def build_toolchain_tools(
    *, toolchain: ToolchainConfig, sandbox: SandboxExecutor
) -> tuple[Tool, ...]:
    """Build one tool per command the project actually declared."""
    specifications = (
        ("build", ToolKind.BUILD, toolchain.build_command, "Compile the project."),
        ("run_tests", ToolKind.TEST, toolchain.test_command, "Run the project's test suite."),
        (
            "static_analysis",
            ToolKind.STATIC_ANALYSIS,
            toolchain.static_analysis_command,
            "Run the project's static analysis.",
        ),
        (
            "install_dependencies",
            ToolKind.BUILD,
            toolchain.install_command,
            "Prepare the project (dependency install, CMake configure).",
        ),
    )
    return tuple(
        ToolchainCommandTool(
            name=name,
            kind=kind,
            description=description,
            command=command,
            sandbox=sandbox,
            working_subdirectory=toolchain.working_subdirectory,
            environment=toolchain.environment,
        )
        for name, kind, command, description in specifications
        if command
    )


def _working_directory(
    tool: str, workspace: WorkspaceHandle, arguments: Mapping[str, object]
) -> WorkspaceHandle:
    requested = arguments.get("working_directory")
    if not isinstance(requested, str) or not requested:
        return workspace
    path = resolve_in_workspace(tool, workspace, requested)
    return replace(workspace, path=str(path))


def _narrowed_limits(tool: str, invocation: ToolInvocation) -> ExecutionLimits:
    """Let a call shorten its own timeout, never extend it."""
    requested = invocation.arguments.get("timeout_seconds")
    if requested is None:
        return invocation.limits
    seconds = int_argument(tool, invocation.arguments, "timeout_seconds", default=0, minimum=1)
    return replace(
        invocation.limits, timeout_seconds=min(float(seconds), invocation.limits.timeout_seconds)
    )
