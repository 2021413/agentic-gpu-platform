"""Tool registry, per-role allow-lists and the executor that enforces them.

The spec states that the planner must never mutate project files. A prompt
asking it not to is not an implementation of that rule: a model that decides to
edit a file would succeed. Here the rule holds *by construction* — a planner
registry cannot be built containing a mutating tool, so the tool does not exist
as far as the planner's tool-calling loop is concerned, and an invocation of it
is refused before anything runs.

Three layers enforce the same thing, deliberately:

1. the role allow-list decides which tools are advertised to the model;
2. :func:`StaticToolRegistry.for_role` refuses to build a read-only registry
   that contains a mutating tool, so a future careless edit to the allow-list
   fails at composition time rather than in production;
3. :class:`AllowListToolExecutor` refuses any invocation naming a tool the
   registry does not hold, and any mutating tool aimed at a read-only workspace.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from domain.entities.project import ToolchainConfig
from domain.enums import AgentRole
from domain.exceptions import ToolExecutionError
from domain.ports.tools import SandboxExecutor, Tool, ToolRegistry
from domain.value_objects.tools import ToolInvocation, ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle
from infrastructure.tools.commands import RunCommandTool, build_toolchain_tools
from infrastructure.tools.files import EditFileTool, ReadFileTool
from infrastructure.tools.search import (
    SearchRepositoryTool,
    SearchSymbolTool,
    TextSearchBackend,
)
from infrastructure.tools.vcs import ApplyPatchTool, GitDiffTool, GitStatusTool

__all__ = [
    "MUTATING_KINDS",
    "READ_ONLY_ROLES",
    "ROLE_TOOLS",
    "AllowListToolExecutor",
    "StaticToolRegistry",
    "build_tool_registry",
]

MUTATING_KINDS: frozenset[ToolKind] = frozenset({ToolKind.EDIT, ToolKind.PATCH})
"""Kinds that change project source. ``COMMAND`` is excluded on purpose: it is
not allowed for read-only roles either, but by the allow-list, since a command
is dangerous for reasons beyond mutating files."""

READ_ONLY_ROLES: frozenset[AgentRole] = frozenset({AgentRole.PLANNER, AgentRole.REVIEWER})

ROLE_TOOLS: Mapping[AgentRole, frozenset[str]] = {
    # The planner reads the repository and produces a plan. Nothing it can call
    # writes a file or runs project code.
    AgentRole.PLANNER: frozenset(
        {"search_repository", "read_file", "search_symbol", "git_status", "git_diff"}
    ),
    AgentRole.CODER: frozenset(
        {
            "search_repository",
            "read_file",
            "search_symbol",
            "edit_file",
            "apply_patch",
            "run_command",
            "build",
            "run_tests",
            "static_analysis",
            "install_dependencies",
            "git_diff",
            "git_status",
        }
    ),
    # The reviewer may re-run the deterministic checks — that is the whole point
    # of "never trust a model saying the tests passed" — but never edits.
    AgentRole.REVIEWER: frozenset(
        {
            "search_repository",
            "read_file",
            "search_symbol",
            "git_diff",
            "git_status",
            "build",
            "run_tests",
            "static_analysis",
        }
    ),
}


class StaticToolRegistry:
    """An immutable set of tools, addressable by name."""

    __slots__ = ("_role", "_tools")

    def __init__(self, tools: Iterable[Tool], *, role: AgentRole | None = None) -> None:
        indexed: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in indexed:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            indexed[tool.name] = tool
        self._tools = indexed
        self._role = role

    @property
    def role(self) -> AgentRole | None:
        return self._role

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> Sequence[str]:
        return tuple(sorted(self._tools))

    def all(self) -> Sequence[Tool]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    @classmethod
    def for_role(cls, role: AgentRole, tools: Iterable[Tool]) -> StaticToolRegistry:
        """Keep only what this role is allowed to use, and prove it is safe.

        The second check is the important one: it makes an unsafe allow-list a
        startup failure instead of a security incident.
        """
        allowed = ROLE_TOOLS[role]
        selected = [tool for tool in tools if tool.name in allowed]
        if role in READ_ONLY_ROLES:
            offending = sorted(tool.name for tool in selected if _kind_of(tool) in MUTATING_KINDS)
            if offending:
                raise ValueError(
                    f"role {role.value} is read-only but was given mutating tools: "
                    f"{', '.join(offending)}"
                )
        return cls(selected, role=role)


class AllowListToolExecutor:
    """Executes invocations, refusing anything outside the registry."""

    __slots__ = ("_registry", "_role")

    def __init__(self, *, registry: ToolRegistry, role: AgentRole | None = None) -> None:
        self._registry = registry
        self._role = role

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        tool = self._registry.get(invocation.tool)
        if tool is None:
            raise ToolExecutionError(
                invocation.tool,
                "tool is not allowed for this role",
                role=self._role.value if self._role else None,
                allowed=list(self._registry.names()),
            )
        if _kind_of(tool) in MUTATING_KINDS and not workspace.is_writable:
            raise ToolExecutionError(
                invocation.tool,
                "mutating tool refused on a read-only workspace",
                workspace_id=str(workspace.id),
                role=workspace.role.value,
            )
        return await tool.execute(invocation=invocation, workspace=workspace)

    async def execute_many(
        self, *, invocations: Sequence[ToolInvocation], workspace: WorkspaceHandle
    ) -> Sequence[ToolResult]:
        """Run a batch in order, stopping at the first tool that could not run.

        A non-zero exit does not stop the batch — a failing build followed by a
        diff is a perfectly sensible sequence. An inability to run does stop it,
        and propagates: the batch's premise was wrong, and continuing would hand
        the agent a plausible-looking but incomplete set of results.
        """
        results: list[ToolResult] = []
        for invocation in invocations:
            try:
                results.append(await self.execute(invocation=invocation, workspace=workspace))
            except ToolExecutionError as exc:
                exc.details["completed"] = len(results)
                raise
        return tuple(results)


def build_tool_registry(
    *,
    role: AgentRole,
    sandbox: SandboxExecutor,
    toolchain: ToolchainConfig | None = None,
    extra_tools: Iterable[Tool] = (),
) -> StaticToolRegistry:
    """Assemble the tools one role is allowed to use.

    Build, test and analysis tools appear only when the project declared the
    corresponding command, so a role's advertised tool list always reflects what
    can actually be run against this project.
    """
    backend = TextSearchBackend(sandbox=sandbox)
    tools: list[Tool] = [
        SearchRepositoryTool(backend),
        SearchSymbolTool(backend),
        ReadFileTool(),
        EditFileTool(),
        ApplyPatchTool(),
        GitDiffTool(),
        GitStatusTool(),
        RunCommandTool(sandbox=sandbox),
    ]
    if toolchain is not None:
        tools.extend(build_toolchain_tools(toolchain=toolchain, sandbox=sandbox))
    tools.extend(extra_tools)
    return StaticToolRegistry.for_role(role, tools)


def _kind_of(tool: Tool) -> ToolKind | None:
    """Tools advertise their kind; the port does not require it, so stay lenient."""
    kind = getattr(tool, "kind", None)
    return kind if isinstance(kind, ToolKind) else None
