"""Role allow-lists: what each agent may call, and what it may never call."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from domain.entities.project import ToolchainConfig
from domain.enums import AgentRole
from domain.exceptions import ToolExecutionError
from domain.ports.repository_context import RepositoryContextProvider
from domain.ports.tools import SandboxExecutor, Tool, ToolExecutor, ToolRegistry
from domain.value_objects.identifiers import CandidateId, RunId, WorkspaceId
from domain.value_objects.tools import ToolInvocation, ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle, WorkspaceKind, WorkspaceRole
from infrastructure.tools.files import EditFileTool
from infrastructure.tools.registry import (
    ROLE_TOOLS,
    AllowListToolExecutor,
    StaticToolRegistry,
    build_tool_registry,
)
from infrastructure.tools.repository_context import RipgrepRepositoryContextProvider
from infrastructure.tools.sandbox import SubprocessSandboxExecutor

MUTATING_TOOL_NAMES = frozenset({"edit_file", "apply_patch", "run_command"})


@pytest.fixture
def sandbox() -> SubprocessSandboxExecutor:
    return SubprocessSandboxExecutor()


def make_workspace(path: Path, role: WorkspaceRole) -> WorkspaceHandle:
    path.mkdir(parents=True, exist_ok=True)
    return WorkspaceHandle(
        id=WorkspaceId.generate(),
        run_id=RunId.generate(),
        role=role,
        kind=WorkspaceKind.GIT_WORKTREE,
        path=str(path),
        candidate_id=CandidateId.generate() if role is WorkspaceRole.CANDIDATE else None,
    )


class RecordingTool:
    """A tool that only remembers it was called."""

    name = "recording"
    kind = ToolKind.READ
    description = "test double"

    def __init__(self, exit_code: int = 0) -> None:
        self.calls = 0
        self._exit_code = exit_code

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return {"type": "object", "properties": {}}

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        self.calls += 1
        return ToolResult(
            tool=self.name, kind=self.kind, command="recording", exit_code=self._exit_code
        )


def test_the_planner_is_given_no_way_to_modify_files(
    sandbox: SubprocessSandboxExecutor,
) -> None:
    registry = build_tool_registry(role=AgentRole.PLANNER, sandbox=sandbox)
    assert MUTATING_TOOL_NAMES.isdisjoint(registry.names())
    assert "search_repository" in registry.names()
    assert all(tool.kind not in (ToolKind.EDIT, ToolKind.PATCH) for tool in registry.all())


def test_the_reviewer_may_verify_but_not_edit(sandbox: SubprocessSandboxExecutor) -> None:
    toolchain = ToolchainConfig(test_command="/bin/true", build_command="/bin/true")
    registry = build_tool_registry(role=AgentRole.REVIEWER, sandbox=sandbox, toolchain=toolchain)
    assert "run_tests" in registry.names()
    assert "build" in registry.names()
    assert MUTATING_TOOL_NAMES.isdisjoint(registry.names())


def test_the_coder_gets_the_full_set(sandbox: SubprocessSandboxExecutor) -> None:
    registry = build_tool_registry(role=AgentRole.CODER, sandbox=sandbox)
    assert MUTATING_TOOL_NAMES.issubset(registry.names())


def test_toolchain_tools_exist_only_when_the_project_declared_them(
    sandbox: SubprocessSandboxExecutor,
) -> None:
    """A guessed build command would report its own failure as a code defect."""
    without = build_tool_registry(role=AgentRole.CODER, sandbox=sandbox)
    assert "build" not in without.names()
    assert "run_tests" not in without.names()

    with_commands = build_tool_registry(
        role=AgentRole.CODER,
        sandbox=sandbox,
        toolchain=ToolchainConfig(build_command="make", test_command="ctest"),
    )
    assert {"build", "run_tests"}.issubset(with_commands.names())
    assert "static_analysis" not in with_commands.names()


def test_a_read_only_role_cannot_be_built_with_a_mutating_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allow-list itself is checked, so a careless edit fails at startup."""
    monkeypatch.setitem(ROLE_TOOLS, AgentRole.PLANNER, frozenset({"edit_file"}))
    with pytest.raises(ValueError, match="read-only"):
        StaticToolRegistry.for_role(AgentRole.PLANNER, [EditFileTool()])


def test_duplicate_tool_names_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        StaticToolRegistry([RecordingTool(), RecordingTool()])


async def test_a_tool_outside_the_role_allow_list_is_refused(
    sandbox: SubprocessSandboxExecutor, tmp_path: Path
) -> None:
    registry = build_tool_registry(role=AgentRole.PLANNER, sandbox=sandbox)
    executor = AllowListToolExecutor(registry=registry, role=AgentRole.PLANNER)
    workspace = make_workspace(tmp_path / "planner", WorkspaceRole.PLANNER)

    with pytest.raises(ToolExecutionError) as error:
        await executor.execute(
            invocation=ToolInvocation(
                tool="edit_file",
                kind=ToolKind.EDIT,
                arguments={"path": "a.txt", "mode": "create", "content": "x"},
            ),
            workspace=workspace,
        )

    assert "not allowed" in error.value.message
    assert error.value.details["role"] == AgentRole.PLANNER.value
    assert not (Path(workspace.path) / "a.txt").exists()


async def test_a_mutating_tool_is_refused_on_a_read_only_workspace(
    sandbox: SubprocessSandboxExecutor, tmp_path: Path
) -> None:
    """Second line of defence: the workspace, not just the role, says no."""
    registry = build_tool_registry(role=AgentRole.CODER, sandbox=sandbox)
    executor = AllowListToolExecutor(registry=registry, role=AgentRole.CODER)
    reviewer_workspace = make_workspace(tmp_path / "reviewer", WorkspaceRole.REVIEWER)

    with pytest.raises(ToolExecutionError, match="read-only workspace"):
        await executor.execute(
            invocation=ToolInvocation(
                tool="edit_file",
                kind=ToolKind.EDIT,
                arguments={"path": "a.txt", "mode": "create", "content": "x"},
            ),
            workspace=reviewer_workspace,
        )


async def test_execute_many_preserves_order_and_survives_failing_exit_codes(
    tmp_path: Path,
) -> None:
    failing = RecordingTool(exit_code=1)
    failing.name = "failing"  # type: ignore[misc]
    registry = StaticToolRegistry([RecordingTool(), failing])
    executor = AllowListToolExecutor(registry=registry)
    workspace = make_workspace(tmp_path / "coder", WorkspaceRole.CANDIDATE)

    results = await executor.execute_many(
        invocations=[
            ToolInvocation(tool="failing", kind=ToolKind.READ),
            ToolInvocation(tool="recording", kind=ToolKind.READ),
        ],
        workspace=workspace,
    )

    assert [result.tool for result in results] == ["failing", "recording"]
    assert [result.exit_code for result in results] == [1, 0]


async def test_execute_many_stops_when_a_tool_cannot_run(tmp_path: Path) -> None:
    registry = StaticToolRegistry([RecordingTool()])
    executor = AllowListToolExecutor(registry=registry)
    workspace = make_workspace(tmp_path / "coder", WorkspaceRole.CANDIDATE)

    with pytest.raises(ToolExecutionError) as error:
        await executor.execute_many(
            invocations=[
                ToolInvocation(tool="recording", kind=ToolKind.READ),
                ToolInvocation(tool="unknown", kind=ToolKind.READ),
                ToolInvocation(tool="recording", kind=ToolKind.READ),
            ],
            workspace=workspace,
        )

    assert error.value.details["completed"] == 1


def test_every_tool_advertises_a_usable_json_schema(
    sandbox: SubprocessSandboxExecutor,
) -> None:
    registry = build_tool_registry(
        role=AgentRole.CODER,
        sandbox=sandbox,
        toolchain=ToolchainConfig(build_command="make", test_command="ctest"),
    )
    for tool in registry.all():
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        assert tool.description
        assert tool.name


def test_the_adapters_satisfy_the_domain_ports(sandbox: SubprocessSandboxExecutor) -> None:
    """Structural conformance, checked once so a signature drift is caught here."""
    registry = build_tool_registry(role=AgentRole.CODER, sandbox=sandbox)

    assert isinstance(sandbox, SandboxExecutor)
    assert isinstance(registry, ToolRegistry)
    assert isinstance(AllowListToolExecutor(registry=registry), ToolExecutor)
    assert isinstance(RipgrepRepositoryContextProvider(sandbox=sandbox), RepositoryContextProvider)
    assert all(isinstance(tool, Tool) for tool in registry.all())
