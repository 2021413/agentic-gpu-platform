"""Git inspection tools, and the controlled application of a patch.

Unlike ``run_command``, these never execute anything a model chose: the binary
is git, the arguments are built here, repository hooks are disabled by
``GitCommandRunner``, and the only model-supplied value is a path that has been
proven to live inside the workspace. They therefore run through the git runner
rather than the sandbox — the sandbox's job is untrusted *code*, not trusted
commands.
"""

from __future__ import annotations

from collections.abc import Mapping

from domain.exceptions import ToolExecutionError
from domain.value_objects.tools import ToolInvocation, ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle
from infrastructure.tools.base import (
    Timer,
    bool_argument,
    json_schema,
    local_result,
    resolve_in_workspace,
    string_argument,
)
from infrastructure.workspace.git_cli import GitCommandRunner

__all__ = ["ApplyPatchTool", "GitDiffTool", "GitStatusTool"]


class GitDiffTool:
    """Show what the workspace changed."""

    name = "git_diff"
    kind = ToolKind.VCS
    description = (
        "Show the workspace's uncommitted changes as a unified diff, including "
        "newly created files. Optionally restricted to one path."
    )

    __slots__ = ("_git",)

    def __init__(self, git: GitCommandRunner | None = None) -> None:
        self._git = git or GitCommandRunner()

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "path": {
                    "type": "string",
                    "description": "Restrict the diff to this path (workspace-relative).",
                },
                "staged_only": {"type": "boolean", "default": False},
            }
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        timer = Timer()
        arguments = invocation.arguments
        staged_only = bool_argument(self.name, arguments, "staged_only", default=False)
        args = ["diff", "--no-color", "--no-ext-diff"]
        if staged_only:
            args.append("--cached")
        else:
            # Intent-to-add makes untracked files show up; without it a candidate
            # that only added files would appear to have changed nothing.
            await self._git.run("add", "--intent-to-add", "--all", cwd=workspace.path, check=False)
        path = arguments.get("path")
        if isinstance(path, str) and path:
            resolve_in_workspace(self.name, workspace, path)
            args.extend(["--", path])

        result = await self._git.run(*args, cwd=workspace.path, check=False)
        return local_result(
            tool=self.name,
            kind=self.kind,
            command="git " + " ".join(args),
            exit_code=result.exit_code,
            timer=timer,
            stdout=result.stdout,
            stderr=result.stderr,
        )


class GitStatusTool:
    """Show which files the workspace has touched."""

    name = "git_status"
    kind = ToolKind.VCS
    description = "List modified, added, deleted and untracked files in the workspace."

    __slots__ = ("_git",)

    def __init__(self, git: GitCommandRunner | None = None) -> None:
        self._git = git or GitCommandRunner()

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema({})

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        timer = Timer()
        result = await self._git.run(
            "status", "--porcelain=v1", "--branch", cwd=workspace.path, check=False
        )
        return local_result(
            tool=self.name,
            kind=self.kind,
            command="git status --porcelain=v1 --branch",
            exit_code=result.exit_code,
            timer=timer,
            stdout=result.stdout,
            stderr=result.stderr,
            metadata={"dirty_files": max(len(result.stdout.strip().splitlines()) - 1, 0)},
        )


class ApplyPatchTool:
    """Apply a unified diff, all of it or none of it.

    ``git apply --check`` runs first: a patch that does not apply must leave the
    workspace untouched, because a half-applied patch gives the repair loop a
    state that neither the model nor a human can reason about. A rejected patch
    is a *result* here (exit code 1), not an exception — the agent is expected to
    read the error and produce a better patch.
    """

    name = "apply_patch"
    kind = ToolKind.PATCH
    description = (
        "Apply a unified diff to the workspace. The patch is validated first and "
        "is applied atomically: on conflict nothing changes and the error is returned."
    )

    __slots__ = ("_git",)

    def __init__(self, git: GitCommandRunner | None = None) -> None:
        self._git = git or GitCommandRunner()

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "patch": {
                    "type": "string",
                    "description": "Unified diff, as produced by 'git diff'.",
                }
            },
            required=["patch"],
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        timer = Timer()
        if not workspace.is_writable:
            raise ToolExecutionError(
                self.name,
                "workspace is read-only",
                workspace_id=str(workspace.id),
                role=workspace.role.value,
            )
        diff = string_argument(self.name, invocation.arguments, "patch")
        if not diff.strip():
            return local_result(
                tool=self.name,
                kind=self.kind,
                command="git apply",
                exit_code=1,
                timer=timer,
                stderr="patch is empty",
            )
        if not diff.endswith("\n"):
            diff += "\n"

        check = await self._git.run(
            "apply",
            "--check",
            "--whitespace=nowarn",
            "-",
            cwd=workspace.path,
            stdin=diff,
            check=False,
        )
        if not check.succeeded:
            return local_result(
                tool=self.name,
                kind=self.kind,
                command="git apply --check",
                exit_code=check.exit_code,
                timer=timer,
                stderr=check.stderr,
                metadata={"applied": False},
            )

        applied = await self._git.run(
            "apply", "--whitespace=nowarn", "-", cwd=workspace.path, stdin=diff, check=False
        )
        return local_result(
            tool=self.name,
            kind=self.kind,
            command="git apply",
            exit_code=applied.exit_code,
            timer=timer,
            stdout=applied.stdout,
            stderr=applied.stderr,
            metadata={"applied": applied.succeeded},
        )
