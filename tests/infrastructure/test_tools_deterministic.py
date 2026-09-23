"""The deterministic tools themselves, against a real workspace.

These are the tools whose results are the only admissible evidence that
something built or passed, so the tests care about two things: the result is
structured and truthful, and a failure is reported as a result rather than
thrown.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from domain.entities.project import ToolchainConfig
from domain.exceptions import ToolExecutionError
from domain.ports.repository_context import ContextRequest
from domain.value_objects.identifiers import CandidateId, RunId, WorkspaceId
from domain.value_objects.tools import ExecutionLimits, ToolInvocation, ToolKind
from domain.value_objects.workspace import WorkspaceHandle, WorkspaceKind, WorkspaceRole
from infrastructure.tools.commands import RunCommandTool, build_toolchain_tools
from infrastructure.tools.files import EditFileTool, ReadFileTool
from infrastructure.tools.repository_context import (
    CHARS_PER_TOKEN,
    RipgrepRepositoryContextProvider,
    estimate_tokens,
)
from infrastructure.tools.sandbox import SubprocessSandboxExecutor
from infrastructure.tools.search import SearchRepositoryTool, SearchSymbolTool, TextSearchBackend
from infrastructure.tools.toolchain import PROFILES, profile, with_sanitizer
from infrastructure.tools.vcs import ApplyPatchTool, GitDiffTool, GitStatusTool

LIMITS = ExecutionLimits(timeout_seconds=30.0, max_output_bytes=1_000_000)


def run_git(repository: Path, *args: str) -> None:
    environment = {
        **os.environ,
        "HOME": str(repository),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@invalid", *args],
        cwd=repository,
        env=environment,
        capture_output=True,
        check=True,
    )


def same_path(left: str | Path, right: str | Path) -> bool:
    """A blocking probe, kept out of the async test bodies."""
    return Path(left).samefile(right)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceHandle:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# demo\n\nA tiny project.\n", encoding="utf-8")
    (root / "src" / "app.py").write_text(
        "def compute_total(values):\n"
        "    return sum(values)\n"
        "\n"
        "\n"
        "class Ledger:\n"
        "    def add(self, value):\n"
        "        self.values.append(value)\n",
        encoding="utf-8",
    )
    (root / "src" / "util.c").write_text(
        "#include <stdio.h>\n\nint compute_total(int *v, int n) { return 0; }\n", encoding="utf-8"
    )
    run_git(root, "init", "-b", "main")
    run_git(root, "add", "--all")
    run_git(root, "commit", "-m", "initial")
    return WorkspaceHandle(
        id=WorkspaceId.generate(),
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        kind=WorkspaceKind.GIT_WORKTREE,
        path=str(root),
        candidate_id=CandidateId.generate(),
    )


@pytest.fixture
def sandbox() -> SubprocessSandboxExecutor:
    return SubprocessSandboxExecutor()


@pytest.fixture
def backend(sandbox: SubprocessSandboxExecutor) -> TextSearchBackend:
    return TextSearchBackend(sandbox=sandbox)


def invocation(tool: str, kind: ToolKind, **arguments: object) -> ToolInvocation:
    return ToolInvocation(tool=tool, kind=kind, arguments=arguments, limits=LIMITS)


# --------------------------------------------------------------------- reading


async def test_read_file_returns_content_and_line_metadata(
    workspace: WorkspaceHandle,
) -> None:
    result = await ReadFileTool().execute(
        invocation=invocation("read_file", ToolKind.READ, path="src/app.py"),
        workspace=workspace,
    )
    assert result.exit_code == 0
    assert "def compute_total" in result.stdout
    assert result.metadata["total_lines"] == 7


async def test_read_file_can_return_a_line_range(workspace: WorkspaceHandle) -> None:
    result = await ReadFileTool().execute(
        invocation=invocation(
            "read_file", ToolKind.READ, path="src/app.py", start_line=5, max_lines=1
        ),
        workspace=workspace,
    )
    assert result.stdout == "class Ledger:"
    assert result.metadata["start_line"] == 5


async def test_reading_a_missing_file_is_a_result_not_an_exception(
    workspace: WorkspaceHandle,
) -> None:
    result = await ReadFileTool().execute(
        invocation=invocation("read_file", ToolKind.READ, path="nope.py"), workspace=workspace
    )
    assert result.exit_code == 1
    assert "no such file" in result.stderr


async def test_paths_cannot_escape_the_workspace(workspace: WorkspaceHandle) -> None:
    with pytest.raises(ToolExecutionError, match="escapes the workspace"):
        await ReadFileTool().execute(
            invocation=invocation("read_file", ToolKind.READ, path="../../etc/passwd"),
            workspace=workspace,
        )


# --------------------------------------------------------------------- editing


async def test_edit_file_replaces_a_unique_snippet(workspace: WorkspaceHandle) -> None:
    result = await EditFileTool().execute(
        invocation=invocation(
            "edit_file",
            ToolKind.EDIT,
            path="src/app.py",
            old_string="return sum(values)",
            new_string="return sum(values) + 1",
        ),
        workspace=workspace,
    )
    assert result.exit_code == 0
    assert "return sum(values) + 1" in (Path(workspace.path) / "src" / "app.py").read_text()


async def test_an_ambiguous_edit_changes_nothing(workspace: WorkspaceHandle) -> None:
    """Guessing which of two identical blocks was meant is silent corruption."""
    target = Path(workspace.path) / "dup.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")

    result = await EditFileTool().execute(
        invocation=invocation(
            "edit_file", ToolKind.EDIT, path="dup.py", old_string="x = 1", new_string="x = 2"
        ),
        workspace=workspace,
    )

    assert result.exit_code == 1
    assert result.metadata["occurrences"] == 2
    assert target.read_text(encoding="utf-8") == "x = 1\nx = 1\n"


async def test_edit_file_creates_and_appends(workspace: WorkspaceHandle) -> None:
    tool = EditFileTool()
    created = await tool.execute(
        invocation=invocation(
            "edit_file", ToolKind.EDIT, path="src/new.py", mode="create", content="a = 1\n"
        ),
        workspace=workspace,
    )
    assert created.exit_code == 0

    refused = await tool.execute(
        invocation=invocation(
            "edit_file", ToolKind.EDIT, path="src/new.py", mode="create", content="b = 2\n"
        ),
        workspace=workspace,
    )
    assert refused.exit_code == 1
    assert "already exists" in refused.stderr

    appended = await tool.execute(
        invocation=invocation(
            "edit_file", ToolKind.EDIT, path="src/new.py", mode="append", content="b = 2\n"
        ),
        workspace=workspace,
    )
    assert appended.exit_code == 0
    assert (Path(workspace.path) / "src" / "new.py").read_text() == "a = 1\nb = 2\n"


async def test_editing_a_read_only_workspace_raises(workspace: WorkspaceHandle) -> None:
    reviewer = replace(workspace, role=WorkspaceRole.REVIEWER, candidate_id=None)
    with pytest.raises(ToolExecutionError, match="read-only"):
        await EditFileTool().execute(
            invocation=invocation(
                "edit_file", ToolKind.EDIT, path="x.py", mode="create", content="x"
            ),
            workspace=reviewer,
        )


# -------------------------------------------------------------------- searching


async def test_search_repository_finds_matches(
    workspace: WorkspaceHandle, backend: TextSearchBackend
) -> None:
    result = await SearchRepositoryTool(backend).execute(
        invocation=invocation("search_repository", ToolKind.SEARCH, pattern="compute_total"),
        workspace=workspace,
    )
    assert result.exit_code == 0
    assert result.metadata["match_count"] >= 2
    assert "src/app.py" in result.stdout
    assert result.metadata["engine"] in ("ripgrep", "grep")


async def test_searching_for_nothing_is_a_success_with_zero_matches(
    workspace: WorkspaceHandle, backend: TextSearchBackend
) -> None:
    """Exit 1 from grep means "no match"; reported raw it would look like failure."""
    result = await SearchRepositoryTool(backend).execute(
        invocation=invocation(
            "search_repository", ToolKind.SEARCH, pattern="zzz_not_in_this_repository"
        ),
        workspace=workspace,
    )
    assert result.exit_code == 0
    assert result.metadata["match_count"] == 0


async def test_search_symbol_finds_definitions_across_languages(
    workspace: WorkspaceHandle, backend: TextSearchBackend
) -> None:
    result = await SearchSymbolTool(backend).execute(
        invocation=invocation("search_symbol", ToolKind.SEARCH, symbol="compute_total"),
        workspace=workspace,
    )
    assert result.exit_code == 0
    assert "src/app.py" in result.stdout
    assert "src/util.c" in result.stdout


async def test_search_symbol_refuses_a_regex_as_a_symbol(
    workspace: WorkspaceHandle, backend: TextSearchBackend
) -> None:
    with pytest.raises(ToolExecutionError, match="plain identifier"):
        await SearchSymbolTool(backend).execute(
            invocation=invocation("search_symbol", ToolKind.SEARCH, symbol=".*"),
            workspace=workspace,
        )


# ------------------------------------------------------------------------- vcs


async def test_git_status_and_diff_report_the_workspace_state(
    workspace: WorkspaceHandle,
) -> None:
    (Path(workspace.path) / "README.md").write_text("# demo changed\n", encoding="utf-8")

    status = await GitStatusTool().execute(
        invocation=invocation("git_status", ToolKind.VCS), workspace=workspace
    )
    diff = await GitDiffTool().execute(
        invocation=invocation("git_diff", ToolKind.VCS), workspace=workspace
    )

    assert status.exit_code == 0
    assert "README.md" in status.stdout
    assert diff.exit_code == 0
    assert "demo changed" in diff.stdout


async def test_apply_patch_tool_reports_a_conflict_without_touching_the_workspace(
    workspace: WorkspaceHandle,
) -> None:
    conflicting = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,1 +1,1 @@\n"
        "-this line is not in the file\n"
        "+something else\n"
    )
    before = (Path(workspace.path) / "README.md").read_text(encoding="utf-8")

    result = await ApplyPatchTool().execute(
        invocation=invocation("apply_patch", ToolKind.PATCH, patch=conflicting),
        workspace=workspace,
    )

    assert result.exit_code != 0
    assert result.metadata["applied"] is False
    assert (Path(workspace.path) / "README.md").read_text(encoding="utf-8") == before


async def test_apply_patch_tool_applies_a_valid_patch(workspace: WorkspaceHandle) -> None:
    patch = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,3 +1,3 @@\n"
        " # demo\n"
        " \n"
        "-A tiny project.\n"
        "+A tiny, patched project.\n"
    )
    result = await ApplyPatchTool().execute(
        invocation=invocation("apply_patch", ToolKind.PATCH, patch=patch), workspace=workspace
    )
    assert result.exit_code == 0
    assert result.metadata["applied"] is True
    assert "patched" in (Path(workspace.path) / "README.md").read_text(encoding="utf-8")


# -------------------------------------------------------------------- commands


async def test_run_command_returns_a_non_zero_exit_as_a_result(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    result = await RunCommandTool(sandbox=sandbox).execute(
        invocation=invocation(
            "run_command", ToolKind.COMMAND, command=[sys.executable, "-c", "raise SystemExit(4)"]
        ),
        workspace=workspace,
    )
    assert result.exit_code == 4
    assert result.tool == "run_command"
    assert result.kind is ToolKind.COMMAND


async def test_run_command_honours_an_executable_allow_list(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    tool = RunCommandTool(sandbox=sandbox, allowed_executables=["/bin/echo"])
    with pytest.raises(ToolExecutionError, match="not allowed"):
        await tool.execute(
            invocation=invocation("run_command", ToolKind.COMMAND, command=["/bin/rm", "-rf", "/"]),
            workspace=workspace,
        )


async def test_run_command_can_lower_but_not_raise_the_timeout(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    tool = RunCommandTool(sandbox=sandbox)
    result = await tool.execute(
        invocation=ToolInvocation(
            tool="run_command",
            kind=ToolKind.COMMAND,
            arguments={
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "timeout_seconds": 1,
            },
            limits=ExecutionLimits(timeout_seconds=300.0),
        ),
        workspace=workspace,
    )
    assert result.exit_code == 124
    assert result.duration_ms < 30_000


async def test_toolchain_tools_run_the_projects_own_commands(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    toolchain = ToolchainConfig(
        language="python",
        build_command=f"{sys.executable} -c pass",
        test_command=f"{sys.executable} -c exit(2)",
    )
    tools = {
        tool.name: tool for tool in build_toolchain_tools(toolchain=toolchain, sandbox=sandbox)
    }
    assert set(tools) == {"build", "run_tests"}

    build = await tools["build"].execute(
        invocation=invocation("build", ToolKind.BUILD), workspace=workspace
    )
    assert build.exit_code == 0
    assert build.kind is ToolKind.BUILD
    assert build.tool == "build"

    tests = await tools["run_tests"].execute(
        invocation=invocation("run_tests", ToolKind.TEST), workspace=workspace
    )
    assert tests.exit_code == 2
    assert tests.kind is ToolKind.TEST


async def test_a_toolchain_tool_passes_the_configured_environment(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    """Sanitizer and compiler settings travel with the project, not the host."""
    (Path(workspace.path) / "show_cc.py").write_text(
        "import os\nprint(os.environ.get('CC', 'unset'))\n", encoding="utf-8"
    )
    toolchain = ToolchainConfig(
        build_command=f"{sys.executable} show_cc.py", environment={"CC": "clang"}
    )

    (build,) = build_toolchain_tools(toolchain=toolchain, sandbox=sandbox)
    result = await build.execute(
        invocation=invocation("build", ToolKind.BUILD), workspace=workspace
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "clang"


async def test_a_toolchain_tool_runs_in_the_configured_subdirectory(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    (Path(workspace.path) / "src" / "show_cwd.py").write_text(
        "import os\nprint(os.getcwd())\n", encoding="utf-8"
    )
    toolchain = ToolchainConfig(
        build_command=f"{sys.executable} show_cwd.py", working_subdirectory="src"
    )
    (build,) = build_toolchain_tools(toolchain=toolchain, sandbox=sandbox)
    result = await build.execute(
        invocation=invocation("build", ToolKind.BUILD), workspace=workspace
    )
    assert same_path(result.stdout.strip(), Path(workspace.path) / "src")


# ------------------------------------------------------------------- toolchain


def test_c_and_cpp_profiles_cover_the_tools_the_spec_names() -> None:
    commands = " ".join(
        " ".join(
            filter(
                None,
                (
                    candidate.build_command,
                    candidate.test_command,
                    candidate.static_analysis_command,
                    candidate.install_command,
                    *candidate.environment.values(),
                ),
            )
        )
        for candidate in PROFILES.values()
    )
    for expected in ("gcc", "clang", "make", "cmake", "Ninja", "clang-tidy", "cppcheck"):
        assert expected in commands


def test_a_sanitizer_profile_sets_flags_and_fails_loudly() -> None:
    sanitized = with_sanitizer(profile("cpp-cmake-ninja"), "address")
    assert "-fsanitize=address" in sanitized.environment["CXXFLAGS"]
    assert "abort_on_error=1" in sanitized.environment["ASAN_OPTIONS"]

    config = sanitized.to_config()
    assert config.language == "cpp"
    assert config.build_command == sanitized.build_command

    with pytest.raises(ValueError, match="unknown sanitizer"):
        with_sanitizer(profile("c-make"), "nonsense")


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown toolchain profile"):
        profile("does-not-exist")


# ------------------------------------------------------------ repository context


async def test_the_context_is_bounded_by_max_files(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)
    context = await provider.build(
        workspace=workspace,
        request=ContextRequest(
            objective="find the total computation",
            queries=("compute_total",),
            max_files=1,
        ),
    )

    assert len(context.excerpts) == 1
    assert context.file_tree
    assert any("omitted" in note for note in context.notes)
    assert any("estimated" in note for note in context.notes)


async def test_the_context_is_bounded_by_max_tokens(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)
    context = await provider.build(
        workspace=workspace,
        request=ContextRequest(
            objective="anything", queries=("compute_total",), max_tokens=1, include_tree=False
        ),
    )
    assert context.excerpts == ()
    assert context.estimated_tokens == 0


async def test_explicit_paths_come_first(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)
    context = await provider.build(
        workspace=workspace,
        request=ContextRequest(
            objective="review the readme",
            paths=("README.md",),
            queries=("compute_total",),
            max_files=1,
        ),
    )
    assert [excerpt.path for excerpt in context.excerpts] == ["README.md"]
    assert context.render().startswith("### Files")


async def test_context_search_and_targeted_read(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)

    excerpts = await provider.search(workspace=workspace, pattern="class Ledger")
    assert [excerpt.path for excerpt in excerpts] == ["src/app.py"]
    assert excerpts[0].start_line == 5

    read = await provider.read_file(workspace=workspace, path="README.md")
    assert read is not None
    assert read.content.startswith("# demo")

    assert await provider.read_file(workspace=workspace, path="../escape") is None
    assert await provider.read_file(workspace=workspace, path="missing.txt") is None


def test_the_token_estimate_is_documented_and_crude() -> None:
    assert CHARS_PER_TOKEN == 4
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


async def test_an_objective_alone_still_finds_the_relevant_code(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    """The shape production actually sends, and the defect that shape hid.

    Every other test here hands the provider an explicit ``queries``. The only
    caller in production hands it neither ``queries`` nor ``paths`` — just the
    objective — and the ranker never read the objective. Agents were given a
    list of filenames and no code at all, and invented the rest.
    """
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)

    context = await provider.build(
        workspace=workspace,
        request=ContextRequest(objective="fix compute_total in the ledger"),
    )

    assert context.excerpts, "the agents were handed a file tree and no code"
    assert "compute_total" in context.render()
    assert context.estimated_tokens > 0


async def test_prose_around_the_terms_does_not_drown_them(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    """An objective is a sentence, not a query. Common words must not rank."""
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)

    context = await provider.build(
        workspace=workspace,
        request=ContextRequest(
            objective="Please could you have a look at the Ledger class and fix it",
        ),
    )

    paths = [excerpt.path for excerpt in context.excerpts]
    assert paths, "nothing matched an objective made mostly of filler"
    assert any("app.py" in path for path in paths), paths


async def test_an_objective_in_another_language_still_matches_identifiers(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    """Objectives are written by the user, in the user's language.

    The identifiers in them are not translated, so matching must not depend on
    the surrounding prose being English.
    """
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)

    context = await provider.build(
        workspace=workspace,
        request=ContextRequest(objective="Corrige la fonction compute_total du projet"),
    )

    assert context.excerpts
    assert "compute_total" in context.render()


async def test_explicit_queries_still_win_over_the_objective(
    workspace: WorkspaceHandle, sandbox: SubprocessSandboxExecutor
) -> None:
    """The objective is a fallback, never an override: a caller that knows
    what to look for must not have its query diluted."""
    provider = RipgrepRepositoryContextProvider(sandbox=sandbox)

    context = await provider.build(
        workspace=workspace,
        request=ContextRequest(objective="rewrite the readme", queries=("compute_total",)),
    )

    first = next(excerpt.path for excerpt in context.excerpts)
    assert first.endswith(("app.py", "util.c")), first
