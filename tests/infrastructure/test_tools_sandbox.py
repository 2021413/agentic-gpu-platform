"""Sandbox behaviour: limits, truncation, secrets and exit codes.

Nothing here needs a GPU, and only the Docker tests need Docker — they are
marked ``integration`` and skip themselves when the daemon or the image is
missing, so the default suite stays hermetic.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from domain.exceptions import ToolExecutionError
from domain.value_objects.identifiers import CandidateId, RunId, WorkspaceId
from domain.value_objects.tools import ExecutionLimits
from domain.value_objects.workspace import WorkspaceHandle, WorkspaceKind, WorkspaceRole
from infrastructure.tools.sandbox import (
    DockerSandboxExecutor,
    SubprocessSandboxExecutor,
    create_sandbox_executor,
)

DOCKER_IMAGE = "busybox:1.36"
"""A tiny image; the test skips rather than pulling it, so no test needs network."""


def docker_usable() -> bool:
    if not DockerSandboxExecutor().is_available():
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return probe.returncode == 0


def same_path(left: str | Path, right: str | Path) -> bool:
    """A blocking probe, kept out of the async test bodies."""
    return Path(left).samefile(right)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceHandle:
    path = tmp_path / "workspace"
    path.mkdir()
    return WorkspaceHandle(
        id=WorkspaceId.generate(),
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        kind=WorkspaceKind.GIT_WORKTREE,
        path=str(path),
        candidate_id=CandidateId.generate(),
    )


@pytest.fixture
def sandbox() -> SubprocessSandboxExecutor:
    return SubprocessSandboxExecutor()


def limits(**overrides: object) -> ExecutionLimits:
    defaults: dict[str, object] = {"timeout_seconds": 30.0, "max_output_bytes": 1_000_000}
    return ExecutionLimits(**{**defaults, **overrides})  # type: ignore[arg-type]


async def test_a_failing_command_is_a_result_not_an_exception(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    """The repair loop needs to read failures, so they must not be exceptions."""
    result = await sandbox.run(
        command=[sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        workspace=workspace,
        limits=limits(),
    )
    assert result.exit_code == 3
    assert result.succeeded is False
    assert "boom" in result.stderr
    assert result.duration_ms >= 0


async def test_a_missing_executable_raises_because_nothing_ran(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    with pytest.raises(ToolExecutionError) as error:
        await sandbox.run(
            command=["/nonexistent/definitely-not-here"], workspace=workspace, limits=limits()
        )
    assert error.value.code == "tool_execution_failed"


async def test_a_missing_workspace_raises(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle, tmp_path: Path
) -> None:
    gone = replace(workspace, path=str(tmp_path / "does-not-exist"))
    with pytest.raises(ToolExecutionError, match="workspace directory"):
        await sandbox.run(command=[sys.executable, "-V"], workspace=gone, limits=limits())


async def test_the_timeout_is_hard(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    result = await sandbox.run(
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
        workspace=workspace,
        limits=limits(timeout_seconds=1.0),
    )
    assert result.exit_code == 124
    assert result.metadata["timed_out"] is True
    assert result.duration_ms < 30_000


async def test_output_is_truncated_and_says_so(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    result = await sandbox.run(
        command=[sys.executable, "-c", "print('x' * 200_000)"],
        workspace=workspace,
        limits=limits(max_output_bytes=1_000),
    )
    assert result.exit_code == 0
    assert result.truncated is True
    assert len(result.stdout.encode()) <= 1_000


async def test_output_under_the_cap_is_not_flagged(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    result = await sandbox.run(
        command=[sys.executable, "-c", "print('small')"],
        workspace=workspace,
        limits=limits(max_output_bytes=1_000),
    )
    assert result.truncated is False
    assert result.stdout.strip() == "small"


async def test_control_plane_secrets_never_reach_the_sandbox(
    sandbox: SubprocessSandboxExecutor,
    workspace: WorkspaceHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allow-list is the point: a new secret leaks nothing by default."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@db/platform")
    monkeypatch.setenv("REDIS_PASSWORD", "top-secret")

    result = await sandbox.run(
        command=[sys.executable, "-c", "import os, json; print(json.dumps(dict(os.environ)))"],
        workspace=workspace,
        limits=limits(),
    )

    assert result.exit_code == 0
    assert "hunter2" not in result.stdout
    assert "DATABASE_URL" not in result.stdout
    assert "REDIS_PASSWORD" not in result.stdout


async def test_explicitly_passed_environment_is_available(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    result = await sandbox.run(
        command=[sys.executable, "-c", "import os; print(os.environ.get('CFLAGS', ''))"],
        workspace=workspace,
        limits=limits(),
        environment={"CFLAGS": "-fsanitize=address"},
    )
    assert result.stdout.strip() == "-fsanitize=address"


async def test_the_command_starts_in_the_workspace(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    result = await sandbox.run(
        command=[sys.executable, "-c", "import os; print(os.getcwd())"],
        workspace=workspace,
        limits=limits(),
    )
    assert same_path(result.stdout.strip(), workspace.path)


async def test_resource_limits_are_applied_to_the_child(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    """Proves the preexec hook ran, without relying on a fragile OOM."""
    script = (
        "import resource; "
        "print(resource.getrlimit(resource.RLIMIT_CORE)[0], "
        "resource.getrlimit(resource.RLIMIT_AS)[0])"
    )
    result = await sandbox.run(
        command=[sys.executable, "-c", script],
        workspace=workspace,
        limits=limits(memory_mb=512),
    )
    core, address_space = result.stdout.split()
    assert core == "0"
    assert int(address_space) == 512 * 1024 * 1024


async def test_rlimits_can_be_disabled_for_hosts_that_forbid_them(
    workspace: WorkspaceHandle,
) -> None:
    executor = SubprocessSandboxExecutor(apply_rlimits=False)
    result = await executor.run(
        command=[
            sys.executable,
            "-c",
            "import resource; print(resource.getrlimit(resource.RLIMIT_CORE)[0])",
        ],
        workspace=workspace,
        limits=limits(),
    )
    assert result.stdout.strip() == str(resource.getrlimit(resource.RLIMIT_CORE)[0])


async def test_network_is_only_advisory_in_the_subprocess_sandbox(
    sandbox: SubprocessSandboxExecutor, workspace: WorkspaceHandle
) -> None:
    """Honesty check: the result must not claim an isolation it cannot provide."""
    result = await sandbox.run(
        command=[sys.executable, "-c", "import os; print(os.environ.get('http_proxy'))"],
        workspace=workspace,
        limits=limits(network_enabled=False),
    )
    assert result.metadata["network_isolated"] is False
    assert result.stdout.strip() == "http://127.0.0.1:9"


def test_docker_absence_degrades_to_a_clear_error() -> None:
    executor = DockerSandboxExecutor(executable="docker-that-does-not-exist")
    assert executor.is_available() is False


async def test_docker_absence_is_reported_as_a_tool_execution_error(
    workspace: WorkspaceHandle,
) -> None:
    executor = DockerSandboxExecutor(executable="docker-that-does-not-exist")
    with pytest.raises(ToolExecutionError, match="docker is not available"):
        await executor.run(command=["true"], workspace=workspace, limits=limits())


def test_the_factory_falls_back_to_the_subprocess_sandbox() -> None:
    executor = create_sandbox_executor(prefer_docker=False)
    assert isinstance(executor, SubprocessSandboxExecutor)


@pytest.mark.integration
@pytest.mark.skipif(not docker_usable(), reason="docker daemon or test image unavailable")
async def test_docker_sandbox_runs_in_a_container(workspace: WorkspaceHandle) -> None:
    executor = DockerSandboxExecutor(image=DOCKER_IMAGE)
    (Path(workspace.path) / "marker.txt").write_text("mounted\n", encoding="utf-8")

    result = await executor.run(
        command=["cat", "marker.txt"], workspace=workspace, limits=limits(memory_mb=256)
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "mounted"
    assert result.metadata["network_isolated"] is True
    assert result.metadata["filesystem_isolated"] is True


@pytest.mark.integration
@pytest.mark.skipif(not docker_usable(), reason="docker daemon or test image unavailable")
async def test_docker_sandbox_reports_failures_as_results(workspace: WorkspaceHandle) -> None:
    executor = DockerSandboxExecutor(image=DOCKER_IMAGE)
    result = await executor.run(
        command=["sh", "-c", "exit 7"], workspace=workspace, limits=limits()
    )
    assert result.exit_code == 7


@pytest.mark.integration
@pytest.mark.skipif(not docker_usable(), reason="docker daemon or test image unavailable")
async def test_docker_sandbox_has_no_network(workspace: WorkspaceHandle) -> None:
    executor = DockerSandboxExecutor(image=DOCKER_IMAGE)
    result = await executor.run(
        command=["ping", "-c", "1", "-W", "1", "1.1.1.1"],
        workspace=workspace,
        limits=limits(timeout_seconds=20.0),
    )
    assert result.exit_code != 0


@pytest.mark.integration
@pytest.mark.skipif(not docker_usable(), reason="docker daemon or test image unavailable")
async def test_docker_sandbox_carries_no_host_environment(
    workspace: WorkspaceHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@db/platform")
    executor = DockerSandboxExecutor(image=DOCKER_IMAGE)
    result = await executor.run(command=["env"], workspace=workspace, limits=limits())
    assert "hunter2" not in result.stdout
    assert os.environ["DATABASE_URL"] not in result.stdout
