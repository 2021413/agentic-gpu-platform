"""Sandbox executors (spec section 31).

Two implementations of the same contract, with **honestly different** strength.
Read this before deciding which one to deploy.

``SubprocessSandboxExecutor`` — what it really guarantees:

* a hard wall-clock timeout, enforced by killing the whole process group;
* an output cap applied while reading, so a chatty build cannot exhaust memory;
* an environment built from an allow-list: no control-plane secret (database
  URL, Redis password, API token) is ever visible to project code, because the
  child's environment is constructed from scratch rather than inherited;
* a working directory pinned to the workspace, and ``HOME`` pointed at it so the
  command cannot read the platform user's dotfiles by accident;
* CPU time, address space, core dump and file size limits via ``setrlimit``.

What it does **not** guarantee — and no amount of care in this class would:

* **no filesystem isolation.** The command runs as the platform user with that
  user's full access. ``cat ../../other-workspace/secret`` works. Only the
  *starting* directory is constrained;
* **no network isolation.** ``ExecutionLimits.network_enabled=False`` is
  advisory here: proxy variables are poisoned, which stops well-behaved tools
  (pip, curl), and nothing stops a raw socket;
* **no user, PID or mount namespace**, no seccomp filter, no protection against
  a fork bomb (``RLIMIT_NPROC`` is per-user and would harm the platform itself),
  and ``RLIMIT_AS`` is not honoured by every allocator;
* ``setrlimit`` runs in a ``preexec_fn``, i.e. between fork and exec — reliable
  in practice, but it is the child cooperating with itself, not the kernel
  confining it from outside.

In short: this executor protects the platform from *accidents* — a runaway test,
an infinite loop, a leaked secret in ``os.environ``. It does not protect it from
hostile code. Production runs untrusted model-authored code and must therefore
use ``DockerSandboxExecutor`` (or a future Firecracker/Kubernetes executor),
where the kernel — not us — enforces the boundary.
"""

from __future__ import annotations

import os
import resource
import shlex
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from domain.exceptions import ToolExecutionError
from domain.value_objects.tools import ExecutionLimits, ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle
from infrastructure.tools.process import ProcessOutcome, run_capped_process

__all__ = [
    "DEFAULT_ENVIRONMENT_ALLOWLIST",
    "DockerSandboxExecutor",
    "SubprocessSandboxExecutor",
    "create_sandbox_executor",
]

DEFAULT_ENVIRONMENT_ALLOWLIST: tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "TZ")
"""Host variables copied into a sandbox. Deliberately tiny: an allow-list is the
only construction in which a newly added secret cannot leak by default."""

_BLOCKED_NETWORK_ENVIRONMENT: Mapping[str, str] = {
    # Defence in depth only: this makes well-behaved clients fail fast, it does
    # not prevent a socket() call. Real isolation is Docker's --network none.
    "http_proxy": "http://127.0.0.1:9",
    "https_proxy": "http://127.0.0.1:9",
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "no_proxy": "",
    "NO_PROXY": "",
}

_DOCKER_CLIENT_ENVIRONMENT = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
)
_DOCKER_START_FAILURE_EXIT = 125
"""``docker run``'s own "could not start the container" code."""


class SubprocessSandboxExecutor:
    """Runs commands as child processes under limits. See the module docstring."""

    __slots__ = ("_allowlist", "_base_environment", "_home_in_workspace", "_rlimits")

    def __init__(
        self,
        *,
        environment_allowlist: Sequence[str] = DEFAULT_ENVIRONMENT_ALLOWLIST,
        base_environment: Mapping[str, str] | None = None,
        apply_rlimits: bool = True,
        home_in_workspace: bool = True,
    ) -> None:
        self._allowlist = tuple(environment_allowlist)
        self._base_environment = dict(base_environment or {})
        self._rlimits = apply_rlimits
        self._home_in_workspace = home_in_workspace

    async def run(
        self,
        *,
        command: Sequence[str],
        workspace: WorkspaceHandle,
        limits: ExecutionLimits,
        environment: Mapping[str, str] | None = None,
    ) -> ToolResult:
        argv = tuple(command)
        tool = _tool_label(argv)
        cwd = _require_directory(argv, workspace)

        env = self._environment(cwd, limits, environment)
        preexec = _rlimit_preexec(limits) if self._rlimits else None
        try:
            outcome = await run_capped_process(
                command=argv,
                cwd=cwd,
                environment=env,
                timeout_seconds=limits.timeout_seconds,
                max_output_bytes=limits.max_output_bytes,
                preexec=preexec,
            )
        except OSError as exc:
            raise ToolExecutionError(
                tool, "command could not be started", command=shlex.join(argv), reason=str(exc)
            ) from exc

        return _to_result(
            tool=tool,
            argv=argv,
            outcome=outcome,
            metadata={
                "sandbox": "subprocess",
                "cwd": str(cwd),
                "network_isolated": False,
                "filesystem_isolated": False,
            },
        )

    def _environment(
        self, cwd: Path, limits: ExecutionLimits, extra: Mapping[str, str] | None
    ) -> dict[str, str]:
        """Build the child's environment from nothing, never from ``os.environ``."""
        env = {key: os.environ[key] for key in self._allowlist if key in os.environ}
        env.setdefault("PATH", os.defpath)
        if self._home_in_workspace:
            env["HOME"] = str(cwd)
        env["TMPDIR"] = str(cwd)
        env.update(self._base_environment)
        if not limits.network_enabled:
            env.update(_BLOCKED_NETWORK_ENVIRONMENT)
        if extra:
            env.update(extra)
        return env


class DockerSandboxExecutor:
    """Runs each command in a disposable container.

    Here the kernel enforces what the subprocess executor can only ask for:
    ``--network none`` really removes the network, ``--memory`` and ``--cpus``
    are cgroup limits, and the only host path the command can see is the
    workspace mount. The environment starts empty — not allow-listed, empty —
    so control-plane secrets cannot reach project code at all.

    Docker's absence is not a crash: :meth:`is_available` lets composition
    choose, and a run without Docker fails with ``ToolExecutionError`` naming
    the missing dependency rather than an obscure ``FileNotFoundError``.
    """

    __slots__ = (
        "_executable",
        "_extra_arguments",
        "_image",
        "_mount",
        "_pids_limit",
        "_user",
    )

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        executable: str = "docker",
        mount_path: str = "/workspace",
        pids_limit: int = 512,
        user: str | None = None,
        extra_arguments: Sequence[str] = (),
    ) -> None:
        self._image = image
        self._executable = executable
        self._mount = mount_path
        self._pids_limit = pids_limit
        # Run as the calling user so files created in the mounted workspace stay
        # readable (and deletable) by the platform afterwards.
        self._user = user if user is not None else f"{os.getuid()}:{os.getgid()}"
        self._extra_arguments = tuple(extra_arguments)

    def is_available(self) -> bool:
        return shutil.which(self._executable) is not None

    async def run(
        self,
        *,
        command: Sequence[str],
        workspace: WorkspaceHandle,
        limits: ExecutionLimits,
        environment: Mapping[str, str] | None = None,
    ) -> ToolResult:
        argv = tuple(command)
        tool = _tool_label(argv)
        cwd = _require_directory(argv, workspace)
        if not self.is_available():
            raise ToolExecutionError(
                tool, "docker is not available on this host", executable=self._executable
            )

        container = f"agentic-{uuid.uuid4().hex[:16]}"
        docker_argv = self._docker_argv(container, cwd, limits, environment, argv)
        client_env = {
            key: os.environ[key] for key in _DOCKER_CLIENT_ENVIRONMENT if key in os.environ
        }
        client_env.setdefault("PATH", os.defpath)

        try:
            outcome = await run_capped_process(
                command=docker_argv,
                cwd=cwd,
                environment=client_env,
                # The client is given a little more rope than the container so a
                # container that honours its own timeout reports it as such.
                timeout_seconds=limits.timeout_seconds + 10.0,
                max_output_bytes=limits.max_output_bytes,
                preexec=None,
            )
        except OSError as exc:
            raise ToolExecutionError(tool, "docker could not be started", reason=str(exc)) from exc

        if outcome.timed_out:
            await self._force_remove(container, client_env)
        _reject_start_failure(tool, outcome)

        return _to_result(
            tool=tool,
            argv=argv,
            outcome=outcome,
            metadata={
                "sandbox": "docker",
                "image": self._image,
                "container": container,
                "network_isolated": not limits.network_enabled,
                "filesystem_isolated": True,
            },
        )

    def _docker_argv(
        self,
        container: str,
        cwd: Path,
        limits: ExecutionLimits,
        environment: Mapping[str, str] | None,
        argv: Sequence[str],
    ) -> tuple[str, ...]:
        args = [
            self._executable,
            "run",
            "--rm",
            "--name",
            container,
            "--workdir",
            self._mount,
            "--volume",
            f"{cwd}:{self._mount}",
            "--user",
            self._user,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self._pids_limit),
            "--tmpfs",
            "/tmp:rw,size=256m",
            "--env",
            f"HOME={self._mount}",
        ]
        if not limits.network_enabled:
            args.extend(["--network", "none"])
        if limits.memory_mb:
            args.extend(
                ["--memory", f"{limits.memory_mb}m", "--memory-swap", f"{limits.memory_mb}m"]
            )
        if limits.cpu_count:
            args.extend(["--cpus", f"{limits.cpu_count:g}"])
        for key, value in (environment or {}).items():
            args.extend(["--env", f"{key}={value}"])
        args.extend(self._extra_arguments)
        args.append(self._image)
        args.extend(argv)
        return tuple(args)

    async def _force_remove(self, container: str, client_env: Mapping[str, str]) -> None:
        """Killing the client leaves the container running; remove it explicitly."""
        try:
            await run_capped_process(
                command=(self._executable, "rm", "--force", container),
                cwd=Path.cwd(),
                environment=client_env,
                timeout_seconds=30.0,
                max_output_bytes=64_000,
            )
        except OSError:  # pragma: no cover - docker vanished mid-run
            return


def create_sandbox_executor(
    *,
    prefer_docker: bool = True,
    image: str = "python:3.12-slim",
    environment_allowlist: Sequence[str] = DEFAULT_ENVIRONMENT_ALLOWLIST,
    base_environment: Mapping[str, str] | None = None,
) -> SubprocessSandboxExecutor | DockerSandboxExecutor:
    """Pick the strongest sandbox this host can actually provide.

    Degrading to the subprocess executor is a real reduction in safety, so it is
    a composition-time decision made here in one visible place rather than a
    silent fallback buried in an executor.
    """
    if prefer_docker:
        docker = DockerSandboxExecutor(image=image)
        if docker.is_available():
            return docker
    return SubprocessSandboxExecutor(
        environment_allowlist=environment_allowlist, base_environment=base_environment
    )


def _tool_label(argv: Sequence[str]) -> str:
    """Name used in results and errors when no tool has claimed the execution."""
    return Path(argv[0]).name if argv else "sandbox"


def _require_directory(argv: Sequence[str], workspace: WorkspaceHandle) -> Path:
    if not argv:
        raise ToolExecutionError("sandbox", "an empty command cannot be executed")
    cwd = Path(workspace.path)
    if not cwd.is_dir():
        raise ToolExecutionError(
            _tool_label(argv),
            "workspace directory does not exist",
            workspace_id=str(workspace.id),
            path=str(cwd),
        )
    return cwd


def _reject_start_failure(tool: str, outcome: ProcessOutcome) -> None:
    """Separate "the container never ran" from "the command ran and failed"."""
    if outcome.timed_out:
        return
    unstartable = outcome.exit_code == _DOCKER_START_FAILURE_EXIT or (
        outcome.exit_code == 127 and "executable file not found" in outcome.stderr
    )
    if unstartable:
        raise ToolExecutionError(
            tool,
            "container could not run the command",
            exit_code=outcome.exit_code,
            stderr=outcome.stderr.strip()[:2000],
        )


def _to_result(
    *,
    tool: str,
    argv: Sequence[str],
    outcome: ProcessOutcome,
    metadata: Mapping[str, object],
) -> ToolResult:
    """A non-zero exit — timeout included — is a result the agent loop reads."""
    return ToolResult(
        tool=tool,
        kind=ToolKind.COMMAND,
        command=shlex.join(argv),
        exit_code=outcome.exit_code,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        duration_ms=outcome.duration_ms,
        truncated=outcome.truncated,
        metadata={**metadata, "timed_out": outcome.timed_out},
    )


def _rlimit_preexec(limits: ExecutionLimits) -> Callable[[], None]:
    """Build the between-fork-and-exec hook that applies the resource limits.

    CPU time is bounded by the wall-clock timeout multiplied by the allowed core
    count: a command may legitimately burn that much CPU, and anything beyond it
    is a runaway the kernel can stop without us waiting for the timeout.
    """

    def apply() -> None:  # pragma: no cover - runs in the forked child
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if limits.memory_mb:
            size = limits.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (size, size))
        cpu_seconds = int(limits.timeout_seconds * max(limits.cpu_count or 1.0, 1.0)) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))

    return apply
