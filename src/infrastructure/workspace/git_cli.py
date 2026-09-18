"""Asyncio wrapper around the ``git`` executable.

No third-party git binding is used: the platform already depends on a real
``git`` for worktrees, and shelling out keeps the behaviour identical to what an
operator would reproduce by hand when a run has to be debugged. Every command
runs with a deterministic, minimal environment so that a developer's global git
configuration can never change what the platform does in production.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from domain.exceptions import WorkspaceError

__all__ = ["GitCommandRunner", "GitResult"]

_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
"""Git's hash of the empty tree, used to diff a repository that has no commit yet."""

_INHERITED_ENVIRONMENT_KEYS = ("PATH", "LANG", "LC_ALL", "TZ", "SSH_AUTH_SOCK")
"""Host variables git legitimately needs. Everything else is dropped on purpose."""


@dataclass(frozen=True, slots=True)
class GitResult:
    """Outcome of one git invocation.

    Non-zero exits are ordinary here (``git diff --quiet`` uses them to answer a
    question), so callers decide what deserves an exception.
    """

    args: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    @property
    def out(self) -> str:
        """Stripped stdout — most git answers are a single line."""
        return self.stdout.strip()


class GitCommandRunner:
    """Runs git commands, converting failures into ``WorkspaceError``."""

    __slots__ = ("_executable", "_identity", "_timeout_seconds")

    def __init__(
        self,
        *,
        executable: str = "git",
        timeout_seconds: float = 120.0,
        author_name: str = "agentic-platform",
        author_email: str = "agents@agentic.invalid",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._identity = (
            "-c",
            f"user.name={author_name}",
            "-c",
            f"user.email={author_email}",
            "-c",
            "commit.gpgsign=false",
            # A repository's own hooks are attacker-controlled input as far as
            # this platform is concerned: never execute them on our behalf.
            "-c",
            "core.hooksPath=/dev/null",
        )

    @property
    def empty_tree(self) -> str:
        return _EMPTY_TREE

    async def run(
        self,
        *args: str,
        cwd: Path | str,
        stdin: str | None = None,
        check: bool = True,
        extra_env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> GitResult:
        """Execute one git command inside ``cwd``.

        ``check`` raises on a non-zero exit; callers that interrogate git (``diff
        --quiet``, ``worktree remove`` during cleanup) pass ``check=False``.
        """
        command = (self._executable, *self._identity, *args)
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=self._environment(cwd, extra_env),
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise WorkspaceError(
                "git could not be executed", executable=self._executable, reason=str(exc)
            ) from exc

        payload = stdin.encode() if stdin is not None else None
        try:
            raw_out, raw_err = await asyncio.wait_for(process.communicate(payload), timeout)
        except TimeoutError as exc:
            _kill(process)
            await process.wait()
            raise WorkspaceError(
                "git command timed out",
                command=" ".join(args),
                timeout_seconds=timeout,
            ) from exc

        result = GitResult(
            args=tuple(args),
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=raw_out.decode("utf-8", errors="replace"),
            stderr=raw_err.decode("utf-8", errors="replace"),
        )
        if check and not result.succeeded:
            raise WorkspaceError(
                "git command failed",
                command=" ".join(args),
                exit_code=result.exit_code,
                stderr=result.stderr.strip()[:2000],
                cwd=str(cwd),
            )
        return result

    def _environment(self, cwd: Path | str, extra: Mapping[str, str] | None) -> dict[str, str]:
        """A minimal, reproducible environment.

        Global and system git configuration are neutralised so that a run's
        behaviour depends on the repository and on this code, not on whatever
        the host happens to have in ``~/.gitconfig``.
        """
        env = {key: os.environ[key] for key in _INHERITED_ENVIRONMENT_KEYS if key in os.environ}
        env.update(
            {
                "HOME": str(cwd),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ASKPASS": "",
                "GIT_PAGER": "cat",
                "LC_ALL": "C",
            }
        )
        if extra:
            env.update(extra)
        return env


def _kill(process: asyncio.subprocess.Process) -> None:
    """Kill the whole process group; git spawns helpers that outlive it."""
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), 9)
    except (ProcessLookupError, PermissionError):  # pragma: no cover - race with exit
        _kill_directly(process)


def _kill_directly(process: asyncio.subprocess.Process) -> None:  # pragma: no cover - rare race
    with contextlib.suppress(ProcessLookupError):
        process.kill()
