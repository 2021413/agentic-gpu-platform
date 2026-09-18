"""Bounded process execution: the primitive every sandbox is built on.

Three properties matter here and nowhere else in the codebase:

* the wall clock is a hard bound — a hung test suite must not pin an executor
  slot forever, so the whole process *group* is killed, not just the child;
* output is capped while it is read, not after — a runaway compiler printing a
  gigabyte of errors must not be able to exhaust the host's memory;
* the pipes are always drained, because a child blocked on a full pipe would
  never reach the point where it could be killed cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TIMEOUT_EXIT_CODE", "ProcessOutcome", "run_capped_process"]

TIMEOUT_EXIT_CODE = 124
"""GNU ``timeout``'s convention; agents and humans already read it that way."""

_CHUNK = 64 * 1024
_DRAIN_GRACE_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """What running a command produced, including how it was cut short."""

    exit_code: int
    stdout: str
    stderr: str
    truncated: bool
    duration_ms: int
    timed_out: bool


async def run_capped_process(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    max_output_bytes: int,
    preexec: Callable[[], None] | None = None,
) -> ProcessOutcome:
    """Run ``command`` to completion, a timeout, or the output cap.

    Raises ``OSError`` when the process cannot be started at all; callers turn
    that into the domain's ``ToolExecutionError``, because "could not run" and
    "ran and failed" must never be confused.
    """
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=dict(environment),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Its own session, so a single killpg reaches every grandchild a build
        # system spawned.
        start_new_session=True,
        preexec_fn=preexec,
    )

    out_stream, err_stream = process.stdout, process.stderr
    if out_stream is None or err_stream is None:  # pragma: no cover - PIPE guarantees both
        raise OSError("subprocess pipes were not created")
    reader = asyncio.gather(
        _read_capped(out_stream, max_output_bytes),
        _read_capped(err_stream, max_output_bytes),
    )

    timed_out = False
    try:
        out, err = await asyncio.wait_for(asyncio.shield(reader), timeout_seconds)
    except TimeoutError:
        timed_out = True
        _kill_group(process)
        # The reader was shielded, so it survives the timeout and finishes as
        # soon as the killed process closes its pipes.
        try:
            out, err = await asyncio.wait_for(reader, _DRAIN_GRACE_SECONDS)
        except TimeoutError:  # pragma: no cover - a pipe held by an unrelated process
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
            out, err = (b"", False), (b"", False)

    exit_code = await process.wait()
    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_bytes, stdout_truncated = out
    stderr_bytes, stderr_truncated = err
    return ProcessOutcome(
        exit_code=TIMEOUT_EXIT_CODE if timed_out else exit_code,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        truncated=stdout_truncated or stderr_truncated,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Keep at most ``limit`` bytes, but keep reading so the child never blocks."""
    kept: list[bytes] = []
    size = 0
    truncated = False
    while True:
        chunk = await stream.read(_CHUNK)
        if not chunk:
            return b"".join(kept), truncated
        if size < limit:
            kept.append(chunk[: limit - size])
        if size + len(chunk) > limit:
            truncated = True
        size += len(chunk)


def _kill_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:  # pragma: no cover - finished while we timed out
        return
    try:
        os.killpg(os.getpgid(process.pid), 9)
    except (ProcessLookupError, PermissionError):  # pragma: no cover - race with exit
        with contextlib.suppress(ProcessLookupError):
            process.kill()
