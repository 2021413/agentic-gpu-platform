"""Default in-process run coordinator.

Enough for a single orchestrator process, which is the supported v1 deployment.
It is deliberately explicit about its limit: with several orchestrator replicas
this must be swapped for a distributed lock, and nothing else in the platform
changes because the coordinator is a port.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from domain.value_objects.identifiers import RunId

__all__ = ["InProcessRunCoordinator"]


class InProcessRunCoordinator:
    """One asyncio lock per run, created on demand and reference-counted."""

    __slots__ = ("_guard", "_holders", "_locks")

    def __init__(self) -> None:
        self._locks: dict[RunId, asyncio.Lock] = {}
        self._holders: dict[RunId, int] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def lock(self, run_id: RunId) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.setdefault(run_id, asyncio.Lock())
            self._holders[run_id] = self._holders.get(run_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                remaining = self._holders[run_id] - 1
                if remaining <= 0:
                    # Drop the lock once nobody waits on it, so a long-lived
                    # orchestrator does not accumulate one entry per run forever.
                    self._holders.pop(run_id, None)
                    self._locks.pop(run_id, None)
                else:
                    self._holders[run_id] = remaining
