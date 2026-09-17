"""Dynamic worker discovery (spec section 4).

The worker pool is *discovered*, never configured. Adding or removing a GPU
must require no restart and no code change, so the orchestrator asks the
registry on every scheduling decision instead of holding a static list.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from domain.entities.worker import Worker
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import WorkerLoad

__all__ = ["WorkerRegistry"]


@runtime_checkable
class WorkerRegistry(Protocol):
    """Registration, heartbeats and liveness for the GPU pool."""

    async def register(self, worker: Worker, *, ttl: timedelta) -> None:
        """Add or refresh a worker registration with a heartbeat TTL."""
        ...

    async def heartbeat(
        self, worker_id: WorkerId, *, load: WorkerLoad, at: datetime, ttl: timedelta
    ) -> bool:
        """Refresh liveness. Returns False if the worker is no longer registered."""
        ...

    async def get(self, worker_id: WorkerId) -> Worker | None: ...

    async def list_all(self) -> Sequence[Worker]:
        """Every known worker, including draining and unavailable ones."""
        ...

    async def list_available(self) -> Sequence[Worker]:
        """Workers that may receive new jobs right now."""
        ...

    async def update(self, worker: Worker) -> None:
        """Persist a state change made on the entity (drain, unhealthy, load)."""
        ...

    async def deregister(self, worker_id: WorkerId) -> None: ...

    async def reap_stale(
        self, *, now: datetime, heartbeat_timeout: timedelta
    ) -> Sequence[WorkerId]:
        """Mark silent workers unavailable and return them.

        Their in-flight jobs are then reclaimed through lease expiry; this is
        the detection half of that mechanism.
        """
        ...
