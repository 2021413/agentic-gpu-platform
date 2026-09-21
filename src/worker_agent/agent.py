"""The worker-side lifecycle (spec sections 3.3, 25 and 44).

Registration, heartbeats, draining and departure — the four things that make a
GPU appear in, and disappear from, an elastic pool without anyone restarting
anything.

The agent is deliberately defensive about one case: the control plane may
forget it. A restarted or re-deployed control plane answers 404 to a heartbeat,
and the correct reaction is to register again rather than to die quietly and
leave a GPU idle.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from domain.enums import AgentRole, WorkerStatus
from worker_agent.client import ControlPlaneClient, ControlPlaneError
from worker_agent.inference import InferenceProbe

__all__ = ["WorkerAgent", "WorkerDescription"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerDescription:
    """What this worker advertises at registration."""

    endpoint: str
    model_id: str
    context_length: int
    max_concurrency: int
    roles: frozenset[AgentRole]
    gpu_type: str | None = None
    gpu_count: int = 1
    tensor_parallel_size: int = 1
    worker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "context_length": self.context_length,
            "max_concurrency": self.max_concurrency,
            "supported_roles": sorted(str(role) for role in self.roles),
            "gpu": {
                "gpu_type": self.gpu_type,
                "gpu_count": self.gpu_count,
                "tensor_parallel_size": self.tensor_parallel_size,
            },
            "metadata": dict(self.metadata),
        }
        if self.worker_id:
            payload["worker_id"] = self.worker_id
        return payload


class WorkerAgent:
    """Keeps one inference server registered and honest about its state."""

    def __init__(
        self,
        *,
        client: ControlPlaneClient,
        probe: InferenceProbe,
        description: WorkerDescription,
        heartbeat_interval_seconds: float = 10.0,
        drain_timeout_seconds: float = 300.0,
        startup_timeout_seconds: float = 900.0,
    ) -> None:
        self._client = client
        self._probe = probe
        self._description = description
        self._interval = heartbeat_interval_seconds
        self._drain_timeout = drain_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._worker_id: str | None = description.worker_id
        self._status = WorkerStatus.STARTING
        self._active_jobs = 0
        self._stopping = asyncio.Event()

    @property
    def worker_id(self) -> str | None:
        return self._worker_id

    @property
    def status(self) -> WorkerStatus:
        return self._status

    # ------------------------------------------------------------------
    async def start(self) -> str:
        """Wait for the engine, then register. Returns the assigned worker id.

        Registering before the engine can serve would advertise capacity that
        does not exist, and the first job scheduled onto it would be wasted.
        """
        self._status = WorkerStatus.STARTING
        ready = await self._probe.wait_until_ready(timeout_seconds=self._startup_timeout)
        if not ready:
            raise ControlPlaneError(
                "the local inference server never became ready; refusing to register"
            )

        self._status = WorkerStatus.REGISTERING
        body = await self._client.register(self._description.as_payload())
        worker_id = str(body.get("id") or self._description.worker_id or "")
        if not worker_id:
            raise ControlPlaneError("the control plane did not return a worker id")
        self._worker_id = worker_id
        self._status = WorkerStatus.READY
        _log.info("registered as worker %s (model %s)", worker_id, self._description.model_id)
        return worker_id

    async def beat_once(self) -> bool:
        """Send one heartbeat. Returns False when the beat could not be delivered.

        A control plane that no longer knows us triggers re-registration: a
        healthy GPU must not be lost because the other side restarted.
        """
        if self._worker_id is None:
            return False
        # Flat, not nested under "load". The domain command carries a WorkerLoad
        # value object, but the HTTP schema flattens it at the boundary and
        # forbids unknown fields — so a nested payload is rejected with a 422
        # that registration, which agrees on its shape, never reveals. The two
        # sides were written against the same idea and never against each other.
        payload = {
            "active_jobs": self._active_jobs,
            "queued_jobs": 0,
            "draining": self._status is WorkerStatus.DRAINING,
        }
        try:
            await self._client.heartbeat(self._worker_id, payload)
        except ControlPlaneError as exc:
            if exc.is_unknown_worker:
                _log.warning("the control plane forgot this worker; registering again")
                with contextlib.suppress(ControlPlaneError):
                    await self.start()
                return self._worker_id is not None
            _log.warning("heartbeat failed: %s", exc)
            return False
        if not await self._probe.is_healthy():
            # Report the truth rather than a comforting default: a worker whose
            # engine died must stop receiving work.
            self._status = WorkerStatus.UNHEALTHY
            return False
        if self._status is WorkerStatus.UNHEALTHY:
            self._status = WorkerStatus.READY
        return True

    async def run(self) -> None:
        """Register, then heartbeat until asked to stop, then leave cleanly."""
        await self.start()
        try:
            while not self._stopping.is_set():
                await self.beat_once()
                await self._wait(self._interval)
        finally:
            await self.shutdown()

    async def request_drain(self) -> None:
        """Stop accepting work. Called on SIGTERM, and by the control plane."""
        if self._worker_id is None:
            return
        self._status = WorkerStatus.DRAINING
        with contextlib.suppress(ControlPlaneError):
            await self._client.drain(self._worker_id)

    async def shutdown(self) -> None:
        """Drain, wait for the work in hand, then deregister.

        Bounded by ``drain_timeout``: a job that never finishes must not hold a
        container hostage forever, and the control plane reclaims it through
        lease expiry anyway.
        """
        self._stopping.set()
        await self.request_drain()
        await self._await_idle()
        if self._worker_id is not None:
            with contextlib.suppress(ControlPlaneError):
                await self._client.deregister(self._worker_id)
        self._status = WorkerStatus.OFFLINE

    def stop(self) -> None:
        self._stopping.set()

    # -- occupancy ------------------------------------------------------
    def job_started(self) -> None:
        self._active_jobs += 1

    def job_finished(self) -> None:
        self._active_jobs = max(self._active_jobs - 1, 0)

    @property
    def is_idle(self) -> bool:
        return self._active_jobs == 0

    async def _await_idle(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._drain_timeout
        while not self.is_idle and loop.time() < deadline:
            with contextlib.suppress(ControlPlaneError):
                await self.beat_once()
            await asyncio.sleep(min(self._interval, 2.0))
        if not self.is_idle:
            _log.warning(
                "leaving with %d job(s) still running; the control plane will "
                "reclaim them when their lease expires",
                self._active_jobs,
            )

    async def _wait(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
