"""The worker-side lifecycle, driven against a fake control plane.

No GPU, no container, no network: the agent talks to an in-memory double, which
is enough to pin down the behaviours that actually matter — never advertise an
engine that cannot serve, re-register when the control plane forgets you, and
leave without stranding work.
"""

from __future__ import annotations

from typing import Any

import pytest

from domain.enums import AgentRole, WorkerStatus
from worker_agent.agent import WorkerAgent, WorkerDescription
from worker_agent.client import ControlPlaneError

DESCRIPTION = WorkerDescription(
    endpoint="http://worker:8000",
    model_id="Qwen3-Coder-30B-A3B",
    context_length=262_144,
    max_concurrency=4,
    roles=frozenset(AgentRole),
    gpu_type="H200",
)


class FakeControlPlane:
    def __init__(self, *, forget_after: int | None = None) -> None:
        self.registrations: list[dict[str, Any]] = []
        self.heartbeats: list[dict[str, Any]] = []
        self.drained: list[str] = []
        self.deregistered: list[str] = []
        self._forget_after = forget_after

    async def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.registrations.append(payload)
        return {"id": f"worker-{len(self.registrations)}"}

    async def heartbeat(self, worker_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.heartbeats.append({"worker_id": worker_id, **payload})
        if self._forget_after is not None and len(self.heartbeats) > self._forget_after:
            raise ControlPlaneError("unknown worker", status_code=404)
        return {}

    async def drain(self, worker_id: str) -> dict[str, Any]:
        self.drained.append(worker_id)
        return {}

    async def deregister(self, worker_id: str) -> None:
        self.deregistered.append(worker_id)


class FakeProbe:
    def __init__(self, *, healthy: bool = True, ready: bool = True) -> None:
        self.healthy = healthy
        self.ready = ready

    async def is_healthy(self) -> bool:
        return self.healthy

    async def wait_until_ready(self, *, timeout_seconds: float = 900.0, **_: object) -> bool:
        return self.ready


def build(control_plane: FakeControlPlane, probe: FakeProbe) -> WorkerAgent:
    return WorkerAgent(
        client=control_plane,  # type: ignore[arg-type]
        probe=probe,  # type: ignore[arg-type]
        description=DESCRIPTION,
        heartbeat_interval_seconds=0.01,
        drain_timeout_seconds=0.05,
    )


async def test_registration_advertises_the_declared_capabilities() -> None:
    plane = FakeControlPlane()
    agent = build(plane, FakeProbe())

    worker_id = await agent.start()

    assert worker_id == "worker-1"
    assert agent.status is WorkerStatus.READY
    payload = plane.registrations[0]
    assert payload["model_id"] == "Qwen3-Coder-30B-A3B"
    assert payload["max_concurrency"] == 4
    assert set(payload["supported_roles"]) == {"PLANNER", "CODER", "REVIEWER"}
    assert payload["gpu"]["gpu_type"] == "H200"


async def test_a_worker_never_registers_before_its_engine_can_serve() -> None:
    """Advertising capacity that does not exist wastes the first job scheduled."""
    plane = FakeControlPlane()
    agent = build(plane, FakeProbe(ready=False))

    with pytest.raises(ControlPlaneError, match="never became ready"):
        await agent.start()
    assert plane.registrations == []


async def test_heartbeats_report_the_real_occupancy() -> None:
    plane = FakeControlPlane()
    agent = build(plane, FakeProbe())
    await agent.start()

    agent.job_started()
    agent.job_started()
    await agent.beat_once()

    assert plane.heartbeats[-1]["load"]["active_jobs"] == 2
    agent.job_finished()
    await agent.beat_once()
    assert plane.heartbeats[-1]["load"]["active_jobs"] == 1


async def test_an_unhealthy_engine_is_reported_rather_than_hidden() -> None:
    plane = FakeControlPlane()
    probe = FakeProbe()
    agent = build(plane, probe)
    await agent.start()

    probe.healthy = False
    assert await agent.beat_once() is False
    assert agent.status is WorkerStatus.UNHEALTHY

    probe.healthy = True
    assert await agent.beat_once() is True
    assert agent.status is WorkerStatus.READY


async def test_a_forgotten_worker_registers_again() -> None:
    """A restarted control plane must not cost the pool a healthy GPU."""
    plane = FakeControlPlane(forget_after=1)
    agent = build(plane, FakeProbe())
    await agent.start()

    await agent.beat_once()
    await agent.beat_once()

    assert len(plane.registrations) == 2
    assert agent.worker_id == "worker-2"


async def test_shutdown_drains_then_deregisters() -> None:
    plane = FakeControlPlane()
    agent = build(plane, FakeProbe())
    worker_id = await agent.start()

    await agent.shutdown()

    assert plane.drained == [worker_id]
    assert plane.deregistered == [worker_id]
    assert agent.status is WorkerStatus.OFFLINE


async def test_shutdown_gives_up_after_the_drain_timeout() -> None:
    """A job that never ends must not hold the container hostage."""
    plane = FakeControlPlane()
    agent = build(plane, FakeProbe())
    await agent.start()
    agent.job_started()

    await agent.shutdown()

    assert agent.status is WorkerStatus.OFFLINE
    assert plane.deregistered, "the worker must still leave the pool"


async def test_draining_is_visible_in_the_heartbeat() -> None:
    plane = FakeControlPlane()
    agent = build(plane, FakeProbe())
    await agent.start()

    await agent.request_drain()
    await agent.beat_once()

    assert agent.status is WorkerStatus.DRAINING
    assert plane.heartbeats[-1]["draining"] is True
