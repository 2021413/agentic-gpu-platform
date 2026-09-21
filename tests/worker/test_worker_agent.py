"""The worker-side lifecycle, driven against a fake control plane.

No GPU, no container, no network: the agent talks to an in-memory double, which
is enough to pin down the behaviours that actually matter — never advertise an
engine that cannot serve, re-register when the control plane forgets you, and
leave without stranding work.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx
import pytest

from domain.enums import AgentRole, WorkerStatus
from worker_agent.agent import WorkerAgent, WorkerDescription
from worker_agent.client import ControlPlaneClient, ControlPlaneError
from worker_agent.inference import InferenceProbe

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


class FakeProbe(InferenceProbe):
    """Stands in for the GPU server, and only for that.

    Subclassed rather than duck-typed so it cannot drift from the real probe's
    surface: when `served_context_length` was added, every test using this
    class failed loudly instead of the double quietly not having it.
    """

    def __init__(self, *, healthy: bool = True, ready: bool = True) -> None:
        super().__init__(base_url="http://inference.invalid")
        self.healthy = healthy
        self.ready = ready

    async def is_healthy(self) -> bool:
        return self.healthy

    async def wait_until_ready(
        self, *, timeout_seconds: float = 900.0, poll_seconds: float = 3.0
    ) -> bool:
        return self.ready

    async def served_context_length(self) -> int | None:
        """Declines to say, so the declared value is kept — see ServedLengthProbe."""
        return None


class ServedLengthProbe(FakeProbe):
    """Healthy, and says what the engine serves — or declines to say."""

    def __init__(self, served: int | None) -> None:
        super().__init__()
        self._served = served

    async def served_context_length(self) -> int | None:
        return self._served


def build(
    control_plane: FakeControlPlane,
    probe: FakeProbe,
    *,
    context_length: int | None = None,
) -> WorkerAgent:
    description = DESCRIPTION
    if context_length is not None:
        description = replace(DESCRIPTION, context_length=context_length)
    return WorkerAgent(
        client=control_plane,  # type: ignore[arg-type]
        probe=probe,
        description=description,
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

    assert plane.heartbeats[-1]["active_jobs"] == 2
    agent.job_finished()
    await agent.beat_once()
    assert plane.heartbeats[-1]["active_jobs"] == 1


async def test_the_heartbeat_payload_matches_what_the_api_accepts() -> None:
    """The HTTP schema is flat and forbids unknown fields.

    The agent used to nest the counters under "load", mirroring the domain
    command, and every heartbeat was rejected with a 422 — which registration,
    agreeing on its own shape, never revealed. The worker then expired from the
    registry 90 seconds later and jobs failed with "no compatible worker".
    """
    plane = FakeControlPlane()
    agent = build(plane, FakeProbe())
    await agent.start()
    agent.job_started()

    await agent.beat_once()

    sent = plane.heartbeats[-1]
    assert set(sent) == {"worker_id", "active_jobs", "queued_jobs", "draining"}
    assert sent["active_jobs"] == 1
    assert sent["draining"] is False


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


async def test_the_service_token_is_sent_even_when_the_client_is_injected() -> None:
    """An injected client used to drop the token silently, answering 401.

    ``ControlPlaneClient`` put the Authorization header on the client it built
    itself, so passing one in — which is how a caller shares a connection pool
    or drives the app in process — took the token as an argument and then never
    used it. Nothing said so; every call simply came back unauthenticated.
    """
    seen: list[httpx.Headers] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={"id": "worker-1"})

    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(record), base_url="http://control-plane"
    )
    async with ControlPlaneClient(
        base_url="http://control-plane", service_token="s3cret", client=injected
    ) as plane:
        await plane.register({"endpoint": "http://gpu:8000"})
        await plane.heartbeat("worker-1", {"active_jobs": 0})

    assert [headers.get("authorization") for headers in seen] == [
        "Bearer s3cret",
        "Bearer s3cret",
    ]


async def test_an_empty_token_sends_no_authorization_header() -> None:
    """Local runs accept unauthenticated internal calls; an empty token must
    not become the literal string ``Bearer ``."""
    seen: list[httpx.Headers] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={})

    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(record), base_url="http://control-plane"
    )
    async with ControlPlaneClient(
        base_url="http://control-plane", service_token="", client=injected
    ) as plane:
        await plane.drain("worker-1")

    assert "authorization" not in seen[0]


# ---------------------------------------------------------------------------
# What the worker advertises.
#
# The agent declared `context_length` from its own settings, which default to
# the model's *native* 262144. vLLM is started with MAX_MODEL_LEN, 16384 by
# default. Nothing reconciled the two, so the control plane believed every
# worker had sixteen times the room it had, and `fits` waved through prompts
# the engine answers with a 400.
# ---------------------------------------------------------------------------


def models_response(max_model_len: int) -> httpx.Response:
    """The shape vLLM's /v1/models actually returns."""
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {
                    "id": "qwen3-coder",
                    "object": "model",
                    "owned_by": "vllm",
                    "max_model_len": max_model_len,
                }
            ],
        },
    )


async def test_the_probe_reads_the_context_length_the_engine_serves() -> None:
    probe = InferenceProbe(
        base_url="http://gpu:8000",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: models_response(16_384)),
            base_url="http://gpu:8000",
        ),
    )

    assert await probe.served_context_length() == 16_384


async def test_an_engine_that_does_not_say_is_not_guessed_at() -> None:
    """Silence must read as "unknown", never as a comfortable default."""
    probe = InferenceProbe(
        base_url="http://gpu:8000",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"object": "list", "data": [{"id": "m"}]})
            ),
            base_url="http://gpu:8000",
        ),
    )

    assert await probe.served_context_length() is None


async def test_registration_advertises_what_the_engine_serves_not_what_was_configured() -> None:
    """The defect: 262144 declared, 16384 served, nothing reconciling them."""
    plane = FakeControlPlane()
    agent = build(plane, ServedLengthProbe(16_384), context_length=262_144)

    await agent.start()

    assert plane.registrations[-1]["context_length"] == 16_384


async def test_an_engine_that_does_not_say_leaves_the_declared_value_alone() -> None:
    plane = FakeControlPlane()
    agent = build(plane, ServedLengthProbe(None), context_length=262_144)

    await agent.start()

    assert plane.registrations[-1]["context_length"] == 262_144
