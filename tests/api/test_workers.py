"""The internal worker API and the public view of the fleet."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from tests.api.conftest import SERVICE_TOKEN, Harness

REGISTRATION: dict[str, Any] = {
    "endpoint": "http://worker-1:8000",
    "model_id": "Qwen3-Coder-30B-A3B",
    "context_length": 262144,
    "max_concurrency": 4,
    "gpu": {"gpu_type": "H200", "gpu_count": 1},
}


async def register(
    client: httpx.AsyncClient, headers: dict[str, str], **overrides: Any
) -> httpx.Response:
    return await client.post(
        "/internal/workers/register", json={**REGISTRATION, **overrides}, headers=headers
    )


async def test_a_worker_registers_and_appears_in_the_pool(
    client: httpx.AsyncClient, service_headers: dict[str, str]
) -> None:
    response = await register(client, service_headers)

    assert response.status_code == 201
    worker = response.json()
    assert worker["model_id"] == "Qwen3-Coder-30B-A3B"
    assert worker["status"] == "READY"
    assert worker["capacity"] == 4
    assert worker["gpu_type"] == "H200"

    listed = await client.get("/v1/workers")
    assert listed.status_code == 200
    assert [w["id"] for w in listed.json()["workers"]] == [worker["id"]]


async def test_registering_the_same_id_twice_refreshes_one_worker(
    client: httpx.AsyncClient, service_headers: dict[str, str], harness: Harness
) -> None:
    """A restarted worker must not double the pool's apparent capacity."""
    worker_id = "6b8f4c5e-0000-4000-8000-000000000011"

    first = await register(client, service_headers, worker_id=worker_id)
    second = await register(client, service_headers, worker_id=worker_id, max_concurrency=8)

    assert first.json()["id"] == second.json()["id"] == worker_id
    assert len(harness.registry.workers) == 1
    assert second.json()["capacity"] == 8


async def test_a_heartbeat_reports_load(
    client: httpx.AsyncClient, service_headers: dict[str, str]
) -> None:
    worker_id = (await register(client, service_headers)).json()["id"]

    response = await client.post(
        f"/internal/workers/{worker_id}/heartbeat",
        json={"active_jobs": 2, "queued_jobs": 1},
        headers=service_headers,
    )

    assert response.status_code == 200
    assert response.json()["active_jobs"] == 2


async def test_a_heartbeat_from_an_unknown_worker_is_a_404(
    client: httpx.AsyncClient, service_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/internal/workers/6b8f4c5e-0000-4000-8000-0000000000aa/heartbeat",
        json={},
        headers=service_headers,
    )

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_draining_removes_a_worker_from_the_available_pool(
    client: httpx.AsyncClient, service_headers: dict[str, str]
) -> None:
    worker_id = (await register(client, service_headers)).json()["id"]

    drained = await client.post(f"/internal/workers/{worker_id}/drain", headers=service_headers)

    assert drained.status_code == 200
    assert drained.json()["status"] == "DRAINING"

    available = await client.get("/v1/workers", params={"only_available": True})
    assert available.json()["workers"] == []
    everything = await client.get("/v1/workers")
    assert len(everything.json()["workers"]) == 1


async def test_deregistering_is_idempotent(
    client: httpx.AsyncClient, service_headers: dict[str, str], harness: Harness
) -> None:
    """A retried deregistration during a rolling shutdown must not 404."""
    worker_id = (await register(client, service_headers)).json()["id"]

    first = await client.delete(f"/internal/workers/{worker_id}", headers=service_headers)
    second = await client.delete(f"/internal/workers/{worker_id}", headers=service_headers)

    assert first.status_code == 204
    assert second.status_code == 204
    assert harness.registry.workers == {}


async def test_worker_health_reports_the_control_planes_opinion(
    client: httpx.AsyncClient, service_headers: dict[str, str]
) -> None:
    worker_id = (await register(client, service_headers)).json()["id"]

    response = await client.get(f"/internal/workers/{worker_id}/health", headers=service_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == worker_id
    assert body["live"] is True
    assert body["accepts_new_jobs"] is True
    assert body["capacity"] == 4


async def test_health_of_an_unknown_worker_is_a_404(
    client: httpx.AsyncClient, service_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/internal/workers/6b8f4c5e-0000-4000-8000-0000000000bb/health", headers=service_headers
    )

    assert response.status_code == 404


async def test_an_invalid_endpoint_is_refused_before_reaching_the_domain(
    client: httpx.AsyncClient, service_headers: dict[str, str]
) -> None:
    response = await register(client, service_headers, endpoint="worker-1:8000")

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/internal/workers/register", "POST"),
        ("/internal/workers/6b8f4c5e-0000-4000-8000-000000000011/heartbeat", "POST"),
        ("/internal/workers/6b8f4c5e-0000-4000-8000-000000000011/drain", "POST"),
        ("/internal/workers/6b8f4c5e-0000-4000-8000-000000000011", "DELETE"),
        ("/internal/workers/6b8f4c5e-0000-4000-8000-000000000011/health", "GET"),
    ],
)
async def test_the_internal_api_refuses_anonymous_callers(
    client: httpx.AsyncClient, path: str, method: str
) -> None:
    response = await client.request(method, path, json={})

    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"
    assert response.headers["www-authenticate"].startswith("Bearer")


async def test_a_wrong_token_is_forbidden_not_unauthenticated(
    client: httpx.AsyncClient,
) -> None:
    """403 says 'the credential was read and rejected': resending it is pointless."""
    response = await register(client, {"Authorization": "Bearer not-the-token"})

    assert response.status_code == 403
    assert response.json()["code"] == "permission_denied"


async def test_the_token_is_also_accepted_from_the_dedicated_header(
    client: httpx.AsyncClient,
) -> None:
    response = await register(client, {"X-Service-Token": SERVICE_TOKEN})

    assert response.status_code == 201


async def test_the_public_api_needs_no_service_token(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/workers")

    assert response.status_code == 200
