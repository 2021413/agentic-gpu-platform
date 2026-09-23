"""The same run, driven entirely over HTTP (spec section 45).

Nothing is mocked here beyond the model: this goes through the real ASGI app,
the real routes, the real use cases, the real orchestrator, real PostgreSQL and
the real worker registry. It is the scenario the spec's acceptance criteria
describe, and the one that proves the layers written separately fit together.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.e2e.test_full_workflow import prepare_schema, settle

from bootstrap.app import _dependencies
from bootstrap.container import Container
from domain.value_objects.identifiers import RunId
from interfaces.api.app import create_api

pytestmark = pytest.mark.e2e

SERVICE_HEADERS = {"authorization": "Bearer e2e-token"}


@pytest.fixture
async def client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    """The real application, without the background loops.

    The executor is driven step by step by the test instead: a background poller
    would make assertions race against it, and the point here is the HTTP
    surface, not the timing.
    """
    await prepare_schema(container)
    app = create_api(dependencies=_dependencies(container))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as http:
        yield http


async def register_worker(client: httpx.AsyncClient, *, concurrency: int = 2) -> dict:
    response = await client.post(
        "/internal/workers/register",
        headers=SERVICE_HEADERS,
        json={
            "endpoint": f"http://fake-worker-{concurrency}:8000",
            "model_id": "fake-model",
            "context_length": 32768,
            "max_concurrency": concurrency,
            "supported_roles": ["PLANNER", "CODER", "REVIEWER"],
        },
    )
    assert response.status_code in (200, 201), response.text
    return response.json()


def _tracked_files(repo: Path) -> list[tuple[str, tuple[str, bytes]]]:
    """Everything in the repository except its `.git`, read off the event loop."""
    return [
        ("files", (str(path.relative_to(repo)), path.read_bytes()))
        for path in sorted(repo.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(repo).parts
    ]


async def create_project(client: httpx.AsyncClient, repo: Path) -> dict:
    """Create the project the way the platform now does it: by upload.

    The JSON route used to accept a `local_path`, and every project in the
    database pointed at the same one. The files of the temporary repository
    are sent instead, and the server decides where they live.
    """
    files = await asyncio.to_thread(_tracked_files, repo)
    response = await client.post(
        "/v1/projects/upload",
        data={
            "name": f"http-{repo.name}",
            "language": "python",
            "build_command": "/bin/true",
            "test_command": "/bin/true",
        },
        files=files,
    )
    assert response.status_code in (200, 201), response.text
    return response.json()


async def create_run(client: httpx.AsyncClient, project_id: str, **body: object) -> dict:
    response = await client.post(
        f"/v1/projects/{project_id}/runs",
        json={"objective": "Implement the packet parser and add tests", **body},
    )
    assert response.status_code in (200, 201, 202), response.text
    return response.json()


# ----------------------------------------------------------------------
async def test_health_and_readiness_are_different_questions(
    client: httpx.AsyncClient,
) -> None:
    live = await client.get("/health")
    assert live.status_code == 200

    ready = await client.get("/ready")
    assert ready.status_code in (200, 503)
    body = ready.json()
    assert "ready" in body or "status" in body


async def test_a_run_created_over_http_completes(
    client: httpx.AsyncClient, container: Container, sample_repository: Path
) -> None:
    await register_worker(client)
    project = await create_project(client, sample_repository)
    run = await create_run(client, project["id"], candidate_count=1)

    await container.orchestrator.start(RunId.parse(run["id"]))
    await settle(container)

    detail = await client.get(f"/v1/runs/{run['id']}")
    assert detail.status_code == 200
    body = detail.json()
    payload = body.get("run", body)
    assert payload["status"] == "COMPLETED", payload.get("failure_reason")

    candidates = await client.get(f"/v1/runs/{run['id']}/candidates")
    assert candidates.status_code == 200
    assert candidates.json()["candidates"]


async def test_the_same_idempotency_key_returns_the_same_run(
    client: httpx.AsyncClient, sample_repository: Path
) -> None:
    project = await create_project(client, sample_repository)
    headers = {"idempotency-key": "http-retry-1"}

    first = await client.post(
        f"/v1/projects/{project['id']}/runs",
        json={"objective": "Add a retry to the client"},
        headers=headers,
    )
    second = await client.post(
        f"/v1/projects/{project['id']}/runs",
        json={"objective": "Add a retry to the client"},
        headers=headers,
    )
    assert first.json()["id"] == second.json()["id"]


async def test_an_unknown_run_is_a_problem_document(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 404
    assert body["title"]


async def test_a_malformed_body_is_rejected_with_details(
    client: httpx.AsyncClient, sample_repository: Path
) -> None:
    project = await create_project(client, sample_repository)
    response = await client.post(f"/v1/projects/{project['id']}/runs", json={"objective": ""})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_the_internal_api_refuses_an_unauthenticated_caller(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/internal/workers/register",
        json={
            "endpoint": "http://intruder:8000",
            "model_id": "m",
            "context_length": 1024,
        },
    )
    assert response.status_code in (401, 403)


async def test_the_worker_lifecycle_is_driven_over_http(
    client: httpx.AsyncClient,
) -> None:
    worker = await register_worker(client)
    worker_id = worker["id"]

    listed = await client.get("/v1/workers")
    assert listed.status_code == 200
    assert worker_id in {w["id"] for w in listed.json()["workers"]}

    beat = await client.post(
        f"/internal/workers/{worker_id}/heartbeat",
        headers=SERVICE_HEADERS,
        json={"active_jobs": 1, "queued_jobs": 0, "draining": False},
    )
    assert beat.status_code == 200
    assert beat.json()["active_jobs"] == 1

    drain = await client.post(
        f"/internal/workers/{worker_id}/drain", headers=SERVICE_HEADERS, json={}
    )
    assert drain.status_code == 200
    assert drain.json()["status"] == "DRAINING"

    removed = await client.delete(f"/internal/workers/{worker_id}", headers=SERVICE_HEADERS)
    assert removed.status_code in (200, 204)

    after = await client.get("/v1/workers")
    assert worker_id not in {w["id"] for w in after.json()["workers"]}


async def test_registering_twice_with_the_same_id_does_not_double_capacity(
    client: httpx.AsyncClient,
) -> None:
    first = await register_worker(client)
    again = await client.post(
        "/internal/workers/register",
        headers=SERVICE_HEADERS,
        json={
            "endpoint": first["endpoint"],
            "model_id": first["model_id"],
            "context_length": first["context_length"],
            "max_concurrency": first["capacity"],
            "worker_id": first["id"],
        },
    )
    assert again.status_code in (200, 201)

    listed = await client.get("/v1/workers")
    assert len(listed.json()["workers"]) == 1


async def test_cancelling_over_http_is_idempotent(
    client: httpx.AsyncClient, container: Container, sample_repository: Path
) -> None:
    await register_worker(client)
    project = await create_project(client, sample_repository)
    run = await create_run(client, project["id"], candidate_count=1)
    await container.orchestrator.start(RunId.parse(run["id"]))

    first = await client.post(f"/v1/runs/{run['id']}/cancel", json={"reason": "user"})
    second = await client.post(f"/v1/runs/{run['id']}/cancel", json={})
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "CANCELLED"


async def test_the_event_stream_replays_history_then_ends(
    client: httpx.AsyncClient, container: Container, sample_repository: Path
) -> None:
    """A client that connects after the fact must still see what happened."""
    await register_worker(client)
    project = await create_project(client, sample_repository)
    run = await create_run(client, project["id"], candidate_count=1)

    await container.orchestrator.start(RunId.parse(run["id"]))
    await settle(container)

    received: list[str] = []
    async with asyncio.timeout(20):
        async with client.stream(
            "GET", f"/v1/runs/{run['id']}/events", headers={"accept": "text/event-stream"}
        ) as stream:
            assert stream.status_code == 200
            async for line in stream.aiter_lines():
                if line.startswith("event:"):
                    received.append(line.split(":", 1)[1].strip())
                if "run.completed" in "".join(received):
                    break

    assert "run.created" in received
    assert "run.completed" in received
    assert not any("think" in name for name in received), "reasoning must never leak"
