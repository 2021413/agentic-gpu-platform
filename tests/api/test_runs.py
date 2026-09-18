"""Run endpoints: creation, idempotency, reads and cancellation."""

from __future__ import annotations

import httpx
from tests.api.conftest import Harness, create_project, create_run


async def test_creating_a_run_is_accepted_not_completed(client: httpx.AsyncClient) -> None:
    """202: the run is durable and scheduled; nothing waits behind inference."""
    project_id = await create_project(client)

    response = await create_run(client, project_id, candidate_count=2)

    assert response.status_code == 202
    body = response.json()
    assert body["project_id"] == project_id
    assert body["status"] == "CREATED"
    assert body["candidate_count"] == 2
    assert body["completed_at"] is None


async def test_replaying_an_idempotency_key_returns_the_same_run(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    """At-least-once delivery: a retried create must not start a second run."""
    project_id = await create_project(client)

    first = await create_run(client, project_id, idempotency_key="retry-me")
    second = await create_run(client, project_id, idempotency_key="retry-me")

    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    assert len(harness.store.runs.items) == 1


async def test_a_blank_idempotency_key_is_refused(client: httpx.AsyncClient) -> None:
    project_id = await create_project(client)

    response = await client.post(
        f"/v1/projects/{project_id}/runs",
        json={"objective": "do the thing"},
        headers={"Idempotency-Key": ""},
    )

    assert response.status_code == 422


async def test_creating_a_run_on_an_unknown_project_is_a_404(client: httpx.AsyncClient) -> None:
    response = await create_run(client, "6b8f4c5e-0000-4000-8000-000000000000")

    assert response.status_code == 404
    assert response.json()["details"]["entity"] == "Project"


async def test_an_empty_objective_is_refused(client: httpx.AsyncClient) -> None:
    project_id = await create_project(client)

    response = await create_run(client, project_id, objective="")

    assert response.status_code == 422


async def test_fetching_an_unknown_run_returns_a_conforming_problem_document(
    client: httpx.AsyncClient,
) -> None:
    unknown = "6b8f4c5e-0000-4000-8000-0000000000ff"

    response = await client.get(f"/v1/runs/{unknown}")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["type"].endswith("/not_found")
    assert body["title"]
    assert body["status"] == 404
    assert body["detail"] == "Run not found"
    assert body["instance"] == f"/v1/runs/{unknown}"
    assert body["code"] == "not_found"
    assert body["details"] == {"entity": "Run", "id": unknown}
    assert body["request_id"] == response.headers["x-request-id"]


async def test_a_malformed_identifier_is_a_validation_problem(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/v1/runs/not-a-uuid")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"]["errors"][0]["location"] == ["path", "run_id"]


async def test_a_run_can_be_read_in_detail(client: httpx.AsyncClient) -> None:
    project_id = await create_project(client)
    run_id = (await create_run(client, project_id)).json()["id"]

    response = await client.get(f"/v1/runs/{run_id}", params={"detailed": True})

    assert response.status_code == 200
    body = response.json()
    assert body["run"]["id"] == run_id
    assert body["plan"] is None
    assert body["candidates"] == []


async def test_cancelling_a_run_is_idempotent(client: httpx.AsyncClient) -> None:
    """The caller's intent is 'this must not continue'; twice is still once."""
    project_id = await create_project(client)
    run_id = (await create_run(client, project_id)).json()["id"]

    first = await client.post(f"/v1/runs/{run_id}/cancel", json={"reason": "changed my mind"})
    second = await client.post(f"/v1/runs/{run_id}/cancel")

    assert first.status_code == 200
    assert first.json()["status"] == "CANCELLED"
    assert second.status_code == 200
    assert second.json()["status"] == "CANCELLED"
    assert second.json()["id"] == run_id


async def test_cancelling_an_unknown_run_is_a_404(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/runs/6b8f4c5e-0000-4000-8000-00000000000a/cancel")

    assert response.status_code == 404


async def test_candidates_of_a_fresh_run_are_empty(client: httpx.AsyncClient) -> None:
    project_id = await create_project(client)
    run_id = (await create_run(client, project_id)).json()["id"]

    response = await client.get(f"/v1/runs/{run_id}/candidates")

    assert response.status_code == 200
    assert response.json() == {"candidates": []}


async def test_candidates_of_an_unknown_run_are_a_404_not_an_empty_list(
    client: httpx.AsyncClient,
) -> None:
    """An empty collection would claim the run exists."""
    response = await client.get("/v1/runs/6b8f4c5e-0000-4000-8000-00000000000b/candidates")

    assert response.status_code == 404
