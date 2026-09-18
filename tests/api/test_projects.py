"""Project endpoints."""

from __future__ import annotations

import httpx
from tests.api.conftest import create_project


async def test_creating_a_project_returns_its_identity(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/projects",
        json={
            "name": "demo",
            "repository_url": "https://example.invalid/demo.git",
            "toolchain": {"language": "python", "test_command": "pytest -q"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "demo"
    assert body["default_branch"] == "main"
    assert body["language"] == "python"
    assert body["id"]


async def test_creating_the_same_project_twice_does_not_duplicate_it(
    client: httpx.AsyncClient,
) -> None:
    """Project creation is idempotent on the name, so a retry is safe."""
    first = await create_project(client, name="same")
    second = await create_project(client, name="same")

    assert first == second
    listing = await client.get("/v1/projects")
    assert [p["id"] for p in listing.json()] == [first]


async def test_a_project_without_a_source_is_rejected_at_the_boundary(
    client: httpx.AsyncClient,
) -> None:
    """A domain invariant re-stated in the schema: 422, never a 500."""
    response = await client.post("/v1/projects", json={"name": "sourceless"})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "validation_error"


async def test_unknown_fields_are_refused(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/projects",
        json={"name": "x", "local_path": "/tmp/x", "surprise": True},
    )

    assert response.status_code == 422


async def test_fetching_an_unknown_project_is_a_problem_document(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/v1/projects/6b8f4c5e-0000-4000-8000-000000000000")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_listing_projects_is_paginated(client: httpx.AsyncClient) -> None:
    for index in range(3):
        await create_project(client, name=f"project-{index}")

    page = await client.get("/v1/projects", params={"limit": 2, "offset": 0})
    assert page.status_code == 200
    assert len(page.json()) == 2

    rejected = await client.get("/v1/projects", params={"limit": 0})
    assert rejected.status_code == 422
