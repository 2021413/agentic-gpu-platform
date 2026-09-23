"""Project endpoints."""

from __future__ import annotations

import httpx
from tests.api.conftest import Harness, create_project, create_run

from domain.enums import FailureKind


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


# ---------------------------------------------------------------------------
# correcting a toolchain
# ---------------------------------------------------------------------------
async def _create(client: httpx.AsyncClient, **toolchain: object) -> str:
    response = await client.post(
        "/v1/projects",
        json={
            "name": "correctable",
            "local_path": "/projects/current",
            "toolchain": {"language": "python", **toolchain},
        },
    )
    assert response.status_code == 201, response.text
    project_id: str = response.json()["id"]
    return project_id


async def test_a_toolchain_can_be_corrected_without_losing_the_project(
    client: httpx.AsyncClient,
) -> None:
    """The point of the route: same project, same id, different commands.

    Creating a second project under another name would also have worked, and
    would have left every previous run of this one behind.
    """
    project_id = await _create(client, test_command="pytest")

    response = await client.put(
        f"/v1/projects/{project_id}/toolchain",
        json={"language": "python", "build_command": "make", "test_command": "make test"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["id"] == project_id
    assert response.json()["toolchain"]["test_command"] == "make test"

    stored = await client.get(f"/v1/projects/{project_id}")
    assert stored.json()["toolchain"] == {
        "language": "python",
        "build_command": "make",
        "test_command": "make test",
        "static_analysis_command": None,
        "install_command": None,
        "working_subdirectory": None,
    }


async def test_replacing_a_toolchain_drops_the_commands_left_out(
    client: httpx.AsyncClient,
) -> None:
    """PUT is a replacement, not a merge: an omitted command is removed.

    The alternative — treating an absent field as "keep what you had" — leaves
    no way at all to say "this project has no test command any more", which is
    exactly the correction someone makes after configuring one by mistake.
    """
    project_id = await _create(client, build_command="make", test_command="pytest")

    response = await client.put(
        f"/v1/projects/{project_id}/toolchain", json={"language": "python"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["toolchain"]["build_command"] is None
    assert response.json()["toolchain"]["test_command"] is None


async def test_a_toolchain_cannot_be_changed_while_a_run_is_in_flight(
    client: httpx.AsyncClient,
) -> None:
    """A run validates against these commands; it must not see two versions."""
    project_id = await _create(client, test_command="pytest")
    run = await create_run(client, project_id)
    assert run.status_code == 202, run.text

    response = await client.put(
        f"/v1/projects/{project_id}/toolchain",
        json={"language": "python", "test_command": "pytest -x"},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "project_not_modifiable"
    # The refusal names what to wait for: "later" is not something a caller
    # can act on.
    assert body["details"]["active_run_ids"] == [run.json()["id"]]

    unchanged = await client.get(f"/v1/projects/{project_id}")
    assert unchanged.json()["toolchain"]["test_command"] == "pytest"


async def test_a_toolchain_can_be_changed_once_every_run_is_over(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    """The refusal is about *live* runs, not about a project that ever ran."""
    project_id = await _create(client, test_command="pytest")
    await create_run(client, project_id)
    for run in harness.store.runs.items.values():
        run.fail(now=harness.clock.now(), kind=FailureKind.INFRASTRUCTURE, reason="ended")

    response = await client.put(
        f"/v1/projects/{project_id}/toolchain",
        json={"language": "python", "test_command": "pytest -x"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["toolchain"]["test_command"] == "pytest -x"


async def test_the_toolchain_route_cannot_reach_a_projects_identity(
    client: httpx.AsyncClient,
) -> None:
    """``local_path`` and friends are not commands: they say which code this is.

    They are absent from the payload rather than merely ignored, so an attempt
    to smuggle one in is a 422 instead of a change the caller believes it made.
    """
    project_id = await _create(client, test_command="pytest")

    for smuggled in ({"local_path": "/elsewhere"}, {"default_branch": "prod"}, {"name": "other"}):
        response = await client.put(
            f"/v1/projects/{project_id}/toolchain", json={"language": "python", **smuggled}
        )
        assert response.status_code == 422, smuggled
        assert response.json()["code"] == "validation_error"

    stored = await client.get(f"/v1/projects/{project_id}")
    assert stored.json()["default_branch"] == "main"
    assert stored.json()["name"] == "correctable"


async def test_replacing_the_toolchain_of_an_unknown_project_is_a_problem_document(
    client: httpx.AsyncClient,
) -> None:
    response = await client.put(
        "/v1/projects/6b8f4c5e-0000-4000-8000-000000000000/toolchain",
        json={"language": "python"},
    )

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
