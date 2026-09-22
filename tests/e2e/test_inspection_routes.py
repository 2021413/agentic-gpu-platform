"""What the operator can actually see (spec section 20).

A run spends GPU money writing code into a repository. Until now the API told
you a candidate had touched three files and moved five lines, and stopped
there: the patch was persisted and unreadable, the reviewer's findings were
persisted and unreachable, and there was no way to list the runs of a project
at all — a dashboard reloading the page had nothing to show.

These are the reads a human needs to trust the thing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.e2e.test_autonomous_workflow import ReadyProbe, advance
from tests.e2e.test_full_workflow import prepare_schema
from tests.e2e.test_http_workflow import create_project, create_run

from bootstrap.app import _dependencies
from bootstrap.container import Container
from domain.enums import AgentRole
from interfaces.api.app import create_api
from worker_agent.agent import WorkerAgent, WorkerDescription
from worker_agent.client import ControlPlaneClient

pytestmark = pytest.mark.e2e

SERVICE_TOKEN = "e2e-token"


@pytest.fixture
async def client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    await prepare_schema(container)
    app = create_api(dependencies=_dependencies(container))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    ) as http:
        yield http


@pytest.fixture
async def worker(client: httpx.AsyncClient) -> str:
    agent = WorkerAgent(
        client=ControlPlaneClient(
            base_url="http://api", service_token=SERVICE_TOKEN, client=client
        ),
        probe=ReadyProbe(),
        description=WorkerDescription(
            endpoint="http://gpu.invalid:8000",
            model_id="fake-model",
            context_length=32768,
            max_concurrency=2,
            roles=frozenset({AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER}),
        ),
    )
    return await agent.start()


async def completed_run(
    client: httpx.AsyncClient, container: Container, repo: Path
) -> tuple[str, str]:
    project = await create_project(client, repo)
    run = await create_run(client, project["id"], candidate_count=1)
    await advance(container)
    return project["id"], run["id"]


# ----------------------------------------------------------------------
async def test_a_project_lists_its_runs(
    client: httpx.AsyncClient, container: Container, worker: str, sample_repository: Path
) -> None:
    """Without this a dashboard is empty on reload: nothing remembers the ids."""
    project_id, run_id = await completed_run(client, container, sample_repository)

    response = await client.get(f"/v1/projects/{project_id}/runs")

    assert response.status_code == 200, response.text
    runs = response.json()["runs"]
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["status"] == "COMPLETED"
    assert runs[0]["objective"]


async def test_the_run_list_is_newest_first_and_bounded(
    client: httpx.AsyncClient, container: Container, worker: str, sample_repository: Path
) -> None:
    project = await create_project(client, sample_repository)
    created = [(await create_run(client, project["id"], candidate_count=1))["id"] for _ in range(3)]
    await advance(container)

    response = await client.get(f"/v1/projects/{project['id']}/runs", params={"limit": 2})

    assert response.status_code == 200, response.text
    runs = response.json()["runs"]
    assert len(runs) == 2
    assert [r["id"] for r in runs] == created[::-1][:2], "the newest run must come first"


async def test_an_unknown_project_listing_runs_is_a_problem_document(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/v1/projects/6f1d5f2e-0000-0000-0000-000000000000/runs")

    assert response.status_code == 404
    assert response.json()["type"].startswith("/problems/")


async def test_the_patch_a_candidate_produced_can_be_read(
    client: httpx.AsyncClient, container: Container, worker: str, sample_repository: Path
) -> None:
    """The code the agents wrote, in full. It was persisted and unreadable."""
    _, run_id = await completed_run(client, container, sample_repository)
    candidate = (await client.get(f"/v1/runs/{run_id}/candidates")).json()["candidates"][0]

    response = await client.get(f"/v1/runs/{run_id}/candidates/{candidate['id']}/diff")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diff"], "the patch came back empty"
    assert "diff --git" in body["diff"] or "@@" in body["diff"]
    assert body["changed_files"] == candidate["changed_files"]
    assert body["total_churn"] == candidate["total_churn"]


async def test_an_unknown_candidate_diff_is_a_problem_document(
    client: httpx.AsyncClient, container: Container, worker: str, sample_repository: Path
) -> None:
    _, run_id = await completed_run(client, container, sample_repository)

    response = await client.get(
        f"/v1/runs/{run_id}/candidates/6f1d5f2e-0000-0000-0000-000000000000/diff"
    )

    assert response.status_code == 404


async def test_the_reviewer_findings_are_reachable(
    client: httpx.AsyncClient, container: Container, worker: str, sample_repository: Path
) -> None:
    """Persisted since the first day, never exposed."""
    _, run_id = await completed_run(client, container, sample_repository)

    response = await client.get(f"/v1/runs/{run_id}/reviews")

    assert response.status_code == 200, response.text
    reviews = response.json()["reviews"]
    assert reviews, "a completed run reviewed at least one candidate"
    first = reviews[0]
    assert first["verdict"] in ("PASS", "FAIL")
    assert first["candidate_id"]
    assert first["iteration"] >= 1
    assert "findings" in first
