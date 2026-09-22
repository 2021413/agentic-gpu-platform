"""The merge waits for a human (spec section 34, extension).

Integration is a write into someone else's repository, and it used to happen
the moment the reviewer said PASS. With REQUIRE_APPROVAL on, the run stops at
AWAITING_APPROVAL with its winner selected and the repository untouched, until
somebody says yes.
"""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.e2e.test_autonomous_workflow import ReadyProbe, advance
from tests.e2e.test_full_workflow import prepare_schema
from tests.e2e.test_http_workflow import create_project, create_run

from bootstrap.app import _dependencies
from bootstrap.config import Settings
from bootstrap.container import Container, build_container
from domain.enums import AgentRole
from infrastructure.llm import FakeLLMProviderFactory
from interfaces.api.app import create_api
from worker_agent.agent import WorkerAgent, WorkerDescription
from worker_agent.client import ControlPlaneClient

pytestmark = pytest.mark.e2e

SERVICE_TOKEN = "e2e-token"


def head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
async def gated_container(e2e_settings: Settings) -> AsyncIterator[Container]:
    built = await build_container(
        e2e_settings.model_copy(update={"require_approval": True}),
        in_memory_messaging=True,
        llm_factory=FakeLLMProviderFactory(
            model_id=e2e_settings.model_id, context_length=e2e_settings.model_context_length
        ),
    )
    try:
        yield built
    finally:
        await built.aclose()


@pytest.fixture
async def client(gated_container: Container) -> AsyncIterator[httpx.AsyncClient]:
    await prepare_schema(gated_container)
    app = create_api(dependencies=_dependencies(gated_container))
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


async def waiting_run(client: httpx.AsyncClient, container: Container, repo: Path) -> str:
    project = await create_project(client, repo)
    run = await create_run(client, project["id"], candidate_count=1)
    await advance(container)
    return str(run["id"])


async def status_of(client: httpx.AsyncClient, run_id: str) -> str:
    body = (await client.get(f"/v1/runs/{run_id}")).json()
    return str(body.get("run", body)["status"])


# ----------------------------------------------------------------------
async def test_a_reviewed_run_stops_before_touching_the_repository(
    client: httpx.AsyncClient,
    gated_container: Container,
    worker: str,
    sample_repository: Path,
) -> None:
    before = head(sample_repository)

    run_id = await waiting_run(client, gated_container, sample_repository)

    assert await status_of(client, run_id) == "AWAITING_APPROVAL"
    assert head(sample_repository) == before, "the patch landed without anyone approving it"

    body = (await client.get(f"/v1/runs/{run_id}")).json()
    assert (body.get("run", body))["selected_candidate_id"], "the winner must already be chosen"


async def test_approving_lands_the_patch(
    client: httpx.AsyncClient,
    gated_container: Container,
    worker: str,
    sample_repository: Path,
) -> None:
    before = head(sample_repository)
    run_id = await waiting_run(client, gated_container, sample_repository)

    response = await client.post(f"/v1/runs/{run_id}/approve")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "COMPLETED"
    assert head(sample_repository) != before, "approval did not merge anything"


async def test_rejecting_sends_the_reason_back_to_the_coder(
    client: httpx.AsyncClient,
    gated_container: Container,
    worker: str,
    sample_repository: Path,
) -> None:
    before = head(sample_repository)
    run_id = await waiting_run(client, gated_container, sample_repository)

    response = await client.post(
        f"/v1/runs/{run_id}/reject", json={"reason": "the retry loop never closes the socket"}
    )

    assert response.status_code == 200, response.text
    assert head(sample_repository) == before, "a rejected patch must not land"
    assert await status_of(client, run_id) != "COMPLETED"


async def test_a_reason_is_required_to_reject(
    client: httpx.AsyncClient,
    gated_container: Container,
    worker: str,
    sample_repository: Path,
) -> None:
    """An empty reason spends a repair round to learn nothing."""
    run_id = await waiting_run(client, gated_container, sample_repository)

    response = await client.post(f"/v1/runs/{run_id}/reject", json={"reason": ""})

    assert response.status_code == 422, response.text


async def test_a_run_that_is_not_waiting_cannot_be_approved(
    client: httpx.AsyncClient,
    gated_container: Container,
    worker: str,
    sample_repository: Path,
) -> None:
    run_id = await waiting_run(client, gated_container, sample_repository)
    assert (await client.post(f"/v1/runs/{run_id}/approve")).status_code == 200

    again = await client.post(f"/v1/runs/{run_id}/approve")

    assert again.status_code == 409, again.text
    assert again.json()["type"].startswith("/problems/")
