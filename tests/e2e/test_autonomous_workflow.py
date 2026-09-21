"""The run nobody nudges, and the worker that registers itself.

Every other end-to-end test calls ``container.orchestrator.start(run.id)`` by
hand, and every worker-agent test posts to a ``FakeControlPlane`` written by the
author of the agent. Both shortcuts hid a real defect:

* a run created over HTTP sat in ``CREATED`` forever, because nothing in the
  deployed system ever called ``start``;
* the agent's heartbeat body was rejected with a 422 by the real route, because
  the double accepted the shape the agent happened to send.

So nothing here is nudged and nothing here is doubled except the GPU. The run is
created over HTTP and advanced only by the loops that actually run in
production; the worker registers and beats through the real ``WorkerAgent``
against the real ASGI app.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.e2e.test_full_workflow import prepare_schema
from tests.e2e.test_http_workflow import create_project, create_run

from bootstrap.app import _dependencies
from bootstrap.container import Container
from domain.enums import AgentRole, WorkerStatus
from interfaces.api.app import create_api
from worker_agent.agent import WorkerAgent, WorkerDescription
from worker_agent.client import ControlPlaneClient
from worker_agent.inference import InferenceProbe

pytestmark = pytest.mark.e2e

SERVICE_TOKEN = "e2e-token"
SETTLE_LIMIT = 200


class ReadyProbe(InferenceProbe):
    """The one thing that cannot be real here: the GPU server itself.

    Subclassed rather than faked wholesale so the agent still receives the type
    it expects, and so a change to the probe's surface breaks this test.
    """

    def __init__(self) -> None:
        super().__init__(base_url="http://inference.invalid")

    async def is_healthy(self) -> bool:
        return True

    async def wait_until_ready(
        self, *, timeout_seconds: float = 900.0, poll_seconds: float = 3.0
    ) -> bool:
        return True

    async def active_requests(self) -> int | None:
        return None


@pytest.fixture
async def client(container: Container) -> AsyncIterator[httpx.AsyncClient]:
    await prepare_schema(container)
    app = create_api(dependencies=_dependencies(container))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as http:
        yield http


@pytest.fixture
async def agent(client: httpx.AsyncClient) -> AsyncIterator[WorkerAgent]:
    """The real worker agent, speaking to the real routes over ASGI.

    The client is injected rather than constructed from a URL so the requests go
    through the application in process — the routes, the auth dependency and the
    request schemas are all the deployed ones.
    """
    control_plane = ControlPlaneClient(
        base_url="http://api", service_token=SERVICE_TOKEN, client=client
    )
    built = WorkerAgent(
        client=control_plane,
        probe=ReadyProbe(),
        description=WorkerDescription(
            endpoint="http://gpu-worker.invalid:8000",
            model_id="fake-model",
            context_length=32768,
            max_concurrency=2,
            roles=frozenset({AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER}),
            gpu_type="H200",
        ),
        heartbeat_interval_seconds=0.01,
    )
    yield built


async def advance(container: Container, *, limit: int = SETTLE_LIMIT) -> int:
    """Run the production loops until nothing is left to do.

    ``maintenance.tick`` first, exactly as ``run_forever`` orders them: it is
    what promotes a freshly created run. Bounded, so a system that never settles
    fails the test instead of hanging it.
    """
    steps = 0
    while True:
        await container.maintenance.tick()
        if not await container.executor.run_once():
            return steps
        steps += 1
        if steps >= limit:
            raise AssertionError(f"the workflow never settled within {limit} jobs")


async def run_status(client: httpx.AsyncClient, run_id: str) -> dict[str, Any]:
    response = await client.get(f"/v1/runs/{run_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    return dict(body.get("run", body))


# ----------------------------------------------------------------------
async def test_a_run_created_over_http_advances_with_nobody_nudging_it(
    client: httpx.AsyncClient,
    container: Container,
    agent: WorkerAgent,
    sample_repository: Path,
) -> None:
    """The defect this exists for: a run that stayed in CREATED forever.

    ``orchestrator.start`` is deliberately never called. If the only thing that
    promotes a run is a test calling it by hand, the deployed system does not
    work, and this test must be the one that says so.
    """
    await agent.start()
    project = await create_project(client, sample_repository)
    created = await create_run(client, project["id"], candidate_count=1)

    assert (await run_status(client, created["id"]))["status"] == "CREATED"

    await advance(container)

    final = await run_status(client, created["id"])
    assert final["status"] == "COMPLETED", final.get("failure_reason")

    candidates = await client.get(f"/v1/runs/{created['id']}/candidates")
    assert candidates.status_code == 200, candidates.text
    winners = candidates.json()["candidates"]
    assert winners, "the run completed without producing a single candidate"

    # A completed run carrying an empty patch is the failure that matters most:
    # it reports success while having changed nothing. The coder now answers
    # with whole file contents and git derives the diff, so a real change must
    # show up as touched files and non-zero churn.
    selected = [c for c in winners if c["status"] == "SELECTED"]
    assert selected, f"nothing was selected: {winners}"
    assert selected[0]["changed_files"], f"the selected candidate changed nothing: {selected[0]}"
    assert selected[0]["total_churn"] > 0
    assert selected[0]["build_passed"] and selected[0]["tests_passed"]
    assert selected[0]["review_verdict"] == "PASS"


async def test_the_worker_agent_registers_and_beats_through_the_real_routes(
    client: httpx.AsyncClient, agent: WorkerAgent
) -> None:
    """The defect this exists for: a heartbeat body the real route answered 422.

    The agent's own tests posted to a double that accepted whatever the agent
    sent, so the schemas drifted apart silently. Here the body is validated by
    the deployed FastAPI model.
    """
    worker_id = await agent.start()
    assert worker_id

    assert await agent.beat_once() is True
    assert agent.status is WorkerStatus.READY

    # The public read-only view is the fleet's observation surface; mutating a
    # worker is what lives behind the service token.
    listing = await client.get("/v1/workers")
    assert listing.status_code == 200, listing.text
    workers = listing.json()["workers"]
    assert [w["id"] for w in workers] == [worker_id]
    assert workers[0]["status"] == "READY"


async def test_a_busy_agent_reports_its_load_and_a_drain_is_honoured(
    client: httpx.AsyncClient, agent: WorkerAgent
) -> None:
    """Capacity accounting crosses the same boundary the 422 broke."""
    worker_id = await agent.start()
    agent.job_started()
    agent.job_started()

    assert await agent.beat_once() is True

    await agent.request_drain()
    assert agent.status is WorkerStatus.DRAINING
    assert await agent.beat_once() is True

    listing = await client.get("/v1/workers")
    assert listing.status_code == 200, listing.text
    drained = next(w for w in listing.json()["workers"] if w["id"] == worker_id)
    assert drained["status"] == "DRAINING"
    assert drained["active_jobs"] == 2, drained

    # A draining worker must disappear from the schedulable pool, or the
    # orchestrator would keep aiming jobs at a worker on its way out.
    available = await client.get("/v1/workers", params={"only_available": True})
    assert available.status_code == 200, available.text
    assert worker_id not in [w["id"] for w in available.json()["workers"]]
