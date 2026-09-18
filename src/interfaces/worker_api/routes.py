"""Internal worker API (spec sections 19 and 25).

This is the surface a GPU worker uses to join the platform, prove it is alive,
stop taking work and leave. Every route is behind service authentication,
declared once on the router so no handler can forget it.

The whole lifecycle is designed for at-least-once delivery: registering twice
with the same id refreshes one worker, deregistering a worker that already left
succeeds, and a heartbeat is naturally repeatable. A worker retrying a call it
never saw the answer to must never corrupt the fleet's view of itself.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Header, Query, Response, status

from domain.exceptions import EntityNotFoundError
from domain.value_objects.identifiers import WorkerId
from interfaces.api.dependencies.providers import (
    DeregisterWorkerDep,
    DrainWorkerDep,
    ListWorkersDep,
    RegisterWorkerDep,
    ServiceAuthenticatorDep,
    WorkerHeartbeatDep,
)
from interfaces.api.schemas.common import ProblemDetails
from interfaces.api.schemas.workers import (
    DrainWorkerRequest,
    HeartbeatRequest,
    RegisterWorkerRequest,
    WorkerHealthResponse,
    WorkerResponse,
    deregister_command,
)
from interfaces.worker_api.auth import ServicePrincipal, extract_service_token

__all__ = ["require_service_principal", "router"]


async def require_service_principal(
    authenticator: ServiceAuthenticatorDep,
    authorization: Annotated[str | None, Header(include_in_schema=False)] = None,
    x_service_token: Annotated[str | None, Header(include_in_schema=False)] = None,
) -> ServicePrincipal:
    """Extract the credential and let the authenticator decide.

    The decision is never taken here: this function only knows which headers
    carry a token, which is a transport detail, while what makes a token
    acceptable is the injected authenticator's business.
    """
    token = extract_service_token(authorization=authorization, service_token_header=x_service_token)
    return authenticator.authenticate(token)


router = APIRouter(
    prefix="/internal/workers",
    tags=["internal"],
    dependencies=[Depends(require_service_principal)],
    responses={
        401: {"model": ProblemDetails, "description": "No service token was supplied."},
        403: {"model": ProblemDetails, "description": "The service token was rejected."},
    },
)


@router.post(
    "/register",
    response_model=WorkerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a worker",
    description=(
        "Idempotent on `worker_id`: re-registering an existing id refreshes its "
        "capabilities instead of creating a twin that would double the pool's "
        "apparent capacity."
    ),
)
async def register_worker(
    payload: RegisterWorkerRequest, use_case: RegisterWorkerDep
) -> WorkerResponse:
    return WorkerResponse.of(await use_case.execute(payload.to_command()))


@router.post(
    "/{worker_id}/heartbeat",
    response_model=WorkerResponse,
    summary="Refresh liveness and report load",
    description=(
        "A heartbeat from a worker previously declared unavailable readmits it: "
        "a network partition must not permanently remove a healthy GPU."
    ),
    responses={404: {"model": ProblemDetails}},
)
async def heartbeat(
    worker_id: UUID, payload: HeartbeatRequest, use_case: WorkerHeartbeatDep
) -> WorkerResponse:
    return WorkerResponse.of(await use_case.execute(payload.to_command(worker_id=worker_id)))


@router.post(
    "/{worker_id}/drain",
    response_model=WorkerResponse,
    summary="Stop sending work to a worker",
    description="The worker finishes what it already holds and accepts nothing new.",
    responses={404: {"model": ProblemDetails}},
)
async def drain_worker(
    worker_id: UUID,
    use_case: DrainWorkerDep,
    payload: Annotated[DrainWorkerRequest, Body(default_factory=DrainWorkerRequest)],
) -> WorkerResponse:
    return WorkerResponse.of(await use_case.execute(payload.to_command(worker_id=worker_id)))


@router.delete(
    "/{worker_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deregister a worker",
    description=(
        "Idempotent, including for a worker that is already gone: a retried "
        "deregistration must not turn into a 404 storm during a rolling "
        "shutdown. Jobs the worker still held are reclaimed through lease expiry."
    ),
)
async def deregister_worker(
    worker_id: UUID,
    use_case: DeregisterWorkerDep,
    graceful: Annotated[
        bool, Query(description="False when the worker is being killed rather than drained.")
    ] = True,
) -> Response:
    await use_case.execute(deregister_command(worker_id=worker_id, graceful=graceful))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{worker_id}/health",
    response_model=WorkerHealthResponse,
    summary="Health of one worker as the control plane sees it",
    description=(
        "The control plane's opinion, built from registration and heartbeats — "
        "not a proxy to the worker's own health endpoint. It answers 'would a "
        "job be sent there right now?', which is the only question scheduling asks."
    ),
    responses={404: {"model": ProblemDetails}},
)
async def worker_health(worker_id: UUID, use_case: ListWorkersDep) -> WorkerHealthResponse:
    # The frozen application layer exposes no "get one worker" use case, so the
    # fleet listing is filtered here. That is a lookup, not a decision — but it
    # is O(fleet) per call and a GetWorkerUseCase would be the right fix.
    identifier = WorkerId(worker_id)
    workers = await use_case.execute()
    matches = [view for view in workers if view.id == identifier]
    if not matches:
        raise EntityNotFoundError("Worker", identifier)
    return WorkerHealthResponse.of(matches[0])
