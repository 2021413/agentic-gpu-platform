"""Run endpoints: creation, inspection, cancellation, candidates."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Header, Query, status

from domain.value_objects.identifiers import RunId
from interfaces.api.dependencies.providers import (
    CancelRunDep,
    CreateRunDep,
    GetRunDep,
    ListCandidatesDep,
)
from interfaces.api.schemas.common import ProblemDetails
from interfaces.api.schemas.runs import (
    CancelRunRequest,
    CandidateListResponse,
    CreateRunRequest,
    RunDetailResponse,
    RunResponse,
)

__all__ = ["projects_router", "router"]

router = APIRouter(prefix="/v1/runs", tags=["runs"])
projects_router = APIRouter(prefix="/v1/projects", tags=["runs"])

IdempotencyKeyHeader = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=255,
        description=(
            "Replaying the same key returns the run created the first time. "
            "Delivery is assumed to be at-least-once, so a client that times "
            "out must retry with the same key instead of creating a twin run."
        ),
    ),
]


@projects_router.post(
    "/{project_id}/runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an agentic run",
    description=(
        "Accepted, not completed: the run is made durable and scheduled, and "
        "the client follows it through GET /v1/runs/{run_id} or the event "
        "stream. Nothing here waits behind inference."
    ),
    responses={404: {"model": ProblemDetails}, 422: {"model": ProblemDetails}},
)
async def create_run(
    project_id: UUID,
    payload: CreateRunRequest,
    use_case: CreateRunDep,
    idempotency_key: IdempotencyKeyHeader = None,
) -> RunResponse:
    command = payload.to_command(project_id=project_id, idempotency_key=idempotency_key)
    return RunResponse.of(await use_case.execute(command))


@router.get(
    "/{run_id}",
    response_model=RunDetailResponse,
    summary="Fetch a run",
    responses={404: {"model": ProblemDetails}},
)
async def get_run(
    run_id: UUID,
    use_case: GetRunDep,
    detailed: Annotated[
        bool,
        Query(description="Also return the latest plan and every candidate."),
    ] = False,
) -> RunDetailResponse:
    return RunDetailResponse.of(await use_case.execute(RunId(run_id), detailed=detailed))


@router.post(
    "/{run_id}/cancel",
    response_model=RunResponse,
    summary="Cancel a run",
    description=(
        "Idempotent: cancelling an already cancelled or finished run returns "
        "its current state with the same 200, because the caller's intent — "
        "'this run must not continue' — is already satisfied."
    ),
    responses={404: {"model": ProblemDetails}},
)
async def cancel_run(
    run_id: UUID,
    use_case: CancelRunDep,
    payload: Annotated[CancelRunRequest, Body(default_factory=CancelRunRequest)],
) -> RunResponse:
    return RunResponse.of(await use_case.execute(payload.to_command(run_id=run_id)))


@router.get(
    "/{run_id}/candidates",
    response_model=CandidateListResponse,
    summary="List the candidates of a run",
    responses={404: {"model": ProblemDetails}},
)
async def list_candidates(
    run_id: UUID, get_run_use_case: GetRunDep, use_case: ListCandidatesDep
) -> CandidateListResponse:
    # The run is read first purely so an unknown run is a 404 instead of an
    # empty list. It costs one read and it lets the use case raise, which keeps
    # the branch out of this handler.
    identifier = RunId(run_id)
    await get_run_use_case.execute(identifier)
    return CandidateListResponse.of(await use_case.execute(identifier))
