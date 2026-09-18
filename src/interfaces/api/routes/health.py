"""Liveness and readiness.

They are two different questions and are answered by two different code paths;
see ``interfaces.api.dependencies.readiness`` for why conflating them turns a
dependency outage into a crash loop.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from interfaces.api.dependencies.providers import ReadinessDep
from interfaces.api.schemas.common import (
    DependencyHealthResponse,
    HealthResponse,
    ReadinessResponse,
)

__all__ = ["router"]

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Answered from the process alone. It never touches PostgreSQL, Redis or "
        "a worker, so a dependency outage does not get this container killed."
    ),
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "At least one dependency is unreachable."}},
    description=(
        "Checks every dependency required to serve a request. Returns 503 when "
        "one is down so the instance leaves the load balancer without restarting."
    ),
)
async def ready(probe: ReadinessDep) -> JSONResponse:
    report = await probe.check()
    body = ReadinessResponse(
        ready=report.ready,
        dependencies=[
            DependencyHealthResponse(
                name=dependency.name, healthy=dependency.healthy, detail=dependency.detail
            )
            for dependency in report.dependencies
        ],
    )
    # The status code *is* the answer for an orchestrator, which reads it and
    # ignores the body; the body exists for the human debugging afterwards.
    return JSONResponse(
        status_code=200 if report.ready else 503,
        content=body.model_dump(mode="json"),
    )
