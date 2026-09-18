"""Public, read-only view of the GPU fleet (spec section 20).

Mutating a worker is never public: registration, heartbeat, drain and
deregistration live behind service authentication in ``interfaces.worker_api``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from interfaces.api.dependencies.providers import ListWorkersDep
from interfaces.api.schemas.workers import WorkerListResponse

__all__ = ["router"]

router = APIRouter(prefix="/v1/workers", tags=["workers"])


@router.get(
    "",
    response_model=WorkerListResponse,
    summary="Inspect the worker pool",
    description=(
        "The pool is discovered, never configured: this is whatever has "
        "registered and is still heartbeating at the time of the call."
    ),
)
async def list_workers(
    use_case: ListWorkersDep,
    only_available: Annotated[
        bool,
        Query(description="Restrict to workers that may receive new jobs right now."),
    ] = False,
) -> WorkerListResponse:
    return WorkerListResponse.of(await use_case.execute(only_available=only_available))
