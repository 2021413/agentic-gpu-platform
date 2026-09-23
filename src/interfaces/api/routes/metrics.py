"""The Prometheus exposition.

Separate from `/health` and `/ready` because it answers a third question. Those
two say whether this process should be killed or sent traffic; this one says
what it has been doing, and a scraper reading it must never be able to affect
either answer.

Unauthenticated, like the probes: the endpoint exposes counts and durations,
never a payload, an identifier or a secret. Whoever can reach the port can
already reach `/v1/workers`.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from interfaces.api.dependencies.providers import ContainerDep

__all__ = ["router"]

router = APIRouter(tags=["health"])


@router.get(
    "/metrics",
    summary="Prometheus exposition",
    description=(
        "Empty with 404 when METRICS_ENABLED is false, rather than an empty "
        "body with 200: a scrape that succeeds and returns nothing is "
        "indistinguishable from a platform doing nothing."
    ),
    response_class=Response,
)
async def metrics(container: ContainerDep) -> Response:
    exposition = container.metrics
    if exposition is None:
        return Response(status_code=404)
    return Response(content=exposition.render(), media_type=exposition.content_type)
