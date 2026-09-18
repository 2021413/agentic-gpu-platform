"""Shared response shapes.

These models exist so the OpenAPI document tells the truth about error and
health payloads; the runtime rendering of a problem document lives in
``interfaces.api.errors`` and is deliberately hand-built, because a failing
request must not depend on a model that could itself fail to serialize.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DependencyHealthResponse",
    "HealthResponse",
    "ProblemDetails",
    "ReadinessResponse",
]


class ProblemDetails(BaseModel):
    """RFC 9457 document, as returned by every error path of this API."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="URI reference identifying the problem type.")
    title: str = Field(description="Short, human-readable summary, stable per type.")
    status: int = Field(description="HTTP status code, repeated inside the body.")
    detail: str = Field(description="Human-readable explanation for this occurrence.")
    instance: str = Field(description="Path of the request that failed.")
    code: str = Field(description="Stable domain error code; branch on this, not on prose.")
    details: dict[str, Any] | None = Field(
        default=None, description="Structured context carried by the error."
    )
    request_id: str | None = Field(
        default=None, description="Correlation id, also returned in X-Request-ID."
    )


class HealthResponse(BaseModel):
    """Liveness. Answered from the process alone, never from a dependency."""

    status: Literal["alive"] = "alive"


class DependencyHealthResponse(BaseModel):
    name: str
    healthy: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness, with the per-dependency verdicts that justify it."""

    ready: bool
    dependencies: list[DependencyHealthResponse] = Field(default_factory=list)
