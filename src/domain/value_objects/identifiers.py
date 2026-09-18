"""Typed identifiers.

Every aggregate gets its own identifier type so that a ``RunId`` can never be
passed where a ``JobId`` is expected. Dataclass equality compares the exact
class, therefore ``RunId(u) != JobId(u)`` even for the same UUID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self
from uuid import UUID, uuid4

__all__ = [
    "CandidateId",
    "EntityId",
    "IdempotencyKey",
    "JobId",
    "PlanId",
    "ProjectId",
    "ReviewId",
    "RunId",
    "TaskId",
    "WorkerId",
    "WorkspaceId",
]


@dataclass(frozen=True, slots=True, order=True)
class EntityId:
    """Base class for UUID-backed identifiers."""

    value: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.value, UUID):  # pragma: no cover - defensive
            raise TypeError(
                f"{type(self).__name__} expects a UUID, got {type(self.value).__name__}"
            )

    @classmethod
    def generate(cls) -> Self:
        return cls(uuid4())

    @classmethod
    def parse(cls, raw: str | UUID | Self) -> Self:
        """Build an identifier from a string, a UUID or an identifier of the same type."""
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, UUID):
            return cls(raw)
        if isinstance(raw, str):
            try:
                return cls(UUID(raw))
            except ValueError as exc:
                raise ValueError(f"invalid {cls.__name__}: {raw!r}") from exc
        raise TypeError(f"cannot build {cls.__name__} from {type(raw).__name__}")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True, order=True)
class ProjectId(EntityId):
    """Identifies a project (a source repository the platform works on)."""


@dataclass(frozen=True, slots=True, order=True)
class RunId(EntityId):
    """Identifies one agentic run against a project."""


@dataclass(frozen=True, slots=True, order=True)
class JobId(EntityId):
    """Identifies a unit of schedulable work (inference or deterministic)."""


@dataclass(frozen=True, slots=True, order=True)
class WorkerId(EntityId):
    """Identifies a GPU worker registration."""


@dataclass(frozen=True, slots=True, order=True)
class CandidateId(EntityId):
    """Identifies one competing implementation attempt inside a run."""


@dataclass(frozen=True, slots=True, order=True)
class PlanId(EntityId):
    """Identifies a plan revision produced by the planner."""


@dataclass(frozen=True, slots=True, order=True)
class TaskId(EntityId):
    """Identifies a task inside a plan."""


@dataclass(frozen=True, slots=True, order=True)
class ReviewId(EntityId):
    """Identifies a review verdict."""


@dataclass(frozen=True, slots=True, order=True)
class WorkspaceId(EntityId):
    """Identifies an isolated filesystem workspace."""


@dataclass(frozen=True, slots=True, order=True)
class IdempotencyKey:
    """Caller-supplied key making an externally triggered mutation replay-safe."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("idempotency key must not be blank")
        if len(self.value) > 255:
            raise ValueError("idempotency key must be at most 255 characters")

    def __str__(self) -> str:
        return self.value
