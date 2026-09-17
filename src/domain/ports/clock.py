"""Time and identity as injected dependencies.

Nothing in the domain calls ``datetime.now()`` or ``uuid4()`` directly: a
deterministic test must be able to control both, and a distributed system must
be able to swap the source.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from domain.value_objects.identifiers import EntityId

__all__ = ["Clock", "IdGenerator"]


@runtime_checkable
class Clock(Protocol):
    """Source of timezone-aware wall-clock time."""

    def now(self) -> datetime:
        """Current instant, always timezone-aware (UTC)."""
        ...


@runtime_checkable
class IdGenerator(Protocol):
    """Source of new aggregate identifiers."""

    def next_id[T: EntityId](self, kind: type[T]) -> T:
        """Produce a fresh identifier of the requested type."""
        ...
