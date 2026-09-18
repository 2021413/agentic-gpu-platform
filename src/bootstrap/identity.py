"""Concrete clock and identifier generator.

Trivial, and deliberately so: the point is that they are injected. A domain
that called ``datetime.now()`` would be untestable, and one that minted UUIDs
inline could not be made reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from domain.value_objects.identifiers import EntityId

__all__ = ["SystemClock", "UuidGenerator"]


class SystemClock:
    """Wall-clock time, always timezone-aware."""

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(UTC)


class UuidGenerator:
    """Random identifiers, typed per aggregate."""

    __slots__ = ()

    def next_id[T: EntityId](self, kind: type[T]) -> T:
        return kind(uuid4())
