"""Aggregate foundations.

Aggregates buffer the events they produce instead of publishing them: the
domain performs no I/O. The application layer drains the buffer inside the same
transaction that persists the change, which is what makes "persisted state and
emitted events agree" a property rather than a hope.
"""

from __future__ import annotations

from typing import Any

from domain.events.base import DomainEvent

__all__ = ["Entity"]


class Entity:
    """Identity-based entity with an outbound event buffer."""

    __slots__ = ("_events",)

    def __init__(self) -> None:
        self._events: list[DomainEvent] = []

    # -- events ---------------------------------------------------------
    def record(self, event: DomainEvent) -> None:
        self._events.append(event)

    def pull_events(self) -> tuple[DomainEvent, ...]:
        """Drain the buffer. Calling twice yields the events only once."""
        drained = tuple(self._events)
        self._events.clear()
        return drained

    @property
    def pending_events(self) -> tuple[DomainEvent, ...]:
        """Peek without draining (tests and assertions)."""
        return tuple(self._events)

    # -- identity -------------------------------------------------------
    @property
    def identity(self) -> Any:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return bool(self.identity == other.identity)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.identity))
