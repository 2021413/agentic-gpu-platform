"""Event fan-out (spec sections 15 and 21).

Two consumers matter: durable audit (PostgreSQL) and live clients (SSE). The
port covers publication and subscription; delivery guarantees belong to the
adapter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from domain.events.base import DomainEvent
from domain.value_objects.identifiers import RunId

__all__ = ["EventBus"]


@runtime_checkable
class EventBus(Protocol):
    """Publishes domain events and lets clients follow a run in real time."""

    async def publish(self, events: Sequence[DomainEvent]) -> None:
        """Publish a batch. Called after the transaction that produced them."""
        ...

    def subscribe(self, run_id: RunId) -> AsyncIterator[DomainEvent]:
        """Stream events for one run, starting from now.

        The iterator terminates when the run reaches a terminal state or the
        subscriber disconnects.
        """
        ...
