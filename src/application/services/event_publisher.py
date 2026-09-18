"""Committing and publishing, in that order.

A subscriber must never learn about something that was rolled back. So the unit
of work drains the aggregates' events into the durable store inside the
transaction, and only once the commit succeeds are those same events pushed to
the live bus. Publication failures are logged, never fatal: the durable trail
already holds the truth, and a late subscriber backfills from it.
"""

from __future__ import annotations

import logging

from domain.ports.event_bus import EventBus
from domain.ports.repositories import UnitOfWork

__all__ = ["commit_and_publish"]

_log = logging.getLogger(__name__)


async def commit_and_publish(uow: UnitOfWork, bus: EventBus) -> None:
    """Commit the unit of work, then broadcast the events it drained."""
    await uow.commit()
    events = tuple(uow.collected_events)
    if not events:
        return
    try:
        await bus.publish(events)
    except Exception:  # live delivery is best effort, by design
        _log.warning(
            "failed to publish %d event(s); the durable event store remains authoritative",
            len(events),
            exc_info=True,
        )
