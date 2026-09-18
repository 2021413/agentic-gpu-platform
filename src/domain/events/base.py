"""Domain event base (spec section 15).

Events are immutable facts about something that already happened. They are
emitted by aggregates, drained by the application layer, and only then handed to
an ``EventBus``: the domain never performs I/O to publish them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

__all__ = ["DomainEvent", "EventName"]

EventName = str


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainEvent:
    """Base class for every domain event.

    ``name`` is the wire identifier used by the event bus and by SSE clients, so
    it is a compatibility contract and must not be renamed casually.
    """

    name: ClassVar[EventName] = "domain.event"

    occurred_at: datetime
    event_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")

    def payload(self) -> Mapping[str, Any]:
        """Serializable body, excluding envelope fields.

        Values are rendered with ``str`` for identifiers and enums, which keeps
        the domain free of any serialization framework.
        """
        body: dict[str, Any] = {}
        for f in fields(self):
            if f.name in ("occurred_at", "event_id"):
                continue
            body[f.name] = _render(getattr(self, f.name))
        return body


def _render(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _render(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_render(v) for v in value]
    return str(value)
