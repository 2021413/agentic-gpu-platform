"""Serialization of domain events to and from the ``run_events`` table.

``DomainEvent.payload()`` renders identifiers and enumerations with ``str``,
which is lossy on its own: ``"RUNNING"`` could be a ``JobStatus`` or just a
word. Reading an event back therefore needs the declared field types, which is
exactly what this module rebuilds — once per event class, from the dataclass
annotations.

The wire identifier is ``DomainEvent.name``: renaming one breaks every row
already stored, so the registry is keyed on it and an unknown name is a loud
failure rather than a silently dropped audit entry.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints
from uuid import UUID

from domain import events as domain_events
from domain.events.base import DomainEvent
from domain.value_objects.identifiers import EntityId, RunId

__all__ = [
    "EVENT_TYPES",
    "UnknownDomainEventError",
    "event_run_id",
    "load_event",
]

_ENVELOPE_FIELDS = ("occurred_at", "event_id")


class UnknownDomainEventError(LookupError):
    """A stored event carries a name no current class claims.

    Raised rather than skipped: the event log is an audit trail, and silently
    dropping a row would make ``list_by_run`` lie about the history of a run.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"no domain event class registered for {name!r}")
        self.name = name


def _registry() -> dict[str, type[DomainEvent]]:
    found: dict[str, type[DomainEvent]] = {}
    for attribute in vars(domain_events).values():
        if (
            isinstance(attribute, type)
            and issubclass(attribute, DomainEvent)
            and attribute is not DomainEvent
        ):
            found[attribute.name] = attribute
    return found


EVENT_TYPES: Mapping[str, type[DomainEvent]] = _registry()


def event_run_id(event: DomainEvent) -> RunId | None:
    """The run an event belongs to, when it belongs to one.

    Worker lifecycle events are global: they describe the fleet, not a run, and
    land in the log with a NULL ``run_id``.
    """
    run_id = getattr(event, "run_id", None)
    return run_id if isinstance(run_id, RunId) else None


def load_event(
    *, name: str, payload: Mapping[str, Any], occurred_at: datetime, event_id: UUID
) -> DomainEvent:
    """Rebuild a typed event from its stored row."""
    try:
        event_type = EVENT_TYPES[name]
    except KeyError as exc:
        raise UnknownDomainEventError(name) from exc

    converters = _converters(event_type)
    body = {key: converters.get(key, _identity)(value) for key, value in payload.items()}
    return event_type(occurred_at=occurred_at, event_id=event_id, **body)


def _identity(value: Any) -> Any:
    return value


# Resolving annotations is not cheap and the set of event classes is closed,
# so the readers are built once per class and kept.
_CONVERTER_CACHE: dict[type[DomainEvent], Mapping[str, Callable[[Any], Any]]] = {}


def _converters(event_type: type[DomainEvent]) -> Mapping[str, Callable[[Any], Any]]:
    """Per-field readers for one event class."""
    cached = _CONVERTER_CACHE.get(event_type)
    if cached is None:
        hints = get_type_hints(event_type)
        cached = {
            field: _converter_for(annotation)
            for field, annotation in hints.items()
            if field not in _ENVELOPE_FIELDS
        }
        _CONVERTER_CACHE[event_type] = cached
    return cached


def _converter_for(annotation: Any) -> Callable[[Any], Any]:
    target = _unwrap_optional(annotation)
    origin = get_origin(target)
    if origin in (tuple, frozenset, set):
        # JSON has no tuple. Restoring one as a list makes the rebuilt event
        # unequal to the event that was stored — which is silent, and which
        # breaks anything comparing an event to what it published.
        item = _converter_for(_element_type(target))
        builder = origin
        return _optional(lambda value: builder(item(v) for v in value))
    if not isinstance(target, type):
        return _identity
    if issubclass(target, EntityId):
        return _optional(lambda value: target(UUID(str(value))))
    if issubclass(target, Enum):
        return _optional(target)
    if issubclass(target, datetime):
        return _optional(datetime.fromisoformat)
    return _identity


def _optional(convert: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Keep ``None`` as ``None``: an absent identifier is not an empty one."""

    def read(value: Any) -> Any:
        return None if value is None else convert(value)

    return read


def _element_type(annotation: Any) -> Any:
    """The element type of ``tuple[X, ...]`` / ``set[X]``, or ``Any``."""
    args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
    return args[0] if args else Any


def _unwrap_optional(annotation: Any) -> Any:
    """``X | None`` -> ``X``; anything else is returned untouched."""
    if get_origin(annotation) in (Union, types.UnionType):
        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(candidates) == 1:
            return candidates[0]
    return annotation
