"""Translating domain objects to and from Redis primitives.

Redis stores flat strings, the domain stores typed aggregates, and this module
is the only place allowed to know both. Two rules keep it honest:

* a hash field is always a string, and the empty string means ``None`` — Redis
  has no null, and a sentinel that is also a legal value would be worse;
* reconstruction goes through the aggregate's own constructor, so a decoded
  entity is indistinguishable from one the application built, minus its event
  buffer (a rehydrated aggregate has nothing to announce).

Hashes rather than one JSON blob: a heartbeat rewrites three fields and the
``claim`` script rewrites six, and doing that without read-modify-write is what
makes those operations atomic.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import fields as dataclass_fields
from datetime import datetime
from enum import StrEnum
from functools import cache
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from uuid import UUID

from domain import events as domain_events
from domain.entities.job import Job
from domain.entities.worker import Worker
from domain.enums import (
    AgentRole,
    FailureKind,
    JobStatus,
    JobType,
    Priority,
    WorkerStatus,
)
from domain.events import DomainEvent
from domain.events.run import RunCancelled, RunCompleted, RunFailed, RunStateChanged
from domain.value_objects.identifiers import (
    CandidateId,
    EntityId,
    IdempotencyKey,
    JobId,
    ProjectId,
    RunId,
    WorkerId,
)
from domain.value_objects.lease import Lease, LeaseToken
from domain.value_objects.worker import (
    GpuSpec,
    JobRequirements,
    WorkerCapabilities,
    WorkerEndpoint,
    WorkerLoad,
)

__all__ = [
    "PRIORITY_BAND",
    "as_text",
    "decode_event",
    "encode_event",
    "event_registry",
    "is_terminal_for_run",
    "job_from_mapping",
    "job_to_mapping",
    "lease_from_mapping",
    "queue_score",
    "run_id_of",
    "text_mapping",
    "worker_from_mapping",
    "worker_to_mapping",
]


# -- replies ------------------------------------------------------------
def as_text(value: object) -> str:
    """Narrow a Redis reply to text.

    The clients these adapters are given are built with
    ``decode_responses=True``, but the driver's own types cannot express that:
    a reply is declared ``bytes | str`` whatever the connection was told to do.
    Decoding explicitly at the boundary keeps the rest of the module honest
    about what it handles, where a blanket cast would simply hide a reply that
    really did come back undecoded.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8")
    return str(value)


def text_mapping(raw: Mapping[Any, Any]) -> dict[str, str]:
    """A hash reply as the codecs below expect it."""
    return {as_text(key): as_text(value) for key, value in raw.items()}


# -- scalars ------------------------------------------------------------
def _text(raw: Mapping[str, str], field: str) -> str | None:
    """A hash field, with the empty string normalised back to ``None``."""
    value = raw.get(field)
    return value if value else None


def _require(raw: Mapping[str, str], field: str) -> str:
    value = raw.get(field)
    if not value:
        raise ValueError(f"malformed redis record: missing field {field!r}")
    return value


def _moment(raw: Mapping[str, str], field: str) -> datetime | None:
    value = _text(raw, field)
    return datetime.fromisoformat(value) if value else None


def _int(raw: Mapping[str, str], field: str, default: int = 0) -> int:
    value = _text(raw, field)
    return int(value) if value else default


def _json_object(raw: Mapping[str, str], field: str) -> dict[str, Any] | None:
    value = _text(raw, field)
    if not value:
        return None
    decoded: Any = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError(f"field {field!r} should hold a JSON object")
    return {str(k): v for k, v in decoded.items()}


# -- worker -------------------------------------------------------------
def _capabilities_to_json(caps: WorkerCapabilities) -> str:
    return json.dumps(
        {
            "model_id": caps.model_id,
            "context_length": caps.context_length,
            "max_concurrency": caps.max_concurrency,
            # Sorted so two equal capability sets serialise identically, which
            # makes a stored record diffable between two heartbeats.
            "supported_roles": sorted(role.value for role in caps.supported_roles),
            "supports_tools": caps.supports_tools,
            "supports_json_schema": caps.supports_json_schema,
            "gpu": {
                "gpu_type": caps.gpu.gpu_type,
                "gpu_count": caps.gpu.gpu_count,
                "memory_gb": caps.gpu.memory_gb,
                "tensor_parallel_size": caps.gpu.tensor_parallel_size,
            },
            "metadata": dict(caps.metadata),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _capabilities_from_json(payload: Mapping[str, Any]) -> WorkerCapabilities:
    gpu = payload.get("gpu") or {}
    return WorkerCapabilities(
        model_id=str(payload["model_id"]),
        context_length=int(payload["context_length"]),
        max_concurrency=int(payload["max_concurrency"]),
        supported_roles=frozenset(AgentRole(role) for role in payload["supported_roles"]),
        gpu=GpuSpec(
            gpu_type=gpu.get("gpu_type"),
            gpu_count=int(gpu.get("gpu_count", 1)),
            memory_gb=gpu.get("memory_gb"),
            tensor_parallel_size=int(gpu.get("tensor_parallel_size", 1)),
        ),
        supports_tools=bool(payload.get("supports_tools", True)),
        supports_json_schema=bool(payload.get("supports_json_schema", True)),
        metadata=dict(payload.get("metadata") or {}),
    )


def worker_to_mapping(worker: Worker) -> dict[str, str]:
    """Full worker record. ``heartbeat`` rewrites only the mutable subset."""
    return {
        "id": str(worker.id),
        "endpoint": str(worker.endpoint),
        "status": worker.status.value,
        "registered_at": worker.registered_at.isoformat(),
        "last_heartbeat_at": worker.last_heartbeat_at.isoformat(),
        "active_jobs": str(worker.load.active_jobs),
        "queued_jobs": str(worker.load.queued_jobs),
        "capabilities": _capabilities_to_json(worker.capabilities),
        "metadata": json.dumps(dict(worker.metadata), separators=(",", ":"), sort_keys=True),
    }


def worker_from_mapping(raw: Mapping[str, str]) -> Worker:
    capabilities = _json_object(raw, "capabilities")
    if capabilities is None:
        raise ValueError("malformed redis record: missing worker capabilities")
    registered_at = _moment(raw, "registered_at")
    if registered_at is None:
        raise ValueError("malformed redis record: missing field 'registered_at'")
    return Worker(
        worker_id=WorkerId.parse(_require(raw, "id")),
        endpoint=WorkerEndpoint(_require(raw, "endpoint")),
        capabilities=_capabilities_from_json(capabilities),
        registered_at=registered_at,
        status=WorkerStatus(_require(raw, "status")),
        load=WorkerLoad(active_jobs=_int(raw, "active_jobs"), queued_jobs=_int(raw, "queued_jobs")),
        last_heartbeat_at=_moment(raw, "last_heartbeat_at") or registered_at,
        metadata=_json_object(raw, "metadata") or {},
    )


# -- job ----------------------------------------------------------------
PRIORITY_BAND = 10**13
"""Width of one priority band, in the same unit as the epoch milliseconds below.

A queue score is ``-rank * BAND + created_ms``: the band dominates, so a higher
priority always sorts first, and inside a band the oldest job wins. The band is
wide enough to cover any plausible timestamp (year 2286 is ~1e13 ms) and small
enough that the result stays an exactly representable double.
"""


def queue_score(priority: Priority, created_at: datetime) -> int:
    """Sort key of a claimable job: priority first, age as the tie-breaker."""
    return -priority.rank * PRIORITY_BAND + int(created_at.timestamp() * 1000)


def _requirements_to_json(requirements: JobRequirements) -> str:
    return json.dumps(
        {
            "role": requirements.role.value,
            "model_id": requirements.model_id,
            "estimated_prompt_tokens": requirements.estimated_prompt_tokens,
            "requires_tools": requirements.requires_tools,
            "requires_json_schema": requirements.requires_json_schema,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _requirements_from_json(payload: Mapping[str, Any]) -> JobRequirements:
    return JobRequirements(
        role=AgentRole(payload["role"]),
        model_id=payload.get("model_id"),
        estimated_prompt_tokens=int(payload.get("estimated_prompt_tokens", 0)),
        requires_tools=bool(payload.get("requires_tools", False)),
        requires_json_schema=bool(payload.get("requires_json_schema", True)),
    )


def job_to_mapping(job: Job) -> dict[str, str]:
    """Flat record of a job, including the ``score`` the ready set is ordered by.

    The score is stored rather than recomputed because the Lua scripts requeue
    jobs (release, reclaim) and must restore the original position without
    knowing how priorities are ranked. Keeping the formula in Python alone is
    what stops the two from drifting apart.
    """
    lease = job.lease
    return {
        "id": str(job.id),
        "run_id": str(job.run_id),
        "project_id": str(job.project_id),
        "type": job.type.value,
        "role": job.role.value if job.role is not None else "",
        "candidate_id": str(job.candidate_id) if job.candidate_id is not None else "",
        "priority": job.priority.value,
        "status": job.status.value,
        "attempt": str(job.attempt),
        "max_attempts": str(job.max_attempts),
        "payload": json.dumps(dict(job.payload), separators=(",", ":"), sort_keys=True),
        "requirements": (
            _requirements_to_json(job.requirements) if job.requirements is not None else ""
        ),
        "idempotency_key": str(job.idempotency_key) if job.idempotency_key is not None else "",
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at is not None else "",
        "completed_at": job.completed_at.isoformat() if job.completed_at is not None else "",
        "result": (
            json.dumps(dict(job.result), separators=(",", ":"), sort_keys=True)
            if job.result is not None
            else ""
        ),
        "failure_kind": job.failure_kind.value if job.failure_kind is not None else "",
        "failure_reason": job.failure_reason or "",
        "assigned_worker_id": (
            str(job.assigned_worker_id) if job.assigned_worker_id is not None else ""
        ),
        "lease_token": str(lease.token) if lease is not None else "",
        "lease_holder": str(lease.holder) if lease is not None else "",
        "lease_acquired_at": lease.acquired_at.isoformat() if lease is not None else "",
        "lease_expires_at": lease.expires_at.isoformat() if lease is not None else "",
        "score": str(queue_score(job.priority, job.created_at)),
        "acked": "",
    }


def lease_from_mapping(raw: Mapping[str, str]) -> Lease | None:
    """The lease half of a job record, or ``None`` when the job is not held."""
    token = _text(raw, "lease_token")
    holder = _text(raw, "lease_holder")
    acquired_at = _moment(raw, "lease_acquired_at")
    expires_at = _moment(raw, "lease_expires_at")
    if token is None or holder is None or acquired_at is None or expires_at is None:
        return None
    return Lease(
        job_id=JobId.parse(_require(raw, "id")),
        holder=WorkerId.parse(holder),
        token=LeaseToken(token),
        acquired_at=acquired_at,
        expires_at=expires_at,
    )


def job_from_mapping(raw: Mapping[str, str]) -> Job:
    created_at = _moment(raw, "created_at")
    if created_at is None:
        raise ValueError("malformed redis record: missing field 'created_at'")
    role = _text(raw, "role")
    candidate_id = _text(raw, "candidate_id")
    idempotency_key = _text(raw, "idempotency_key")
    failure_kind = _text(raw, "failure_kind")
    assigned = _text(raw, "assigned_worker_id")
    requirements = _json_object(raw, "requirements")
    return Job(
        job_id=JobId.parse(_require(raw, "id")),
        run_id=RunId.parse(_require(raw, "run_id")),
        project_id=ProjectId.parse(_require(raw, "project_id")),
        job_type=JobType(_require(raw, "type")),
        created_at=created_at,
        role=AgentRole(role) if role else None,
        candidate_id=CandidateId.parse(candidate_id) if candidate_id else None,
        priority=Priority(_require(raw, "priority")),
        status=JobStatus(_require(raw, "status")),
        attempt=_int(raw, "attempt"),
        max_attempts=_int(raw, "max_attempts", default=1),
        payload=_json_object(raw, "payload") or {},
        requirements=_requirements_from_json(requirements) if requirements else None,
        idempotency_key=IdempotencyKey(idempotency_key) if idempotency_key else None,
        lease=lease_from_mapping(raw),
        assigned_worker_id=WorkerId.parse(assigned) if assigned else None,
        started_at=_moment(raw, "started_at"),
        completed_at=_moment(raw, "completed_at"),
        result=_json_object(raw, "result"),
        failure_kind=FailureKind(failure_kind) if failure_kind else None,
        failure_reason=_text(raw, "failure_reason"),
    )


# -- events -------------------------------------------------------------
_ENVELOPE = ("occurred_at", "event_id")

_RUN_TERMINAL_NAMES = frozenset(
    {RunCancelled.name, RunCompleted.name, RunFailed.name},
)


@cache
def event_registry() -> Mapping[str, type[DomainEvent]]:
    """Wire name to event class, derived from the domain's own exports.

    Built by reflection rather than hand-maintained: a new event must be
    publishable the moment it is added to ``domain.events``, and a registry that
    has to be updated in two places is a registry that will be wrong.
    """
    registry: dict[str, type[DomainEvent]] = {}
    for exported in domain_events.__all__:
        candidate = getattr(domain_events, exported, None)
        if isinstance(candidate, type) and issubclass(candidate, DomainEvent):
            registry[candidate.name] = candidate
    return registry


@cache
def _body_types() -> Mapping[str, Mapping[str, Any]]:
    """Declared field types of every known event, resolved once per process.

    ``get_type_hints`` is not cheap and the annotations are strings here
    (``from __future__ import annotations`` everywhere), so this is resolved on
    first use and never again.
    """
    resolved: dict[str, Mapping[str, Any]] = {}
    for name, event_type in event_registry().items():
        hints = get_type_hints(event_type)
        resolved[name] = {
            field.name: hints[field.name]
            for field in dataclass_fields(event_type)
            if field.name not in _ENVELOPE
        }
    return resolved


_SCALARS: Mapping[type, Callable[[Any], Any]] = {bool: bool, int: int, float: float, str: str}
"""Dispatch table rather than a chain of ``if``s: JSON already decoded these to
roughly the right shape, this only pins the exact type the event declares."""


def _coerce(annotation: Any, value: Any) -> Any:
    """Rebuild a typed field from the string ``DomainEvent.payload`` produced.

    The event's own dataclass annotations drive this, so identifiers and enums
    survive a round trip without any per-event mapping code.
    """
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is UnionType or origin is Union:
        # ``JobId | None`` and friends: one concrete arm is the whole point of
        # an optional field, anything richer is left as decoded JSON.
        concrete = [arg for arg in get_args(annotation) if arg is not NoneType]
        return _coerce(concrete[0], value) if len(concrete) == 1 else value
    if origin in (list, set, frozenset, tuple) and isinstance(value, list):
        args = get_args(annotation)
        item_type = args[0] if args else Any
        return origin(_coerce(item_type, item) for item in value)
    return _coerce_class(annotation, value) if isinstance(annotation, type) else value


def _coerce_class(annotation: type, value: Any) -> Any:
    if issubclass(annotation, EntityId):
        return annotation.parse(value)
    if issubclass(annotation, StrEnum):
        return annotation(value)
    if issubclass(annotation, IdempotencyKey):
        return IdempotencyKey(str(value))
    if issubclass(annotation, datetime):
        return datetime.fromisoformat(str(value))
    if issubclass(annotation, UUID):
        return UUID(str(value))
    converter = _SCALARS.get(annotation)
    return converter(value) if converter is not None else value


def encode_event(event: DomainEvent) -> dict[str, str]:
    """Envelope plus JSON body — the exact shape an SSE client is handed."""
    return {
        "name": type(event).name,
        "event_id": str(event.event_id),
        "occurred_at": event.occurred_at.isoformat(),
        "payload": json.dumps(dict(event.payload()), separators=(",", ":"), sort_keys=True),
    }


def decode_event(raw: Mapping[str, str]) -> DomainEvent | None:
    """Rebuild an event, or ``None`` when its name is unknown here.

    Unknown names are expected rather than exceptional: during a rolling deploy
    an older reader tails a stream written by a newer publisher, and dropping
    what it cannot understand beats crashing the subscription.
    """
    name = raw.get("name", "")
    event_type = event_registry().get(name)
    if event_type is None:
        return None
    payload = _json_object(raw, "payload") or {}
    body = {
        field: _coerce(annotation, payload[field])
        for field, annotation in _body_types()[name].items()
        if field in payload
    }
    return event_type(
        occurred_at=datetime.fromisoformat(_require(raw, "occurred_at")),
        event_id=UUID(_require(raw, "event_id")),
        **body,
    )


def is_terminal_for_run(event: DomainEvent) -> bool:
    """Whether this event ends a run, and with it any subscription following it.

    Recognised by wire name rather than by class, so the rule matches what a
    subscriber actually decoded off the wire and stays true for an event that
    crossed a process boundary.
    """
    name = type(event).name
    if name in _RUN_TERMINAL_NAMES:
        return True
    return isinstance(event, RunStateChanged) and event.current.is_terminal


def run_id_of(event: DomainEvent) -> RunId | None:
    """The run an event belongs to, if any; worker lifecycle events have none."""
    run_id = getattr(event, "run_id", None)
    return run_id if isinstance(run_id, RunId) else None
