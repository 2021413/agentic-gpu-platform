"""Serialisation round trips, with no Redis in sight.

These are the unit tests behind the integration ones: if a worker or a job does
not survive a round trip through the hash representation, every adapter test
above becomes a test of the wrong thing. The event codec gets a generated pass
over *every* event the domain exports, because the one that breaks will be the
one somebody adds next month.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from enum import StrEnum
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from uuid import UUID, uuid4

import pytest
from test_redis_support import T0, later, make_job, make_worker

from domain.enums import FailureKind, JobStatus, JobType, Priority, WorkerStatus
from domain.events.base import DomainEvent
from domain.value_objects.identifiers import EntityId, IdempotencyKey, WorkerId
from domain.value_objects.lease import LeaseToken
from domain.value_objects.worker import WorkerLoad
from infrastructure.redis.codecs import (
    decode_event,
    encode_event,
    event_registry,
    job_from_mapping,
    job_to_mapping,
    queue_score,
    worker_from_mapping,
    worker_to_mapping,
)


# -- workers ------------------------------------------------------------
def test_a_worker_survives_a_round_trip() -> None:
    worker = make_worker(concurrency=4)
    worker.heartbeat(now=later(30), load=WorkerLoad(active_jobs=2, queued_jobs=1))
    worker.mark_unhealthy(now=later(60), reason="probe failed")

    restored = worker_from_mapping(worker_to_mapping(worker))

    assert restored.id == worker.id
    assert restored.endpoint == worker.endpoint
    assert restored.capabilities == worker.capabilities
    assert restored.status is WorkerStatus.UNHEALTHY
    assert restored.load == WorkerLoad(active_jobs=2, queued_jobs=1)
    assert restored.registered_at == T0
    assert restored.last_heartbeat_at == later(30)
    assert restored.metadata == worker.metadata


def test_a_rehydrated_worker_has_nothing_to_announce() -> None:
    """Events belong to the transition that produced them, never to a read."""
    worker = make_worker()
    worker.mark_unavailable(now=later(1), reason="gone")

    restored = worker_from_mapping(worker_to_mapping(worker))

    assert restored.pending_events == ()


# -- jobs ---------------------------------------------------------------
def test_a_queued_job_survives_a_round_trip() -> None:
    job = make_job(job_type=JobType.TEST, priority=Priority.HIGH)

    restored = job_from_mapping(job_to_mapping(job))

    assert restored.id == job.id
    assert restored.run_id == job.run_id
    assert restored.project_id == job.project_id
    assert restored.type is JobType.TEST
    assert restored.role is None
    assert restored.priority is Priority.HIGH
    assert restored.status is JobStatus.QUEUED
    assert restored.payload == job.payload
    assert restored.requirements == job.requirements
    assert restored.lease is None
    assert restored.assigned_worker_id is None
    assert restored.started_at is None
    assert restored.result is None


def test_a_finished_job_survives_a_round_trip() -> None:
    """Every optional field populated at once: the empty-string convention has
    to be reversible for all of them, not just the ones a happy path uses."""
    job = make_job()
    worker_id = WorkerId.generate()
    lease = job.lease_to(worker_id=worker_id, now=later(1), duration=later(31) - later(1))
    job.complete(token=lease.token, now=later(20), result={"patch": "diff --git"})

    restored = job_from_mapping(job_to_mapping(job))

    assert restored.status is JobStatus.SUCCEEDED
    assert restored.attempt == 1
    assert restored.started_at == later(1)
    assert restored.completed_at == later(20)
    assert restored.result == {"patch": "diff --git"}
    assert restored.lease is None


def test_a_leased_job_keeps_its_lease() -> None:
    job = make_job()
    worker_id = WorkerId.generate()
    lease = job.lease_to(worker_id=worker_id, now=later(1), duration=later(31) - later(1))

    restored = job_from_mapping(job_to_mapping(job))

    assert restored.lease == lease
    assert restored.lease is not None
    assert restored.lease.token == lease.token
    assert restored.assigned_worker_id == worker_id


def test_a_failed_job_keeps_why() -> None:
    job = make_job()
    lease = job.lease_to(worker_id=WorkerId.generate(), now=later(1), duration=later(31) - later(1))
    job.fail(token=lease.token, now=later(5), kind=FailureKind.INFERENCE, reason="timeout")

    restored = job_from_mapping(job_to_mapping(job))

    assert restored.failure_kind is FailureKind.INFERENCE
    assert restored.failure_reason == "timeout"
    assert restored.status is JobStatus.FAILED


def test_an_empty_optional_field_comes_back_as_none() -> None:
    """Redis has no null; the empty string stands in for it and must reverse."""
    job = make_job(job_type=JobType.BUILD)
    record = job_to_mapping(job)

    assert record["role"] == ""
    assert record["lease_token"] == ""
    assert job_from_mapping(record).idempotency_key is None


# -- ordering -----------------------------------------------------------
def test_the_queue_score_ranks_priority_first_then_age() -> None:
    critical_late = queue_score(Priority.CRITICAL, later(100))
    high_early = queue_score(Priority.HIGH, later(0))
    normal_early = queue_score(Priority.NORMAL, later(0))
    normal_late = queue_score(Priority.NORMAL, later(1))

    assert critical_late < high_early < normal_early < normal_late


def test_the_queue_score_resolves_to_the_millisecond() -> None:
    """Sub-millisecond differences collapse, which is why the adapters break
    ties on the job id — otherwise they would disagree."""
    first = queue_score(Priority.NORMAL, datetime(2026, 1, 1, tzinfo=UTC))
    second = queue_score(Priority.NORMAL, datetime(2026, 1, 1, 0, 0, 0, 400, tzinfo=UTC))

    assert first == second


# -- events -------------------------------------------------------------
def test_the_registry_knows_every_exported_event() -> None:
    registry = event_registry()
    exported = {
        event_type
        for event_type in vars(__import__("domain.events", fromlist=["*"])).values()
        if isinstance(event_type, type) and issubclass(event_type, DomainEvent)
    }

    assert exported <= set(registry.values())
    assert len(registry) == len({event_type.name for event_type in exported})


def test_an_unknown_event_name_is_dropped_not_raised() -> None:
    """A reader that meets an event from a newer deployment keeps going."""
    raw = {
        "name": "run.teleported",
        "event_id": str(uuid4()),
        "occurred_at": T0.isoformat(),
        "payload": "{}",
    }

    assert decode_event(raw) is None


_SAMPLES: dict[type, Any] = {
    IdempotencyKey: IdempotencyKey("key"),
    LeaseToken: LeaseToken("token"),
    datetime: T0,
    bool: True,
    int: 7,
    float: 1.5,
    str: "sample",
}


def _sample(annotation: Any) -> Any:
    """A plausible value for a declared field type."""
    origin = get_origin(annotation)
    if origin is UnionType or origin is Union:
        return _sample(next(arg for arg in get_args(annotation) if arg is not NoneType))
    if origin in (list, set, frozenset, tuple):
        return origin()
    return _sample_class(annotation) if isinstance(annotation, type) else "sample"


def _sample_class(annotation: type) -> Any:
    if issubclass(annotation, EntityId):
        return annotation.generate()
    if issubclass(annotation, StrEnum):
        return next(iter(annotation))
    if issubclass(annotation, UUID):
        return uuid4()
    return _SAMPLES.get(annotation, "sample")


@pytest.mark.parametrize("event_type", sorted(event_registry().values(), key=lambda c: c.name))
def test_every_domain_event_survives_a_round_trip(event_type: type[DomainEvent]) -> None:
    hints = get_type_hints(event_type)
    body = {
        field.name: _sample(hints[field.name])
        for field in dataclass_fields(event_type)
        if field.name not in ("occurred_at", "event_id")
    }
    event = event_type(occurred_at=T0, **body)

    restored = decode_event(encode_event(event))

    assert restored == event
