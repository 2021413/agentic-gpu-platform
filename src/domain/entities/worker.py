"""GPU worker registration aggregate (spec sections 3.3 and 25).

A worker is replaceable infrastructure, but its *lifecycle* is business logic:
whether a worker may receive a job, when it stops being trustworthy, and what
happens to its work when it vanishes. That logic lives here and is testable
without a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from domain.entities.base import Entity
from domain.enums import WorkerStatus
from domain.events.worker import (
    WorkerDeregistered,
    WorkerDraining,
    WorkerHeartbeatReceived,
    WorkerRegistered,
    WorkerStatusChanged,
    WorkerUnavailable,
)
from domain.exceptions import InvalidStateTransitionError, WorkerUnavailableError
from domain.value_objects.identifiers import WorkerId
from domain.value_objects.worker import (
    JobRequirements,
    WorkerCapabilities,
    WorkerEndpoint,
    WorkerLoad,
)

__all__ = ["Worker"]

_UNAVAILABLE = (WorkerStatus.OFFLINE, WorkerStatus.UNHEALTHY)


class Worker(Entity):
    """One registered inference worker."""

    __slots__ = (
        "_capabilities",
        "_endpoint",
        "_id",
        "_last_heartbeat_at",
        "_load",
        "_metadata",
        "_registered_at",
        "_status",
    )

    def __init__(
        self,
        *,
        worker_id: WorkerId,
        endpoint: WorkerEndpoint,
        capabilities: WorkerCapabilities,
        registered_at: datetime,
        status: WorkerStatus = WorkerStatus.READY,
        load: WorkerLoad | None = None,
        last_heartbeat_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._id = worker_id
        self._endpoint = endpoint
        self._capabilities = capabilities
        self._registered_at = registered_at
        self._status = status
        self._load = load or WorkerLoad()
        self._last_heartbeat_at = last_heartbeat_at or registered_at
        self._metadata: dict[str, Any] = dict(metadata or {})

    # -- construction ---------------------------------------------------
    @classmethod
    def register(
        cls,
        *,
        worker_id: WorkerId,
        endpoint: WorkerEndpoint,
        capabilities: WorkerCapabilities,
        now: datetime,
        metadata: Mapping[str, Any] | None = None,
    ) -> Worker:
        worker = cls(
            worker_id=worker_id,
            endpoint=endpoint,
            capabilities=capabilities,
            registered_at=now,
            status=WorkerStatus.READY,
            last_heartbeat_at=now,
            metadata=metadata,
        )
        worker.record(
            WorkerRegistered(
                occurred_at=now,
                worker_id=worker_id,
                model_id=capabilities.model_id,
                endpoint=str(endpoint),
                max_concurrency=capabilities.max_concurrency,
            )
        )
        return worker

    # -- identity and state ---------------------------------------------
    @property
    def identity(self) -> WorkerId:
        return self._id

    @property
    def id(self) -> WorkerId:
        return self._id

    @property
    def endpoint(self) -> WorkerEndpoint:
        return self._endpoint

    @property
    def capabilities(self) -> WorkerCapabilities:
        return self._capabilities

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def load(self) -> WorkerLoad:
        return self._load

    @property
    def registered_at(self) -> datetime:
        return self._registered_at

    @property
    def last_heartbeat_at(self) -> datetime:
        return self._last_heartbeat_at

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    @property
    def active_jobs(self) -> int:
        return self._load.active_jobs

    @property
    def available_slots(self) -> int:
        return max(self._capabilities.max_concurrency - self._load.active_jobs, 0)

    @property
    def utilization(self) -> float:
        """Occupancy in [0, 1]; the least-loaded scheduler orders on this."""
        return self._load.active_jobs / self._capabilities.max_concurrency

    # -- eligibility ----------------------------------------------------
    def is_stale(self, now: datetime, timeout: timedelta) -> bool:
        """Heartbeat silence beyond the configured timeout."""
        return now - self._last_heartbeat_at > timeout

    def can_accept(self, requirements: JobRequirements) -> bool:  # noqa: PLR0911
        """Whether this worker may take that job right now.

        Guard clauses rather than a single boolean: each rejection has its own
        reason, mirrored by ``rejection_reason`` for diagnosable scheduling.

        Draining workers are excluded here rather than in the scheduler, so no
        scheduling policy can accidentally bypass the rule.
        """
        if not self._status.accepts_new_jobs:
            return False
        if self.available_slots <= 0:
            return False
        caps = self._capabilities
        if not caps.supports_role(requirements.role):
            return False
        if requirements.model_id is not None and requirements.model_id != caps.model_id:
            return False
        if requirements.requires_tools and not caps.supports_tools:
            return False
        if requirements.requires_json_schema and not caps.supports_json_schema:
            return False
        return caps.fits(requirements.estimated_prompt_tokens)

    def rejection_reason(self, requirements: JobRequirements) -> str | None:  # noqa: PLR0911
        """Why ``can_accept`` said no — used for diagnosable scheduling errors."""
        if not self._status.accepts_new_jobs:
            return f"status={self._status}"
        if self.available_slots <= 0:
            return "at capacity"
        caps = self._capabilities
        if not caps.supports_role(requirements.role):
            return f"role {requirements.role} unsupported"
        if requirements.model_id is not None and requirements.model_id != caps.model_id:
            return f"model {caps.model_id} != {requirements.model_id}"
        if requirements.requires_tools and not caps.supports_tools:
            return "tool calling unsupported"
        if requirements.requires_json_schema and not caps.supports_json_schema:
            return "structured output unsupported"
        if not caps.fits(requirements.estimated_prompt_tokens):
            return f"prompt exceeds context length {caps.context_length}"
        return None

    # -- lifecycle ------------------------------------------------------
    def heartbeat(
        self,
        *,
        now: datetime,
        status: WorkerStatus | None = None,
        load: WorkerLoad | None = None,
        capabilities: WorkerCapabilities | None = None,
    ) -> None:
        """Record a heartbeat, optionally refreshing self-reported state.

        A worker that heartbeats after being declared unavailable is readmitted:
        a network partition must not permanently remove a healthy GPU.
        """
        self._last_heartbeat_at = now
        if capabilities is not None:
            self._capabilities = capabilities
        if load is not None:
            self._load = load
        previous = self._status
        if status is not None and status is not previous:
            self._transition(status, now)
        elif previous in _UNAVAILABLE:
            self._transition(WorkerStatus.READY, now)
        # The worker's own load report can make it full or free it again.
        # ``_sync_busy`` only touches READY/BUSY, so DRAINING survives a heartbeat.
        self._sync_busy()
        self.record(
            WorkerHeartbeatReceived(
                occurred_at=now,
                worker_id=self._id,
                status=self._status,
                active_jobs=self._load.active_jobs,
            )
        )

    def start_draining(self, now: datetime) -> None:
        """Stop accepting new jobs; already-accepted work may finish."""
        if self._status is WorkerStatus.DRAINING:
            return
        if self._status in _UNAVAILABLE:
            raise InvalidStateTransitionError("Worker", self._status, WorkerStatus.DRAINING)
        self._transition(WorkerStatus.DRAINING, now)
        self.record(
            WorkerDraining(occurred_at=now, worker_id=self._id, active_jobs=self.active_jobs)
        )

    def mark_unavailable(self, *, now: datetime, reason: str) -> None:
        """Heartbeat timeout or failed health probe: its jobs become retryable."""
        if self._status is WorkerStatus.OFFLINE:
            return
        self._transition(WorkerStatus.OFFLINE, now)
        self.record(WorkerUnavailable(occurred_at=now, worker_id=self._id, reason=reason))

    def mark_unhealthy(self, *, now: datetime, reason: str) -> None:
        if self._status is WorkerStatus.UNHEALTHY:
            return
        self._transition(WorkerStatus.UNHEALTHY, now)
        self.record(WorkerUnavailable(occurred_at=now, worker_id=self._id, reason=reason))

    def deregister(self, *, now: datetime, graceful: bool = True) -> None:
        if self._status is not WorkerStatus.OFFLINE:
            self._transition(WorkerStatus.OFFLINE, now)
        self.record(WorkerDeregistered(occurred_at=now, worker_id=self._id, graceful=graceful))

    @property
    def is_drained(self) -> bool:
        """A draining worker that finished its work can be removed safely."""
        return self._status is WorkerStatus.DRAINING and self._load.active_jobs == 0

    # -- occupancy ------------------------------------------------------
    def reserve_slot(self, requirements: JobRequirements) -> None:
        """Account for a job assigned to this worker.

        Kept on the entity so the in-memory view of load stays consistent
        between heartbeats, which are far too coarse for scheduling decisions.
        """
        if not self.can_accept(requirements):
            raise WorkerUnavailableError(self._id, self._status)
        self._load = WorkerLoad(
            active_jobs=self._load.active_jobs + 1,
            queued_jobs=self._load.queued_jobs,
        )
        self._sync_busy()

    def release_slot(self) -> None:
        self._load = WorkerLoad(
            active_jobs=max(self._load.active_jobs - 1, 0),
            queued_jobs=self._load.queued_jobs,
        )
        self._sync_busy()

    def _sync_busy(self) -> None:
        """Keep READY/BUSY consistent with capacity; never override DRAINING."""
        if self._status not in (WorkerStatus.READY, WorkerStatus.BUSY):
            return
        self._status = WorkerStatus.BUSY if self.available_slots == 0 else WorkerStatus.READY

    def _transition(self, target: WorkerStatus, now: datetime) -> None:
        previous = self._status
        if previous is target:
            return
        self._status = target
        self.record(
            WorkerStatusChanged(
                occurred_at=now, worker_id=self._id, previous=previous, current=target
            )
        )

    def __repr__(self) -> str:
        return (
            f"Worker(id={self._id}, model={self._capabilities.model_id}, "
            f"status={self._status}, active={self.active_jobs}/"
            f"{self._capabilities.max_concurrency})"
        )
