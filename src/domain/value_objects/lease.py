"""Job leases (spec section 36).

A worker holds a job for a bounded time and renews while it works. If it
vanishes, the lease expires and the job becomes retryable — that is the whole
mechanism preventing permanently stuck jobs, so it lives in the domain and is
unit-testable against a fake clock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from uuid import uuid4

from domain.value_objects.identifiers import JobId, WorkerId

__all__ = ["Lease", "LeaseToken"]


@dataclass(frozen=True, slots=True)
class LeaseToken:
    """Proves that the caller reporting on a job still owns the lease it was granted.

    A worker that lost and re-acquired a job gets a different token, so a late
    report from the previous holder is rejected instead of corrupting state.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("lease token must not be empty")

    @classmethod
    def generate(cls) -> LeaseToken:
        return cls(uuid4().hex)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Lease:
    """A time-bounded claim on a job held by one consumer."""

    job_id: JobId
    holder: WorkerId
    token: LeaseToken
    acquired_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if self.expires_at <= self.acquired_at:
            raise ValueError("lease must expire strictly after it was acquired")
        if self.acquired_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("lease timestamps must be timezone-aware")

    @classmethod
    def granted(
        cls,
        *,
        job_id: JobId,
        holder: WorkerId,
        now: datetime,
        duration: timedelta,
        token: LeaseToken | None = None,
    ) -> Lease:
        return cls(
            job_id=job_id,
            holder=holder,
            token=token or LeaseToken.generate(),
            acquired_at=now,
            expires_at=now + duration,
        )

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def remaining(self, now: datetime) -> timedelta:
        return max(self.expires_at - now, timedelta(0))

    def renewed(self, *, now: datetime, duration: timedelta) -> Lease:
        """Extend the lease. Renewing an expired lease is a programming error."""
        if self.is_expired(now):
            raise ValueError("an expired lease cannot be renewed; the job was reassigned")
        return replace(self, expires_at=now + duration)

    def held_by(self, worker_id: WorkerId, token: LeaseToken) -> bool:
        return self.holder == worker_id and self.token == token
