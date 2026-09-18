"""Pluggable scheduling policy (spec section 5).

Policy is kept strictly apart from the registry: the registry knows *who
exists*, the scheduler decides *who gets the job*. New strategies (latency
aware, cost aware, affinity) are added by implementing this port alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.entities.worker import Worker
from domain.value_objects.worker import JobRequirements

__all__ = ["WorkerScheduler"]


@runtime_checkable
class WorkerScheduler(Protocol):
    """Chooses a worker for a job among the currently available ones."""

    @property
    def name(self) -> str:
        """Stable policy name, recorded on the job for observability."""
        ...

    def select(
        self, *, candidates: Sequence[Worker], requirements: JobRequirements
    ) -> Worker | None:
        """Pick a worker, or ``None`` when none is compatible.

        Returning ``None`` rather than raising is deliberate: "no worker right
        now" is an ordinary, transient condition in an elastic pool.
        """
        ...

    def rank(
        self, *, candidates: Sequence[Worker], requirements: JobRequirements
    ) -> Sequence[Worker]:
        """Eligible workers, best first — used for fan-out across workers."""
        ...
