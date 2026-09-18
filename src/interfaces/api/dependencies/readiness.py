"""Readiness: can this process serve traffic right now?

Liveness and readiness answer different questions and must never share an
implementation:

* ``/health`` (liveness) asks "is this process running?". It must not touch
  PostgreSQL or Redis: a probe that fails when the database blinks gets the
  container *killed*, turning a recoverable dependency outage into a crash loop.
* ``/ready`` (readiness) asks "can this process serve a request end to end?". It
  must touch the dependencies, because that is the whole point — a pod that
  cannot reach Redis should be taken out of the load balancer, not restarted.

The probe itself is a port: this layer declares what it needs, ``bootstrap``
supplies an implementation that knows about the real adapters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["AlwaysReadyProbe", "DependencyHealth", "ReadinessProbe", "ReadinessReport"]


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    """Verdict for one dependency the API cannot serve traffic without."""

    name: str
    healthy: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    dependencies: tuple[DependencyHealth, ...] = field(default_factory=tuple)

    @property
    def ready(self) -> bool:
        """Ready only when every checked dependency answered.

        An empty report is ready: a deployment that declares no dependency is
        trivially able to serve, and failing closed there would block every
        single-process setup for no reason.
        """
        return all(dependency.healthy for dependency in self.dependencies)

    @classmethod
    def of(cls, dependencies: Sequence[DependencyHealth]) -> ReadinessReport:
        return cls(dependencies=tuple(dependencies))


@runtime_checkable
class ReadinessProbe(Protocol):
    """Checks every dependency the API needs to answer a request."""

    async def check(self) -> ReadinessReport:
        """Never raises: an unreachable dependency is a *result*, not an error."""
        ...


class AlwaysReadyProbe:
    """Reports readiness without checking anything.

    For tests and for a single-process development run with no external
    dependency. Wiring it in production would make ``/ready`` a lie.
    """

    async def check(self) -> ReadinessReport:
        return ReadinessReport()
