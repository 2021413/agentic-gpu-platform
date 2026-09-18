"""Readiness of the control plane.

Liveness and readiness are different questions and must not be conflated: the
process being alive says nothing about whether PostgreSQL will answer. A
readiness probe that lies gets traffic routed to an instance that cannot serve
it.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from interfaces.api.dependencies.readiness import DependencyHealth, ReadinessReport

__all__ = ["PlatformReadinessProbe"]

_log = logging.getLogger(__name__)


class PlatformReadinessProbe:
    """Checks every dependency the API cannot serve a request without."""

    __slots__ = ("_engine", "_redis", "_timeout")

    def __init__(
        self,
        *,
        engine: AsyncEngine | None = None,
        redis: object | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._engine = engine
        self._redis = redis
        self._timeout = timeout_seconds

    async def check(self) -> ReadinessReport:
        """Never raises: an unreachable dependency is a result, not an error."""
        checks = []
        if self._engine is not None:
            checks.append(self._check("database", self._ping_database()))
        if self._redis is not None:
            checks.append(self._check("redis", self._ping_redis()))
        if not checks:
            return ReadinessReport()
        return ReadinessReport.of(await asyncio.gather(*checks))

    async def _check(self, name: str, probe: object) -> DependencyHealth:
        try:
            await asyncio.wait_for(probe, timeout=self._timeout)  # type: ignore[arg-type]
        except TimeoutError:
            return DependencyHealth(name, False, f"did not answer within {self._timeout:g}s")
        except Exception as exc:
            return DependencyHealth(name, False, f"{type(exc).__name__}: {exc}")
        return DependencyHealth(name, True)

    async def _ping_database(self) -> None:
        assert self._engine is not None
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def _ping_redis(self) -> None:
        ping = getattr(self._redis, "ping", None)
        if ping is None:  # pragma: no cover - defensive
            raise RuntimeError("the redis client exposes no ping")
        await ping()
