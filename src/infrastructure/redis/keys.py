"""The Redis keyspace, in one place.

Every key the platform touches is minted here so the layout can be reviewed as
a whole, prefixed per environment (staging and production sharing one Redis is
a real deployment), and reasoned about from the Lua scripts.

Some scripts derive a key from a prefix instead of receiving it in ``KEYS``
(``claim`` cannot know which job it will pop before it pops it). That is only
safe while every key resolves to the same Redis Cluster slot, so a clustered
deployment must use a hash-tagged prefix such as ``"{agp}"`` — hence the
``prefix`` being configurable rather than hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.enums import JobType
from domain.value_objects.identifiers import JobId, RunId, WorkerId

__all__ = ["DEFAULT_PREFIX", "Keyspace"]

DEFAULT_PREFIX = "agp"


@dataclass(frozen=True, slots=True)
class Keyspace:
    """Names every Redis key and the prefixes the Lua scripts build keys from."""

    prefix: str = DEFAULT_PREFIX

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("redis key prefix must not be empty")

    # -- worker registry ------------------------------------------------
    @property
    def worker_index(self) -> str:
        """Set of registered worker ids; the only way to enumerate the pool."""
        return f"{self.prefix}:workers"

    def worker(self, worker_id: WorkerId | str) -> str:
        """Accepts a raw id too: enumerating the index yields strings, and
        rebuilding a typed id only to render it again would be ceremony."""
        return f"{self.prefix}:worker:{worker_id}"

    # -- job queue ------------------------------------------------------
    def job(self, job_id: JobId | str) -> str:
        return f"{self.prefix}:job:{job_id}"

    @property
    def job_prefix(self) -> str:
        return f"{self.prefix}:job:"

    def ready(self, job_type: JobType) -> str:
        """Sorted set of claimable jobs of one type, ordered by queue score."""
        return f"{self.prefix}:ready:{job_type.value}"

    @property
    def ready_prefix(self) -> str:
        return f"{self.prefix}:ready:"

    @property
    def leases(self) -> str:
        """Sorted set of in-flight jobs scored by lease expiry, so reclaiming is a range query."""
        return f"{self.prefix}:leases"

    def run_jobs(self, run_id: RunId) -> str:
        return f"{self.prefix}:run:{run_id}:jobs"

    @property
    def run_jobs_prefix(self) -> str:
        return f"{self.prefix}:run:"

    @property
    def run_jobs_suffix(self) -> str:
        return ":jobs"

    # -- events ---------------------------------------------------------
    @property
    def event_stream(self) -> str:
        """Every published event, for projections and audit tailing."""
        return f"{self.prefix}:events"

    def run_stream(self, run_id: RunId) -> str:
        """One stream per run: an SSE subscriber reads only what it needs."""
        return f"{self.prefix}:events:run:{run_id}"

    # -- locks ----------------------------------------------------------
    def lock(self, name: str) -> str:
        return f"{self.prefix}:lock:{name}"
