"""Bounded retry budgets (spec section 11).

Every loop in the platform is bounded. These values travel with the run so a
restart of the orchestrator cannot silently reset a budget and let a run spin
forever.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RunLimits"]


@dataclass(frozen=True, slots=True)
class RunLimits:
    """Per-run bounds on the agentic loops."""

    max_plan_revisions: int = 2
    max_coder_iterations: int = 6
    max_repair_iterations: int = 3
    max_worker_retries: int = 3
    max_parallel_candidates: int = 3

    def __post_init__(self) -> None:
        for name in (
            "max_plan_revisions",
            "max_coder_iterations",
            "max_repair_iterations",
            "max_worker_retries",
            "max_parallel_candidates",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.max_parallel_candidates < 1:
            raise ValueError("max_parallel_candidates must be at least 1")
        if self.max_worker_retries < 1:
            raise ValueError("max_worker_retries must be at least 1")
