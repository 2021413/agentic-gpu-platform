"""Structured plans produced by the Planner role.

The planner decides what work exists and which parts of it may run
concurrently. That dependency graph is validated here — a plan with a cycle or
a dangling dependency is rejected before any coder is scheduled.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from domain.exceptions import PlanValidationError
from domain.value_objects.identifiers import PlanId, RunId, TaskId

__all__ = ["Plan", "PlanTask"]


@dataclass(frozen=True, slots=True)
class PlanTask:
    """One unit of the plan. ``key`` is the planner-chosen stable name."""

    id: TaskId
    key: str
    title: str
    description: str = ""
    target_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("task key must not be blank")
        if not self.title.strip():
            raise ValueError("task title must not be blank")
        if self.key in self.depends_on:
            raise PlanValidationError("a task cannot depend on itself", task=self.key)


@dataclass(frozen=True, slots=True)
class Plan:
    """One plan revision for a run."""

    id: PlanId
    run_id: RunId
    revision: int
    objective: str
    created_at: datetime
    tasks: tuple[PlanTask, ...] = ()
    assumptions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    risk_areas: tuple[str, ...] = ()
    validation_requirements: tuple[str, ...] = ()
    prompt_version: str = "v1"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError("plan revisions are 1-based")
        if not self.tasks:
            raise PlanValidationError("a plan must contain at least one task")
        self._validate_graph()

    # -- graph ----------------------------------------------------------
    def _validate_graph(self) -> None:
        keys = [t.key for t in self.tasks]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise PlanValidationError("duplicate task keys", keys=sorted(duplicates))

        known = set(keys)
        for task in self.tasks:
            dangling = [d for d in task.depends_on if d not in known]
            if dangling:
                raise PlanValidationError(
                    "task depends on unknown tasks", task=task.key, missing=dangling
                )
        # Building the layers is itself the cycle check.
        self.execution_layers()

    def task(self, key: str) -> PlanTask:
        for candidate in self.tasks:
            if candidate.key == key:
                return candidate
        raise PlanValidationError("unknown task", task=key)

    def execution_layers(self) -> tuple[tuple[PlanTask, ...], ...]:
        """Group tasks into layers; tasks inside a layer may run concurrently.

        Raises ``PlanValidationError`` when the dependency graph contains a
        cycle, which is the only way a plan can deadlock the orchestrator.
        """
        remaining: dict[str, PlanTask] = {t.key: t for t in self.tasks}
        satisfied: set[str] = set()
        layers: list[tuple[PlanTask, ...]] = []

        while remaining:
            ready = tuple(
                task
                for task in remaining.values()
                if all(dep in satisfied for dep in task.depends_on)
            )
            if not ready:
                raise PlanValidationError(
                    "plan dependency graph contains a cycle",
                    tasks=sorted(remaining),
                )
            layers.append(ready)
            for task in ready:
                del remaining[task.key]
                satisfied.add(task.key)
        return tuple(layers)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def max_parallelism(self) -> int:
        """Widest layer: how much concurrency this plan could actually use."""
        return max((len(layer) for layer in self.execution_layers()), default=0)

    def with_revision(self, revision: int) -> Plan:
        return replace(self, revision=revision)

    @classmethod
    def create(
        cls,
        *,
        plan_id: PlanId,
        run_id: RunId,
        revision: int,
        objective: str,
        tasks: Sequence[PlanTask],
        now: datetime,
        assumptions: Iterable[str] = (),
        constraints: Iterable[str] = (),
        risk_areas: Iterable[str] = (),
        validation_requirements: Iterable[str] = (),
        prompt_version: str = "v1",
    ) -> Plan:
        return cls(
            id=plan_id,
            run_id=run_id,
            revision=revision,
            objective=objective,
            created_at=now,
            tasks=tuple(tasks),
            assumptions=tuple(assumptions),
            constraints=tuple(constraints),
            risk_areas=tuple(risk_areas),
            validation_requirements=tuple(validation_requirements),
            prompt_version=prompt_version,
        )
