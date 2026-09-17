"""Plans must be executable: no cycles, no dangling dependencies."""

from __future__ import annotations

from datetime import datetime

import pytest

from domain.entities.plan import Plan, PlanTask
from domain.exceptions import PlanValidationError
from domain.value_objects.identifiers import PlanId, RunId, TaskId


def task(key: str, *deps: str) -> PlanTask:
    return PlanTask(id=TaskId.generate(), key=key, title=key.title(), depends_on=deps)


def make_plan(now: datetime, *tasks: PlanTask, revision: int = 1) -> Plan:
    return Plan.create(
        plan_id=PlanId.generate(),
        run_id=RunId.generate(),
        revision=revision,
        objective="Implement the packet parser",
        tasks=tasks,
        now=now,
    )


def test_independent_tasks_share_one_layer(now: datetime) -> None:
    plan = make_plan(now, task("a"), task("b"), task("c", "a", "b"))
    layers = plan.execution_layers()
    assert [sorted(t.key for t in layer) for layer in layers] == [["a", "b"], ["c"]]
    assert plan.max_parallelism == 2


def test_a_linear_plan_has_no_parallelism(now: datetime) -> None:
    plan = make_plan(now, task("a"), task("b", "a"), task("c", "b"))
    assert plan.max_parallelism == 1
    assert len(plan.execution_layers()) == 3


def test_a_cycle_is_rejected(now: datetime) -> None:
    with pytest.raises(PlanValidationError, match="cycle"):
        make_plan(now, task("a", "b"), task("b", "a"))


def test_a_dangling_dependency_is_rejected(now: datetime) -> None:
    with pytest.raises(PlanValidationError, match="unknown tasks"):
        make_plan(now, task("a", "ghost"))


def test_duplicate_keys_are_rejected(now: datetime) -> None:
    with pytest.raises(PlanValidationError, match="duplicate"):
        make_plan(now, task("a"), task("a"))


def test_a_self_dependency_is_rejected() -> None:
    with pytest.raises(PlanValidationError, match="itself"):
        task("a", "a")


def test_an_empty_plan_is_rejected(now: datetime) -> None:
    with pytest.raises(PlanValidationError, match="at least one task"):
        make_plan(now)


def test_revisions_are_one_based(now: datetime) -> None:
    with pytest.raises(ValueError, match="1-based"):
        make_plan(now, task("a"), revision=0)


def test_unknown_task_lookup_raises(now: datetime) -> None:
    plan = make_plan(now, task("a"))
    assert plan.task("a").key == "a"
    with pytest.raises(PlanValidationError, match="unknown task"):
        plan.task("nope")
