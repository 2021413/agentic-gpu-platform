"""Scheduling policy: pure, deterministic, GPU-free."""

from __future__ import annotations

from datetime import datetime

from tests.conftest import make_worker

from domain.entities.worker import Worker
from domain.enums import AgentRole
from domain.services.scheduling import (
    LeastLoadedCompatibleScheduler,
    RoundRobinScheduler,
    eligible_workers,
)
from domain.value_objects.worker import JobRequirements

CODE = JobRequirements(role=AgentRole.CODER)


def busy(worker: Worker, count: int) -> Worker:
    for _ in range(count):
        worker.reserve_slot(CODE)
    return worker


def test_the_least_loaded_compatible_worker_wins(now: datetime) -> None:
    loaded = busy(make_worker(now=now, concurrency=4), 3)
    idle = make_worker(now=now, concurrency=2)
    full = busy(make_worker(now=now, concurrency=1), 1)

    chosen = LeastLoadedCompatibleScheduler().select(
        candidates=[loaded, idle, full], requirements=CODE
    )
    assert chosen is idle


def test_a_full_worker_is_not_eligible(now: datetime) -> None:
    full = busy(make_worker(now=now, concurrency=1), 1)
    assert eligible_workers([full], CODE) == []


def test_a_draining_worker_is_never_scheduled(now: datetime) -> None:
    draining = make_worker(now=now, concurrency=4)
    draining.start_draining(now)
    only_ready = make_worker(now=now, concurrency=1)

    scheduler = LeastLoadedCompatibleScheduler()
    assert scheduler.select(candidates=[draining, only_ready], requirements=CODE) is only_ready
    assert scheduler.select(candidates=[draining], requirements=CODE) is None


def test_no_compatible_worker_returns_none(now: datetime) -> None:
    planner_only = make_worker(now=now, roles=frozenset({AgentRole.PLANNER}))
    assert (
        LeastLoadedCompatibleScheduler().select(candidates=[planner_only], requirements=CODE)
        is None
    )


def test_an_empty_pool_returns_none() -> None:
    assert LeastLoadedCompatibleScheduler().select(candidates=[], requirements=CODE) is None


def test_ranking_orders_every_eligible_worker(now: datetime) -> None:
    workers = [busy(make_worker(now=now, concurrency=4), n) for n in (3, 1, 2)]
    ranked = LeastLoadedCompatibleScheduler().rank(candidates=workers, requirements=CODE)
    assert [w.active_jobs for w in ranked] == [1, 2, 3]


def test_ranking_is_deterministic(now: datetime) -> None:
    workers = [make_worker(now=now, concurrency=2) for _ in range(5)]
    scheduler = LeastLoadedCompatibleScheduler()
    first = scheduler.rank(candidates=workers, requirements=CODE)
    second = scheduler.rank(candidates=list(reversed(workers)), requirements=CODE)
    assert [w.id for w in first] == [w.id for w in second]


def test_round_robin_spreads_work(now: datetime) -> None:
    workers = [make_worker(now=now, concurrency=8) for _ in range(3)]
    scheduler = RoundRobinScheduler()
    picked = [scheduler.select(candidates=workers, requirements=CODE) for _ in range(6)]
    assert len({id(w) for w in picked}) == 3


def test_a_worker_joining_later_becomes_schedulable(now: datetime) -> None:
    """The pool is discovered on every decision, so growth needs no restart."""
    scheduler = LeastLoadedCompatibleScheduler()
    pool: list[Worker] = []
    assert scheduler.select(candidates=pool, requirements=CODE) is None

    pool.append(make_worker(now=now, concurrency=1))
    assert scheduler.select(candidates=pool, requirements=CODE) is not None
