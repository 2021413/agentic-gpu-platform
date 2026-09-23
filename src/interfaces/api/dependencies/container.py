"""The contract ``bootstrap`` must satisfy to start this API.

This layer builds nothing: it declares, once, every collaborator a route may
need. ``bootstrap`` (the composition root) constructs the adapters, wires the
use cases and hands over a single ``ApiDependencies`` instance — either through
``create_api(dependencies=...)`` or by assigning ``app.state.dependencies``.

Why a frozen dataclass rather than a service locator: the set of things the API
can reach is then visible in one place and checked by mypy. A route that needs
something absent from this container is a route that is about to grow logic it
should not have.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.ports import MetricsExposition
from application.use_cases.approvals import ApproveRunUseCase
from application.use_cases.projects import (
    CreateProjectUseCase,
    GetProjectUseCase,
    ListProjectsUseCase,
    ReplaceProjectToolchainUseCase,
)
from application.use_cases.runs import (
    CancelRunUseCase,
    CreateRunUseCase,
    GetCandidatePatchUseCase,
    GetRunUseCase,
    ListCandidatesUseCase,
    ListReviewsUseCase,
    ListRunEventsUseCase,
    ListRunsUseCase,
)
from application.use_cases.workers import (
    DeregisterWorkerUseCase,
    DrainWorkerUseCase,
    HeartbeatUseCase,
    ListWorkersUseCase,
    RegisterWorkerUseCase,
)
from domain.ports.event_bus import EventBus
from interfaces.api.dependencies.readiness import ReadinessProbe
from interfaces.worker_api.auth import ServiceAuthenticator

__all__ = ["ApiDependencies"]


@dataclass(frozen=True, slots=True)
class ApiDependencies:
    """Everything the HTTP layer is allowed to reach.

    ``event_bus`` is the one port used directly by a route rather than through a
    use case: SSE needs the *live* stream, and a use case returning an infinite
    async iterator would be a use case in name only. History still comes from
    ``list_run_events``, so the durable and live halves stay separate.
    """

    # projects
    create_project: CreateProjectUseCase
    get_project: GetProjectUseCase
    list_projects: ListProjectsUseCase
    replace_project_toolchain: ReplaceProjectToolchainUseCase
    # runs
    create_run: CreateRunUseCase
    get_run: GetRunUseCase
    cancel_run: CancelRunUseCase
    list_candidates: ListCandidatesUseCase
    list_runs: ListRunsUseCase
    candidate_patch: GetCandidatePatchUseCase
    list_reviews: ListReviewsUseCase
    approve_run: ApproveRunUseCase
    list_run_events: ListRunEventsUseCase
    # workers
    list_workers: ListWorkersUseCase
    register_worker: RegisterWorkerUseCase
    worker_heartbeat: HeartbeatUseCase
    drain_worker: DrainWorkerUseCase
    deregister_worker: DeregisterWorkerUseCase
    # ports and probes
    event_bus: EventBus
    metrics: MetricsExposition | None
    """The registry the exposition renders, or None when disabled."""
    readiness: ReadinessProbe
    service_authenticator: ServiceAuthenticator
