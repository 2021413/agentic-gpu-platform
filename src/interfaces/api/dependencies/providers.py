"""FastAPI providers: the seams ``bootstrap`` and the tests plug into.

Every provider resolves out of the single ``ApiDependencies`` attached to
``app.state``. Two override points follow from that, both supported on purpose:

* replace the whole container — ``app.state.dependencies = ApiDependencies(...)``
  or ``app.dependency_overrides[get_dependencies] = lambda: container``;
* replace one collaborator — ``app.dependency_overrides[get_create_run] = ...``,
  which is what a test wanting a single stub should use.

Routes depend on the ``Annotated`` aliases exported here, never on
``app.state``: that indirection is what makes both override points work.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from starlette.requests import Request

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
from interfaces.api.dependencies.container import ApiDependencies
from interfaces.api.dependencies.readiness import ReadinessProbe
from interfaces.worker_api.auth import ServiceAuthenticator

__all__ = [
    "CancelRunDep",
    "CreateProjectDep",
    "CreateRunDep",
    "DeregisterWorkerDep",
    "DrainWorkerDep",
    "EventBusDep",
    "GetProjectDep",
    "GetRunDep",
    "ListCandidatesDep",
    "ListProjectsDep",
    "ListRunEventsDep",
    "ListWorkersDep",
    "ReadinessDep",
    "RegisterWorkerDep",
    "ReplaceProjectToolchainDep",
    "ServiceAuthenticatorDep",
    "WorkerHeartbeatDep",
    "get_dependencies",
]

_STATE_ATTRIBUTE = "dependencies"


def get_dependencies(request: Request) -> ApiDependencies:
    """The container ``bootstrap`` attached to the application.

    Failing loudly here — rather than returning ``None`` and exploding deeper —
    turns a wiring mistake into one unambiguous message at the first request
    instead of an ``AttributeError`` inside a route.
    """
    container = getattr(request.app.state, _STATE_ATTRIBUTE, None)
    if container is None:
        raise RuntimeError(
            "no ApiDependencies on app.state.dependencies: the composition root "
            "must call create_api(dependencies=...) or set app.state.dependencies"
        )
    if not isinstance(container, ApiDependencies):  # pragma: no cover - defensive
        raise RuntimeError(
            f"app.state.dependencies must be an ApiDependencies, got {type(container).__name__}"
        )
    return container


ContainerDep = Annotated[ApiDependencies, Depends(get_dependencies)]


def get_create_project(container: ContainerDep) -> CreateProjectUseCase:
    return container.create_project


def get_get_project(container: ContainerDep) -> GetProjectUseCase:
    return container.get_project


def get_list_projects(container: ContainerDep) -> ListProjectsUseCase:
    return container.list_projects


def get_replace_project_toolchain(container: ContainerDep) -> ReplaceProjectToolchainUseCase:
    return container.replace_project_toolchain


def get_create_run(container: ContainerDep) -> CreateRunUseCase:
    return container.create_run


def get_get_run(container: ContainerDep) -> GetRunUseCase:
    return container.get_run


def get_cancel_run(container: ContainerDep) -> CancelRunUseCase:
    return container.cancel_run


def get_list_candidates(container: ContainerDep) -> ListCandidatesUseCase:
    return container.list_candidates


def get_list_runs(container: ContainerDep) -> ListRunsUseCase:
    return container.list_runs


def get_candidate_patch(container: ContainerDep) -> GetCandidatePatchUseCase:
    return container.candidate_patch


def get_list_reviews(container: ContainerDep) -> ListReviewsUseCase:
    return container.list_reviews


def get_approve_run(container: ContainerDep) -> ApproveRunUseCase:
    return container.approve_run


def get_list_run_events(container: ContainerDep) -> ListRunEventsUseCase:
    return container.list_run_events


def get_list_workers(container: ContainerDep) -> ListWorkersUseCase:
    return container.list_workers


def get_register_worker(container: ContainerDep) -> RegisterWorkerUseCase:
    return container.register_worker


def get_worker_heartbeat(container: ContainerDep) -> HeartbeatUseCase:
    return container.worker_heartbeat


def get_drain_worker(container: ContainerDep) -> DrainWorkerUseCase:
    return container.drain_worker


def get_deregister_worker(container: ContainerDep) -> DeregisterWorkerUseCase:
    return container.deregister_worker


def get_event_bus(container: ContainerDep) -> EventBus:
    return container.event_bus


def get_readiness_probe(container: ContainerDep) -> ReadinessProbe:
    return container.readiness


def get_service_authenticator(container: ContainerDep) -> ServiceAuthenticator:
    return container.service_authenticator


CreateProjectDep = Annotated[CreateProjectUseCase, Depends(get_create_project)]
GetProjectDep = Annotated[GetProjectUseCase, Depends(get_get_project)]
ListProjectsDep = Annotated[ListProjectsUseCase, Depends(get_list_projects)]
ReplaceProjectToolchainDep = Annotated[
    ReplaceProjectToolchainUseCase, Depends(get_replace_project_toolchain)
]
CreateRunDep = Annotated[CreateRunUseCase, Depends(get_create_run)]
GetRunDep = Annotated[GetRunUseCase, Depends(get_get_run)]
CancelRunDep = Annotated[CancelRunUseCase, Depends(get_cancel_run)]
ListCandidatesDep = Annotated[ListCandidatesUseCase, Depends(get_list_candidates)]
ListRunsDep = Annotated[ListRunsUseCase, Depends(get_list_runs)]
CandidatePatchDep = Annotated[GetCandidatePatchUseCase, Depends(get_candidate_patch)]
ListReviewsDep = Annotated[ListReviewsUseCase, Depends(get_list_reviews)]
ApproveRunDep = Annotated[ApproveRunUseCase, Depends(get_approve_run)]
ListRunEventsDep = Annotated[ListRunEventsUseCase, Depends(get_list_run_events)]
ListWorkersDep = Annotated[ListWorkersUseCase, Depends(get_list_workers)]
RegisterWorkerDep = Annotated[RegisterWorkerUseCase, Depends(get_register_worker)]
WorkerHeartbeatDep = Annotated[HeartbeatUseCase, Depends(get_worker_heartbeat)]
DrainWorkerDep = Annotated[DrainWorkerUseCase, Depends(get_drain_worker)]
DeregisterWorkerDep = Annotated[DeregisterWorkerUseCase, Depends(get_deregister_worker)]
EventBusDep = Annotated[EventBus, Depends(get_event_bus)]
ReadinessDep = Annotated[ReadinessProbe, Depends(get_readiness_probe)]
ServiceAuthenticatorDep = Annotated[ServiceAuthenticator, Depends(get_service_authenticator)]
