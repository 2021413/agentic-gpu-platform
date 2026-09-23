"""Project endpoints.

Each handler does exactly three things: build a command or read the path
parameters, call one use case, render the view. Anything that looks like a
decision belongs in the application layer, not here.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from application.dto.commands import ReplaceProjectToolchainCommand
from domain.value_objects.identifiers import ProjectId
from interfaces.api.dependencies.providers import (
    CreateProjectDep,
    GetProjectDep,
    ListProjectsDep,
    ReplaceProjectToolchainDep,
)
from interfaces.api.schemas.common import ProblemDetails
from interfaces.api.schemas.projects import (
    CreateProjectRequest,
    ProjectResponse,
    ToolchainPayload,
)

__all__ = ["router"]

router = APIRouter(prefix="/v1/projects", tags=["projects"])


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a project",
    description=(
        "Creating a project is idempotent on its name: sending the same name "
        "twice returns the existing project rather than a duplicate."
    ),
    responses={422: {"model": ProblemDetails}},
)
async def create_project(
    payload: CreateProjectRequest, use_case: CreateProjectDep
) -> ProjectResponse:
    return ProjectResponse.of(await use_case.execute(payload.to_command()))


@router.put(
    "/{project_id}/toolchain",
    response_model=ProjectResponse,
    summary="Replace a project's toolchain",
    description=(
        "Replaces the whole toolchain: what you send is what the project will "
        "run, and an omitted command means the project no longer has one. A "
        "project's source, branch and name are not part of this resource and "
        "cannot be changed — those identify the code its run history was "
        "produced against. Refused with 409 while any run of this project is "
        "still in flight, because a run reads these commands every time it "
        "validates a candidate."
    ),
    responses={404: {"model": ProblemDetails}, 409: {"model": ProblemDetails}},
)
async def replace_project_toolchain(
    project_id: UUID, payload: ToolchainPayload, use_case: ReplaceProjectToolchainDep
) -> ProjectResponse:
    command = ReplaceProjectToolchainCommand(
        project_id=ProjectId(project_id), toolchain=payload.to_config()
    )
    return ProjectResponse.of(await use_case.execute(command))


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Fetch a project",
    responses={404: {"model": ProblemDetails}},
)
async def get_project(project_id: UUID, use_case: GetProjectDep) -> ProjectResponse:
    return ProjectResponse.of(await use_case.execute(ProjectId(project_id)))


@router.get(
    "",
    response_model=list[ProjectResponse],
    summary="List projects",
)
async def list_projects(
    use_case: ListProjectsDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ProjectResponse]:
    views = await use_case.execute(limit=limit, offset=offset)
    return [ProjectResponse.of(view) for view in views]
