"""Project endpoints.

Each handler does exactly three things: build a command or read the path
parameters, call one use case, render the view. Anything that looks like a
decision belongs in the application layer, not here.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from application.dto.commands import (
    ReplaceProjectToolchainCommand,
    UploadedFile,
    UploadProjectCommand,
)
from domain.value_objects.identifiers import ProjectId
from interfaces.api.dependencies.providers import (
    CreateProjectDep,
    GetProjectDep,
    ListProjectsDep,
    ReplaceProjectToolchainDep,
    UploadProjectDep,
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


@router.post(
    "/upload",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a project from uploaded files",
    description=(
        "Send one .zip, or several files whose filenames are their paths "
        "relative to the project root. The server chooses where the files "
        "live, commits them as the baseline the agents branch from, and "
        "detects the toolchain from what it finds unless a field is given. "
        "Not idempotent on the name: a second upload under an existing name "
        "is a 409, because it may carry different code."
    ),
    responses={
        400: {"model": ProblemDetails},
        409: {"model": ProblemDetails},
        422: {"model": ProblemDetails},
    },
)
async def upload_project(
    *,
    use_case: UploadProjectDep,
    name: Annotated[str, Form(min_length=1, max_length=200)],
    files: Annotated[list[UploadFile], File()],
    language: Annotated[str | None, Form(max_length=40)] = None,
    build_command: Annotated[str | None, Form(max_length=1000)] = None,
    test_command: Annotated[str | None, Form(max_length=1000)] = None,
) -> ProjectResponse:
    uploaded = [
        UploadedFile(path=upload.filename or "", content=await upload.read()) for upload in files
    ]
    command = UploadProjectCommand(
        name=name,
        files=uploaded,
        language=language or None,
        # An empty form field means "not given", not "no command": a browser
        # sends every field it renders, and a blank one must fall back to
        # detection rather than silently disable a stage.
        build_command=build_command or None,
        test_command=test_command or None,
    )
    return ProjectResponse.of(await use_case.execute(command))


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
