"""Project use cases."""

from __future__ import annotations

from collections.abc import Sequence

from application.dto.commands import CreateProjectCommand
from application.dto.views import ProjectView
from application.ports import UnitOfWorkFactory
from domain.entities.project import Project
from domain.exceptions import EntityNotFoundError
from domain.ports.clock import Clock, IdGenerator
from domain.value_objects.identifiers import ProjectId

__all__ = ["CreateProjectUseCase", "GetProjectUseCase", "ListProjectsUseCase"]


class CreateProjectUseCase:
    """Register a repository the platform may work on."""

    def __init__(self, *, uow_factory: UnitOfWorkFactory, clock: Clock, ids: IdGenerator) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._ids = ids

    async def execute(self, command: CreateProjectCommand) -> ProjectView:
        async with self._uow_factory() as uow:
            existing = await uow.projects.get_by_name(command.name)
            if existing is not None:
                # Creating a project is naturally idempotent on its name: the
                # caller gets the same project instead of a duplicate.
                return ProjectView.of(existing)

            project = Project.create(
                project_id=self._ids.next_id(ProjectId),
                name=command.name,
                now=self._clock.now(),
                repository_url=command.repository_url,
                local_path=command.local_path,
                default_branch=command.default_branch,
                toolchain=command.toolchain,
                metadata=command.metadata,
            )
            await uow.projects.add(project)
            await uow.commit()
            return ProjectView.of(project)


class GetProjectUseCase:
    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, project_id: ProjectId) -> ProjectView:
        async with self._uow_factory() as uow:
            project = await uow.projects.get(project_id)
            if project is None:
                raise EntityNotFoundError("Project", project_id)
            return ProjectView.of(project)


class ListProjectsUseCase:
    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, *, limit: int = 100, offset: int = 0) -> Sequence[ProjectView]:
        async with self._uow_factory() as uow:
            projects = await uow.projects.list_all(limit=limit, offset=offset)
            return [ProjectView.of(p) for p in projects]
