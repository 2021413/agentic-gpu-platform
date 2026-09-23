"""Project use cases."""

from __future__ import annotations

from collections.abc import Sequence

from application.dto.commands import CreateProjectCommand, ReplaceProjectToolchainCommand
from application.dto.views import ProjectView
from application.ports import UnitOfWorkFactory
from domain.entities.project import Project
from domain.exceptions import EntityNotFoundError, ProjectNotModifiableError
from domain.ports.clock import Clock, IdGenerator
from domain.value_objects.identifiers import ProjectId

__all__ = [
    "CreateProjectUseCase",
    "GetProjectUseCase",
    "ListProjectsUseCase",
    "ReplaceProjectToolchainUseCase",
]


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


class ReplaceProjectToolchainUseCase:
    """Correct the commands a project builds, tests and analyses itself with.

    This exists because a toolchain used to be frozen at creation: creating a
    project that already exists returns the existing record, so a caller who
    fixed a wrong test command had no way to tell the platform, and spent a run
    executing the old one. Making a second project instead would have worked,
    at the price of leaving the run history behind.
    """

    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, command: ReplaceProjectToolchainCommand) -> ProjectView:
        async with self._uow_factory() as uow:
            project = await uow.projects.get(command.project_id)
            if project is None:
                raise EntityNotFoundError("Project", command.project_id)

            # Refused, rather than applied and hoped for: a run reads the
            # toolchain each time it validates a candidate, so a change landing
            # mid-run would judge that run's candidates by two different
            # definitions of "passing" — the ones coded before the change
            # discarded for failing a check the later ones never ran. Waiting
            # for a run to end (or cancelling it) is a decision only the caller
            # can take, so the refusal hands back the runs to act on.
            #
            # ``list_active`` rather than ``list_by_project``: the latter is
            # paginated over the project's whole history, and an in-flight run
            # older than the page would slip through. The active set is small
            # and complete, which is what a guard needs.
            in_flight = [
                run.id for run in await uow.runs.list_active() if run.project_id == project.id
            ]
            if in_flight:
                raise ProjectNotModifiableError(project.id, in_flight)

            project.replace_toolchain(command.toolchain)
            await uow.projects.update_toolchain(project)
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
