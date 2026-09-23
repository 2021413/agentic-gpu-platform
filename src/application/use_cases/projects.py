"""Project use cases."""

from __future__ import annotations

from collections.abc import Sequence

from application.dto.commands import (
    CreateProjectCommand,
    ReplaceProjectToolchainCommand,
    UploadProjectCommand,
)
from application.dto.views import ProjectView
from application.ports import CommandProbe, ProjectFilesStore, UnitOfWorkFactory
from domain.entities.project import Project, ToolchainConfig
from domain.exceptions import (
    EntityNotFoundError,
    ProjectAlreadyExistsError,
    ProjectNotModifiableError,
    ToolchainCommandUnavailableError,
)
from domain.ports.clock import Clock, IdGenerator
from domain.value_objects.identifiers import ProjectId

__all__ = [
    "CreateProjectUseCase",
    "GetProjectUseCase",
    "ListProjectsUseCase",
    "ReplaceProjectToolchainUseCase",
    "UploadProjectUseCase",
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


def _refuse_unstartable(toolchain: ToolchainConfig, probe: CommandProbe) -> None:
    """Refuse a toolchain whose commands cannot start, naming the first word.

    Only ever a refusal, never a silent drop: turning a command the caller
    typed into "no command" would record validation as "did not run" and let
    a run proceed on the belief that nothing was asked of it.
    """
    for field, command in (
        ("build_command", toolchain.build_command),
        ("test_command", toolchain.test_command),
        ("static_analysis_command", toolchain.static_analysis_command),
    ):
        executable = probe(command)
        if executable is not None and command is not None:
            raise ToolchainCommandUnavailableError(field, command, executable)


class UploadProjectUseCase:
    """Create a project from uploaded files.

    Not idempotent on the name, unlike creation by reference. Two uploads under
    one name may carry different files, and answering the second with the first
    project would run the caller's agents on code they did not send. A name
    collision is a conflict, and the caller picks another name.

    The files are materialised *before* the record exists, so a refused upload
    leaves nothing in the database — and the directory is removed by the store
    if anything after writing fails.
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        clock: Clock,
        ids: IdGenerator,
        files: ProjectFilesStore,
        command_probe: CommandProbe,
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._ids = ids
        self._files = files
        self._probe = command_probe

    async def execute(self, command: UploadProjectCommand) -> ProjectView:
        async with self._uow_factory() as uow:
            existing = await uow.projects.get_by_name(command.name)
            if existing is not None:
                raise ProjectAlreadyExistsError(command.name, existing.id)

        # The caller's own commands are checked before a byte is written: a
        # sentence typed where a command belongs must not cost an upload, let
        # alone a run.
        _refuse_unstartable(
            ToolchainConfig(
                language=command.language or "unknown",
                build_command=command.build_command,
                test_command=command.test_command,
            ),
            self._probe,
        )

        project_id = self._ids.next_id(ProjectId)
        stored = await self._files.materialise(project_id, command.files)

        # Detection fills in whatever the caller left unsaid, field by field:
        # a caller who knows the test command but not the language should not
        # have to guess the language to keep the command.
        detected = stored.toolchain
        toolchain = ToolchainConfig(
            language=command.language or detected.language,
            build_command=(
                command.build_command
                if command.build_command is not None
                else detected.build_command
            ),
            test_command=(
                command.test_command if command.test_command is not None else detected.test_command
            ),
        )

        async with self._uow_factory() as uow:
            project = Project.create(
                project_id=project_id,
                name=command.name,
                now=self._clock.now(),
                local_path=str(stored.path),
                default_branch=command.default_branch,
                toolchain=toolchain,
                metadata={"source": "upload", "uploaded_files": str(stored.file_count)},
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

    def __init__(
        self, *, uow_factory: UnitOfWorkFactory, command_probe: CommandProbe | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._probe = command_probe

    async def execute(self, command: ReplaceProjectToolchainCommand) -> ProjectView:
        if self._probe is not None:
            _refuse_unstartable(command.toolchain, self._probe)
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
