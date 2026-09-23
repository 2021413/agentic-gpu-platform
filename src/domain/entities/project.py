"""The project a run operates on."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from domain.value_objects.identifiers import ProjectId

__all__ = ["Project", "ToolchainConfig"]


@dataclass(frozen=True, slots=True)
class ToolchainConfig:
    """How to build, test and analyse this project.

    Commands are stored per project rather than inferred, because guessing a
    build command and reporting its failure as a code defect would poison the
    repair loop.
    """

    language: str = "python"
    build_command: str | None = None
    test_command: str | None = None
    static_analysis_command: str | None = None
    install_command: str | None = None
    working_subdirectory: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    @property
    def can_build(self) -> bool:
        return bool(self.build_command)

    @property
    def can_test(self) -> bool:
        return bool(self.test_command)


@dataclass(slots=True)
class Project:
    """A source repository the platform performs agentic work on."""

    id: ProjectId
    name: str
    repository_url: str | None
    default_branch: str
    created_at: datetime
    toolchain: ToolchainConfig = field(default_factory=ToolchainConfig)
    local_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("project name must not be blank")
        if not self.repository_url and not self.local_path:
            raise ValueError("a project needs either a repository_url or a local_path")

    def replace_toolchain(self, toolchain: ToolchainConfig) -> None:
        """Swap the commands, and only the commands.

        The toolchain is the one part of a project that can legitimately be
        corrected: it says *how* the same code is built and judged, and getting
        it wrong costs a run that validated the wrong thing. Everything else is
        identity. ``local_path`` and ``repository_url`` say *which* code is
        worked on and ``default_branch`` says where the work lands, so editing
        them in place would silently re-point the run history already attached
        to this project at a different tree — the records would survive and
        stop meaning anything. Those need a new project; this does not.
        """
        self.toolchain = toolchain

    @classmethod
    def create(
        cls,
        *,
        project_id: ProjectId,
        name: str,
        now: datetime,
        repository_url: str | None = None,
        local_path: str | None = None,
        default_branch: str = "main",
        toolchain: ToolchainConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Project:
        return cls(
            id=project_id,
            name=name.strip(),
            repository_url=repository_url,
            local_path=local_path,
            default_branch=default_branch,
            created_at=now,
            toolchain=toolchain or ToolchainConfig(),
            metadata=dict(metadata or {}),
        )
