"""Project request and response schemas.

Requests are the *only* place untrusted input is validated. They are kept
separate from ``application.dto.commands`` on purpose: a command is an internal
intent that may be reshaped freely, while these models are a published contract
whose every field is a compatibility promise.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from application.dto.commands import CreateProjectCommand
from application.dto.views import ProjectView
from domain.entities.project import ToolchainConfig

__all__ = ["CreateProjectRequest", "ProjectResponse", "ToolchainPayload"]


class ToolchainPayload(BaseModel):
    """How to build, test and analyse the project.

    Commands are configured, never guessed: inferring a build command and then
    reporting its failure as a code defect would poison the repair loop.

    Also the body of ``PUT /v1/projects/{id}/toolchain``, unchanged. Replacing
    the toolchain sends exactly what creating it sent, so the two paths cannot
    drift into accepting different commands, and ``extra="forbid"`` makes the
    fields this route refuses to touch — ``name``, ``local_path``,
    ``default_branch`` — a 422 rather than a silently dropped key.
    """

    model_config = ConfigDict(extra="forbid")

    language: str = Field(default="python", min_length=1, max_length=64)
    build_command: str | None = Field(default=None, max_length=1000)
    test_command: str | None = Field(default=None, max_length=1000)
    static_analysis_command: str | None = Field(default=None, max_length=1000)
    install_command: str | None = Field(default=None, max_length=1000)
    working_subdirectory: str | None = Field(default=None, max_length=500)
    environment: dict[str, str] = Field(default_factory=dict)

    def to_config(self) -> ToolchainConfig:
        return ToolchainConfig(
            language=self.language,
            build_command=self.build_command,
            test_command=self.test_command,
            static_analysis_command=self.static_analysis_command,
            install_command=self.install_command,
            working_subdirectory=self.working_subdirectory,
            environment=dict(self.environment),
        )


class CreateProjectRequest(BaseModel):
    """A repository the platform is allowed to work on."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    repository_url: str | None = Field(default=None, max_length=2000)
    local_path: str | None = Field(default=None, max_length=2000)
    default_branch: str = Field(default="main", min_length=1, max_length=255)
    toolchain: ToolchainPayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _requires_a_source(self) -> Self:
        """Duplicates a domain invariant, deliberately.

        ``Project`` refuses to exist without a source and that stays its last
        line of defence. Re-stating it here is what turns "the server exploded"
        (500) into "your payload is incomplete" (422), which is the boundary's
        entire job.
        """
        if not self.repository_url and not self.local_path:
            raise ValueError("either repository_url or local_path must be provided")
        return self

    def to_command(self) -> CreateProjectCommand:
        return CreateProjectCommand(
            name=self.name,
            repository_url=self.repository_url,
            local_path=self.local_path,
            default_branch=self.default_branch,
            toolchain=self.toolchain.to_config() if self.toolchain else None,
            metadata=dict(self.metadata),
        )


class ToolchainResponse(BaseModel):
    """The commands this project runs against the caller's code."""

    language: str
    build_command: str | None
    test_command: str | None
    static_analysis_command: str | None
    install_command: str | None
    working_subdirectory: str | None

    @classmethod
    def of(cls, toolchain: ToolchainConfig) -> ToolchainResponse:
        return cls(
            language=toolchain.language,
            build_command=toolchain.build_command,
            test_command=toolchain.test_command,
            static_analysis_command=toolchain.static_analysis_command,
            install_command=toolchain.install_command,
            working_subdirectory=toolchain.working_subdirectory,
        )


class ProjectResponse(BaseModel):
    """A registered project."""

    id: UUID
    name: str
    repository_url: str | None
    default_branch: str
    language: str
    created_at: datetime
    toolchain: ToolchainResponse

    @classmethod
    def of(cls, view: ProjectView) -> ProjectResponse:
        return cls(
            id=view.id.value,
            name=view.name,
            repository_url=view.repository_url,
            default_branch=view.default_branch,
            language=view.language,
            created_at=view.created_at,
            toolchain=ToolchainResponse.of(view.toolchain),
        )
