"""Per-project tool executors.

Which tools exist depends on the project: its build and test commands come from
its own toolchain, and a project that configures none simply has no such tool.
Caching the registries keeps repeated validation stages cheap without pinning a
project to a stale toolchain, since the cache key includes it.
"""

from __future__ import annotations

from collections.abc import Sequence

from domain.entities.project import Project, ToolchainConfig
from domain.enums import AgentRole
from domain.ports.tools import SandboxExecutor, ToolExecutor, ToolRegistry
from infrastructure.tools import AllowListToolExecutor, build_tool_registry

__all__ = ["ProjectToolExecutorFactory"]


class ProjectToolExecutorFactory:
    """Builds the allow-listed tool set a role may use on a project."""

    __slots__ = ("_cache", "_sandbox")

    def __init__(self, *, sandbox: SandboxExecutor) -> None:
        self._sandbox = sandbox
        self._cache: dict[tuple[AgentRole, tuple[str | None, ...]], ToolRegistry] = {}

    @staticmethod
    def _fingerprint(toolchain: ToolchainConfig) -> tuple[str | None, ...]:
        """A hashable summary of what actually shapes the tool set.

        ``ToolchainConfig`` itself cannot key the cache: it carries an
        ``environment`` mapping, and a frozen dataclass holding a dict is not
        hashable. Only the commands change which tools exist.
        """
        return (
            toolchain.language,
            toolchain.build_command,
            toolchain.test_command,
            toolchain.static_analysis_command,
            toolchain.install_command,
            toolchain.working_subdirectory,
            "\x00".join(f"{k}={v}" for k, v in sorted(toolchain.environment.items())),
        )

    def registry(self, project: Project, *, role: AgentRole) -> ToolRegistry:
        key = (role, self._fingerprint(project.toolchain))
        cached = self._cache.get(key)
        if cached is None:
            cached = build_tool_registry(
                role=role, sandbox=self._sandbox, toolchain=project.toolchain
            )
            self._cache[key] = cached
        return cached

    def for_project(self, project: Project, *, role: AgentRole) -> ToolExecutor:
        return AllowListToolExecutor(registry=self.registry(project, role=role), role=role)

    def available_tools(self, project: Project, *, role: AgentRole) -> Sequence[str]:
        return self.registry(project, role=role).names()
