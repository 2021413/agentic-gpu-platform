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
        self._cache: dict[tuple[AgentRole, ToolchainConfig], ToolRegistry] = {}

    def registry(self, project: Project, *, role: AgentRole) -> ToolRegistry:
        key = (role, project.toolchain)
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
