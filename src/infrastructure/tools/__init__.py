"""Deterministic tool layer and sandboxing (spec sections 7 and 31).

Tools are not LLMs: their results are the only admissible evidence that a build
compiled or that a test suite passed. Everything a model can trigger from here
is an explicit capability with a JSON schema, an allow-list and a sandbox.
"""

from __future__ import annotations

from infrastructure.tools.commands import (
    RunCommandTool,
    ToolchainCommandTool,
    build_toolchain_tools,
)
from infrastructure.tools.files import EditFileTool, ReadFileTool
from infrastructure.tools.registry import (
    ROLE_TOOLS,
    AllowListToolExecutor,
    StaticToolRegistry,
    build_tool_registry,
)
from infrastructure.tools.repository_context import RipgrepRepositoryContextProvider
from infrastructure.tools.sandbox import (
    DockerSandboxExecutor,
    SubprocessSandboxExecutor,
    create_sandbox_executor,
)
from infrastructure.tools.search import SearchRepositoryTool, SearchSymbolTool, TextSearchBackend
from infrastructure.tools.toolchain import PROFILES, ToolchainProfile, profile, with_sanitizer
from infrastructure.tools.vcs import ApplyPatchTool, GitDiffTool, GitStatusTool

__all__ = [
    "PROFILES",
    "ROLE_TOOLS",
    "AllowListToolExecutor",
    "ApplyPatchTool",
    "DockerSandboxExecutor",
    "EditFileTool",
    "GitDiffTool",
    "GitStatusTool",
    "ReadFileTool",
    "RipgrepRepositoryContextProvider",
    "RunCommandTool",
    "SearchRepositoryTool",
    "SearchSymbolTool",
    "StaticToolRegistry",
    "SubprocessSandboxExecutor",
    "TextSearchBackend",
    "ToolchainCommandTool",
    "ToolchainProfile",
    "build_tool_registry",
    "build_toolchain_tools",
    "create_sandbox_executor",
    "profile",
    "with_sanitizer",
]
