"""Text search over a workspace: ripgrep when present, grep otherwise.

The fallback is not decoration — plenty of container images ship without
ripgrep, and a search tool that simply disappears would silently change how an
agent explores a repository. The two engines are close enough for the patterns
agents write, and the difference is reported in the result metadata rather than
hidden: ripgrep uses Rust regex syntax and honours ``.gitignore``, GNU grep uses
POSIX extended syntax and needs explicit exclusions.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace

from domain.exceptions import ToolExecutionError
from domain.ports.tools import SandboxExecutor
from domain.value_objects.tools import ExecutionLimits, ToolInvocation, ToolKind, ToolResult
from domain.value_objects.workspace import WorkspaceHandle
from infrastructure.tools.base import (
    bool_argument,
    int_argument,
    json_schema,
    resolve_in_workspace,
    string_argument,
)

__all__ = ["SearchMatch", "SearchRepositoryTool", "SearchSymbolTool", "TextSearchBackend"]

_EXCLUDED_DIRECTORIES = (".git", "node_modules", "__pycache__", ".venv", "build", "dist")
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]{0,127}$")
_NO_MATCH_EXIT = 1


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """One matching line, addressed the way an editor would address it."""

    path: str
    line_number: int
    line: str


class TextSearchBackend:
    """Chooses a search engine once and reports which one it chose."""

    __slots__ = ("_engine", "_executable", "_sandbox")

    def __init__(
        self, *, sandbox: SandboxExecutor, ripgrep: str = "rg", grep: str = "grep"
    ) -> None:
        self._sandbox = sandbox
        found = shutil.which(ripgrep)
        self._engine = "ripgrep" if found else "grep"
        self._executable = found or grep

    @property
    def engine(self) -> str:
        return self._engine

    async def run(
        self,
        *,
        workspace: WorkspaceHandle,
        pattern: str,
        path: str = ".",
        limits: ExecutionLimits,
        max_results: int = 200,
        case_sensitive: bool = True,
        glob: str | None = None,
    ) -> ToolResult:
        """Search and return the raw result, exit code normalised.

        "No match" is exit 1 for both engines; reported as such it would look
        like a failure to an agent and poison the repair loop, so it becomes a
        successful result with ``match_count: 0``.
        """
        argv = self._argv(pattern, path, max_results, case_sensitive, glob)
        result = await self._sandbox.run(
            command=argv, workspace=workspace, limits=limits, environment=None
        )
        matches = parse_matches(result.stdout, limit=max_results)
        exit_code = 0 if result.exit_code == _NO_MATCH_EXIT and not matches else result.exit_code
        return replace(
            result,
            kind=ToolKind.SEARCH,
            exit_code=exit_code,
            metadata={
                **dict(result.metadata),
                "engine": self._engine,
                "match_count": len(matches),
                "pattern": pattern,
            },
        )

    async def matches(
        self,
        *,
        workspace: WorkspaceHandle,
        pattern: str,
        limits: ExecutionLimits,
        path: str = ".",
        max_results: int = 200,
    ) -> tuple[SearchMatch, ...]:
        result = await self.run(
            workspace=workspace,
            pattern=pattern,
            path=path,
            limits=limits,
            max_results=max_results,
        )
        if result.exit_code not in (0, _NO_MATCH_EXIT):
            raise ToolExecutionError(
                "search", "search engine failed", engine=self._engine, stderr=result.stderr[:1000]
            )
        return parse_matches(result.stdout, limit=max_results)

    def _argv(
        self,
        pattern: str,
        path: str,
        max_results: int,
        case_sensitive: bool,
        glob: str | None,
    ) -> tuple[str, ...]:
        if self._engine == "ripgrep":
            args = [
                self._executable,
                "--line-number",
                "--no-heading",
                "--color",
                "never",
                "--max-count",
                str(max_results),
            ]
            if not case_sensitive:
                args.append("--ignore-case")
            if glob:
                args.extend(["--glob", glob])
            args.extend(["--regexp", pattern, "--", path])
            return tuple(args)

        args = [self._executable, "-r", "-n", "-I", "-E", "-m", str(max_results)]
        args.extend(f"--exclude-dir={name}" for name in _EXCLUDED_DIRECTORIES)
        if not case_sensitive:
            args.append("-i")
        if glob:
            args.append(f"--include={glob}")
        args.extend(["-e", pattern, path])
        return tuple(args)


def parse_matches(stdout: str, *, limit: int) -> tuple[SearchMatch, ...]:
    """Parse ``path:line:text`` output, which both engines emit identically."""
    matches: list[SearchMatch] = []
    for raw in stdout.splitlines():
        parts = raw.split(":", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        path = parts[0].removeprefix("./")
        matches.append(SearchMatch(path=path, line_number=int(parts[1]), line=parts[2]))
        if len(matches) >= limit:
            break
    return tuple(matches)


class SearchRepositoryTool:
    """Regex search across the workspace."""

    name = "search_repository"
    kind = ToolKind.SEARCH
    description = (
        "Search the workspace for a regular expression and return matching lines "
        "as path:line:text. Use it before reading files, to find where something lives."
    )

    __slots__ = ("_backend",)

    def __init__(self, backend: TextSearchBackend) -> None:
        self._backend = backend

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {
                    "type": "string",
                    "description": "Directory or file to search, relative to the workspace root.",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Only search files matching this glob, e.g. '*.py'.",
                },
                "case_sensitive": {"type": "boolean", "default": True},
                "max_results": {"type": "integer", "default": 200, "minimum": 1},
            },
            required=["pattern"],
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        arguments = invocation.arguments
        pattern = string_argument(self.name, arguments, "pattern")
        path = string_argument(self.name, arguments, "path", default=".")
        resolve_in_workspace(self.name, workspace, path)
        glob = arguments.get("glob")
        result = await self._backend.run(
            workspace=workspace,
            pattern=pattern,
            path=path,
            limits=invocation.limits,
            max_results=int_argument(self.name, arguments, "max_results", default=200, minimum=1),
            case_sensitive=bool_argument(self.name, arguments, "case_sensitive", default=True),
            glob=glob if isinstance(glob, str) else None,
        )
        return replace(result, tool=self.name, kind=self.kind)


class SearchSymbolTool:
    """Find where a symbol is defined.

    This is regex-based and says so: it finds definitions the way an experienced
    developer would grep for them, not the way a compiler resolves them.
    Tree-sitter or a language server would replace this class without changing
    its contract — which is exactly why it is a separate tool rather than a
    prompt instruction telling the model to grep.
    """

    name = "search_symbol"
    kind = ToolKind.SEARCH
    description = (
        "Find likely definitions of a symbol (function, class, struct, typedef) "
        "in the workspace. Heuristic and regex-based, not a compiler."
    )

    __slots__ = ("_backend",)

    def __init__(self, backend: TextSearchBackend) -> None:
        self._backend = backend

    @property
    def parameters_schema(self) -> Mapping[str, object]:
        return json_schema(
            {
                "symbol": {"type": "string", "description": "Identifier to locate."},
                "path": {"type": "string", "default": "."},
                "max_results": {"type": "integer", "default": 50, "minimum": 1},
            },
            required=["symbol"],
        )

    async def execute(
        self, *, invocation: ToolInvocation, workspace: WorkspaceHandle
    ) -> ToolResult:
        arguments = invocation.arguments
        symbol = string_argument(self.name, arguments, "symbol")
        if not _SYMBOL_RE.match(symbol):
            # Identifiers only: the symbol is interpolated into a regex, and an
            # arbitrary string would be a pattern-injection vector.
            raise ToolExecutionError(
                self.name, "symbol must be a plain identifier", symbol=symbol[:80]
            )
        path = string_argument(self.name, arguments, "path", default=".")
        resolve_in_workspace(self.name, workspace, path)
        result = await self._backend.run(
            workspace=workspace,
            pattern=definition_pattern(symbol),
            path=path,
            limits=invocation.limits,
            max_results=int_argument(self.name, arguments, "max_results", default=50, minimum=1),
        )
        return replace(
            result,
            tool=self.name,
            kind=self.kind,
            metadata={**dict(result.metadata), "symbol": symbol},
        )


def definition_pattern(symbol: str) -> str:
    """A pattern matching the shapes a definition takes in mainstream languages.

    Written in the intersection of POSIX ERE and Rust regex so that both search
    engines understand it unchanged.
    """
    keywords = "def|class|struct|union|enum|typedef|interface|fn|func|impl|trait|type"
    return (
        rf"(({keywords})[[:space:]]+{symbol}\b)"
        rf"|(\b{symbol}[[:space:]]*\()"
        rf"|(\b{symbol}[[:space:]]*=[[:space:]]*(lambda|function)\b)"
    )
