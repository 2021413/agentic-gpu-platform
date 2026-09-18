"""Versioned prompt templates (spec section 29).

Prompts are data. They are loaded from ``prompts/<role>/<version>.md`` and
rendered by substituting ``{{variable}}`` placeholders — nothing more. No
branching, no conditionals, no business rule may live in a template, and no
prompt text may be built by concatenation elsewhere in the codebase: the version
string returned with every rendering is stored on the job, and it is only a
reproducibility guarantee if the version fully determines the text.

Substitution uses ``{{name}}`` rather than ``str.format`` because templates are
full of literal braces: they embed JSON schemas and JSON examples.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from domain.enums import AgentRole
from domain.value_objects.llm import ChatMessage
from infrastructure.llm.errors import PromptNotFoundError, PromptRenderError

__all__ = [
    "DEFAULT_PROMPT_ROOT",
    "PromptLibrary",
    "PromptTemplate",
    "RenderedPrompt",
]


def _discover_prompt_root() -> Path:
    """Locate the shipped templates without hard-coding a layout.

    ``Settings.prompts_root`` is the real answer in a deployment; this is the
    fallback used by tests and by ad-hoc scripts. It walks up from this module
    because the directory sits beside ``src/`` in a checkout and beside the
    installed packages in a wheel, and a fixed ``parents[n]`` would be right in
    exactly one of the two.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "prompts"
        if (candidate / "planner").is_dir():
            return candidate
    return Path("prompts")


DEFAULT_PROMPT_ROOT = _discover_prompt_root()

_VARIABLE_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt ready to send, carrying the version that produced it."""

    role: AgentRole
    version: str
    text: str

    def as_system_message(self) -> ChatMessage:
        return ChatMessage.system(self.text)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One immutable (role, version) template."""

    role: AgentRole
    version: str
    body: str
    variables: frozenset[str]

    @classmethod
    def from_text(cls, role: AgentRole, version: str, body: str) -> PromptTemplate:
        return cls(
            role=role,
            version=version,
            body=body,
            variables=frozenset(match.group(1) for match in _VARIABLE_RE.finditer(body)),
        )

    def render(self, variables: Mapping[str, object]) -> RenderedPrompt:
        """Substitute every placeholder, or fail naming the ones missing.

        Rendering is strict on purpose: a silently empty section produces a
        plausible-looking prompt that quietly drops the objective or the patch.
        """
        missing = tuple(sorted(self.variables - set(variables)))
        if missing:
            raise PromptRenderError(self.role, self.version, missing)
        text = _VARIABLE_RE.sub(lambda m: str(variables[m.group(1)]), self.body)
        return RenderedPrompt(role=self.role, version=self.version, text=text)


class PromptLibrary:
    """Loads and caches templates from a prompt directory.

    Files are read once: prompts do not change under a running orchestrator, and
    a run that re-read them mid-flight would stop being reproducible.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        default_versions: Mapping[AgentRole, str] | None = None,
    ) -> None:
        self._root = root or DEFAULT_PROMPT_ROOT
        self._default_versions = dict(default_versions or {})
        self._cache: dict[tuple[AgentRole, str], PromptTemplate] = {}

    @property
    def root(self) -> Path:
        return self._root

    def available_versions(self, role: AgentRole) -> tuple[str, ...]:
        directory = self._root / role.value.lower()
        if not directory.is_dir():
            return ()
        return tuple(sorted((path.stem for path in directory.glob("*.md")), key=_version_key))

    def latest_version(self, role: AgentRole) -> str:
        versions = self.available_versions(role)
        if not versions:
            raise PromptNotFoundError(role)
        return versions[-1]

    def default_version(self, role: AgentRole) -> str:
        """The pinned version when one is configured, else the newest on disk."""
        return self._default_versions.get(role) or self.latest_version(role)

    def get(self, role: AgentRole, version: str | None = None) -> PromptTemplate:
        resolved = version or self.default_version(role)
        cached = self._cache.get((role, resolved))
        if cached is not None:
            return cached
        path = self._root / role.value.lower() / f"{resolved}.md"
        if not path.is_file():
            raise PromptNotFoundError(role, resolved)
        template = PromptTemplate.from_text(role, resolved, path.read_text(encoding="utf-8"))
        self._cache[(role, resolved)] = template
        return template

    def render(
        self,
        role: AgentRole,
        variables: Mapping[str, object],
        *,
        version: str | None = None,
    ) -> RenderedPrompt:
        return self.get(role, version).render(variables)


def _version_key(version: str) -> tuple[int, str]:
    """Sort numerically, so that ``v10`` follows ``v9`` instead of preceding it."""
    match = _VERSION_RE.match(version)
    return (int(match.group(1)), "") if match else (-1, version)


@lru_cache(maxsize=1)
def default_library() -> PromptLibrary:
    """Process-wide library over the repository's ``prompts/`` directory."""
    return PromptLibrary()
