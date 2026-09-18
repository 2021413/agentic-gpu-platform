"""Versioned prompt templates (spec section 29).

A prompt is data whose version is stored on every job, so two things are worth
testing: that rendering is strict and complete, and that the shipped templates
actually say what the spec requires them to say — role, allowed tools, expected
JSON output, constraints, retry semantics, and the ban on hidden reasoning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.enums import AgentRole
from infrastructure.llm.errors import PromptNotFoundError, PromptRenderError
from infrastructure.llm.prompts import DEFAULT_PROMPT_ROOT, PromptLibrary, PromptTemplate
from infrastructure.llm.structured import OUTPUT_MODELS, StructuredOutputParser

VARIABLES: dict[AgentRole, dict[str, str]] = {
    AgentRole.PLANNER: {
        "objective": "add retries to the uploader",
        "project_context": "a Python service",
        "constraints": "no new dependencies",
        "allowed_tools": "read_file, search",
        "output_schema": "{}",
        "retry_context": "first attempt",
    },
    AgentRole.CODER: {
        "objective": "add retries to the uploader",
        "task": "implement the retry loop",
        "workspace_context": "clean checkout at abc123",
        "allowed_tools": "read_file, edit_file, run_tests",
        "output_schema": "{}",
        "retry_context": "first attempt",
    },
    AgentRole.REVIEWER: {
        "objective": "add retries to the uploader",
        "task": "implement the retry loop",
        "patch": "diff --git a/a.py b/a.py",
        "validation_report": "build=pass tests=pass",
        "allowed_tools": "read_file",
        "output_schema": "{}",
        "retry_context": "first attempt",
    },
}


@pytest.fixture
def library() -> PromptLibrary:
    return PromptLibrary()


# -- shipped templates -----------------------------------------------------
def test_the_default_root_is_the_repository_prompt_directory() -> None:
    assert DEFAULT_PROMPT_ROOT.is_dir()
    assert DEFAULT_PROMPT_ROOT.name == "prompts"


@pytest.mark.parametrize("role", list(AgentRole))
def test_every_role_ships_a_v1_template(library: PromptLibrary, role: AgentRole) -> None:
    assert library.available_versions(role) == ("v1",)
    assert library.default_version(role) == "v1"
    assert library.get(role).version == "v1"


@pytest.mark.parametrize("role", list(AgentRole))
def test_every_template_declares_what_the_spec_requires(
    library: PromptLibrary, role: AgentRole
) -> None:
    body = library.get(role).body
    lowered = body.lower()

    assert body.startswith(f"ROLE: {role.value}")  # role
    assert "{{allowed_tools}}" in body  # allowed tools
    assert "{{output_schema}}" in body  # expected JSON output
    assert "{{retry_context}}" in body  # retry semantics
    assert "rejected" in lowered  # retry semantics, spelled out
    assert "attempts are limited" in lowered
    assert "hidden reasoning" in lowered  # spec section 21
    assert "<think>" in lowered


@pytest.mark.parametrize("role", list(AgentRole))
def test_every_template_declares_exactly_the_expected_variables(
    library: PromptLibrary, role: AgentRole
) -> None:
    assert library.get(role).variables == set(VARIABLES[role])


@pytest.mark.parametrize("role", list(AgentRole))
def test_rendering_leaves_no_placeholder_behind(library: PromptLibrary, role: AgentRole) -> None:
    parser = StructuredOutputParser(OUTPUT_MODELS[role])
    variables = {**VARIABLES[role], "output_schema": parser.schema_text()}

    rendered = library.render(role, variables)

    assert "{{" not in rendered.text
    assert rendered.role is role
    assert rendered.version == "v1"
    assert variables["objective"] in rendered.text
    assert parser.schema_name in rendered.text
    assert rendered.as_system_message().content == rendered.text


def test_the_templates_carry_no_model_name(library: PromptLibrary) -> None:
    """Model-specific wording would silently tie a prompt version to one engine."""
    for role in AgentRole:
        lowered = library.get(role).body.lower()
        for forbidden in ("qwen", "gpt-4", "claude", "llama"):
            assert forbidden not in lowered


# -- rendering -------------------------------------------------------------
def test_missing_variables_are_reported_by_name(library: PromptLibrary) -> None:
    variables = dict(VARIABLES[AgentRole.PLANNER])
    del variables["objective"]
    del variables["constraints"]

    with pytest.raises(PromptRenderError) as excinfo:
        library.render(AgentRole.PLANNER, variables)

    assert excinfo.value.missing == ("constraints", "objective")


def test_extra_variables_are_ignored(library: PromptLibrary) -> None:
    variables = {**VARIABLES[AgentRole.PLANNER], "unused": "whatever"}

    rendered = library.render(AgentRole.PLANNER, variables)

    assert "whatever" not in rendered.text


def test_substitution_tolerates_spacing_and_repeats() -> None:
    template = PromptTemplate.from_text(AgentRole.PLANNER, "v9", "{{ a }} then {{a}} then {{b}}")

    rendered = template.render({"a": 1, "b": "two"})

    assert rendered.text == "1 then 1 then two"
    assert template.variables == {"a", "b"}


def test_literal_braces_in_a_template_survive_rendering() -> None:
    """Templates embed JSON schemas; a format-string engine would choke on them."""
    template = PromptTemplate.from_text(
        AgentRole.CODER, "v1", 'Schema: {"type": "object"} for {{objective}}'
    )

    assert template.render({"objective": "x"}).text == 'Schema: {"type": "object"} for x'


# -- versions --------------------------------------------------------------
def test_an_unknown_version_is_refused(library: PromptLibrary) -> None:
    with pytest.raises(PromptNotFoundError):
        library.get(AgentRole.PLANNER, "v99")


def test_a_role_without_templates_is_refused(tmp_path: Path) -> None:
    empty = PromptLibrary(tmp_path)

    assert empty.available_versions(AgentRole.CODER) == ()
    with pytest.raises(PromptNotFoundError):
        empty.get(AgentRole.CODER)


def test_versions_are_ordered_numerically(tmp_path: Path) -> None:
    directory = tmp_path / "planner"
    directory.mkdir()
    for version in ("v1", "v2", "v10"):
        (directory / f"{version}.md").write_text("ROLE: PLANNER", encoding="utf-8")

    library = PromptLibrary(tmp_path)

    assert library.available_versions(AgentRole.PLANNER) == ("v1", "v2", "v10")
    assert library.latest_version(AgentRole.PLANNER) == "v10"


def test_a_pinned_default_version_wins_over_the_newest(tmp_path: Path) -> None:
    """Reproducibility: a run must not silently migrate to a new prompt."""
    directory = tmp_path / "planner"
    directory.mkdir()
    for version in ("v1", "v2"):
        (directory / f"{version}.md").write_text(f"ROLE: PLANNER {version}", encoding="utf-8")

    library = PromptLibrary(tmp_path, default_versions={AgentRole.PLANNER: "v1"})

    assert library.default_version(AgentRole.PLANNER) == "v1"
    assert library.get(AgentRole.PLANNER).body.endswith("v1")
    assert library.get(AgentRole.PLANNER, "v2").body.endswith("v2")


def test_a_template_is_read_once(tmp_path: Path) -> None:
    """A version must mean one exact text, even if the file changes underneath."""
    directory = tmp_path / "planner"
    directory.mkdir()
    path = directory / "v1.md"
    path.write_text("original", encoding="utf-8")
    library = PromptLibrary(tmp_path)

    first = library.get(AgentRole.PLANNER)
    path.write_text("edited on disk", encoding="utf-8")
    second = library.get(AgentRole.PLANNER)

    assert first is second
    assert second.body == "original"
