"""The prompts must ask for what the code relies on (spec section 30).

`repair_brief()` renders `(file:line)` in front of every finding, and a real
run produced six findings with neither: the reviewer prompt never asked. The
coder then received a list of complaints with nowhere to look.

These check the templates against the fields the platform actually reads —
the same class of mismatch as a client declaring a response field the server
never sends, and just as silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROMPTS = Path("prompts")


def template(role: str) -> str:
    return (PROMPTS / role / "v1.md").read_text(encoding="utf-8")


def test_the_reviewer_is_told_to_locate_its_findings() -> None:
    body = template("reviewer")

    assert "`file`" in body, "nothing asks the reviewer where the problem is"
    assert "`line`" in body


def test_the_reviewer_is_told_why_the_location_matters() -> None:
    """An instruction without a reason is the first thing a model drops."""
    body = template("reviewer").lower()

    assert "coder" in body and ("where" in body or "searching" in body)


@pytest.mark.parametrize("role", ["planner", "coder", "reviewer"])
def test_every_prompt_renders_its_retry_context(role: str) -> None:
    """The slot that carries the previous failure back to the model.

    A template that drops it turns every retry into a repeat of the attempt
    that just failed — which is exactly what the repair loop was doing.
    """
    assert "{{retry_context}}" in template(role)


def test_the_coder_prompt_names_no_model_vendor() -> None:
    """Model neutrality: a prompt written for one engine is a lock-in."""
    body = template("coder").lower()

    for vendor in ("qwen", "gpt-4", "claude", "llama", "mistral"):
        assert vendor not in body, f"the coder prompt names {vendor}"


def test_the_coder_is_told_to_read_the_tool_output() -> None:
    """It now receives build and test output; a slot nothing points at is a
    slot a model skims past."""
    body = template("coder")

    assert "Recent tool output" in body, "the coder is never told the output is there"
    assert "exit code" in body.lower()
