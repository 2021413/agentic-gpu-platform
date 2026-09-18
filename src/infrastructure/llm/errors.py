"""Prompt-loading errors.

Transport failures are *not* here: an inference engine that refuses, fails or
answers something unusable raises ``domain.exceptions.InferenceError``, so the
retry policy and the RFC 9457 mapper see one vocabulary whatever the adapter.
An ``httpx`` exception must never escape this package.
"""

from __future__ import annotations

from domain.exceptions import DomainError

__all__ = ["PromptNotFoundError", "PromptRenderError"]


class PromptNotFoundError(DomainError):
    """No template exists for that (role, version) pair."""

    code = "prompt_not_found"

    def __init__(self, role: object, version: str | None = None) -> None:
        super().__init__(
            "no prompt template for this role and version",
            role=str(role),
            version=version,
        )


class PromptRenderError(DomainError):
    """A template was rendered without all of the variables it declares."""

    code = "prompt_render_failed"

    def __init__(self, role: object, version: str, missing: tuple[str, ...]) -> None:
        super().__init__(
            "prompt template is missing variables",
            role=str(role),
            version=version,
            missing=list(missing),
        )
        self.missing = missing
