"""A module that no longer imports is not a failing test.

This exists because of one run. The coder was asked for a ten-line validator in
a 357-line settings module. It wrote the four correct lines, then rewrote an
unrelated class from memory — dropping fields that existed, adding a decorator
referencing one it never declared. The module stopped importing, so all 173
tests failed at collection, and every one of the three repair rounds received
the same traceback and rewrote the same whole file again.

Nothing in that output said "you broke the file".
"""

from __future__ import annotations

from application.orchestration.orchestrator import diagnose_broken_module

# Trimmed from the real run, verbatim in shape.
PYDANTIC_AT_IMPORT = """
==================================== ERRORS ====================================
_____________ ERROR collecting tests/application/test_settings.py ______________
tests/application/test_settings.py:8: in <module>
    from bootstrap.config import Environment, Settings, WorkerSettings
src/bootstrap/config.py:216: in <module>
    class WorkerSettings(BaseSettings):
pydantic/_internal/_generate_schema.py:239: in check_decorator_fields_exist
    raise PydanticUserError(
E   pydantic.errors.PydanticUserError: check_decorator_fields_exist
"""

ORDINARY_FAILURE = """
=================================== FAILURES ===================================
_______________________ test_the_budget_is_what_it_says ________________________
    assert budget == 26624
E   assert 28672 == 26624
=========================== short test summary info ============================
FAILED tests/application/test_context_budget.py::test_the_budget_is_what_it_says
1 failed, 172 passed
"""


def test_a_module_that_stopped_importing_is_named_as_such() -> None:
    diagnosis = diagnose_broken_module(PYDANTIC_AT_IMPORT)

    assert diagnosis is not None
    assert "no longer loads" in diagnosis.lower()


def test_the_diagnosis_says_the_other_failures_are_consequences() -> None:
    """173 red tests from one broken import is the shape that misleads: it
    looks like the change was catastrophically wrong rather than unloadable."""
    diagnosis = diagnose_broken_module(PYDANTIC_AT_IMPORT)

    assert diagnosis is not None
    assert "consequence" in diagnosis.lower()


def test_the_diagnosis_names_the_cause_that_actually_happened() -> None:
    """Rewriting a whole file makes the model reproduce from memory everything
    it is not touching. Telling it to look there is the point of the message."""
    diagnosis = diagnose_broken_module(PYDANTIC_AT_IMPORT)

    assert diagnosis is not None
    assert "not asked to touch" in diagnosis


def test_an_ordinary_test_failure_gets_no_warning() -> None:
    """A warning that fires on everything is a warning nobody reads."""
    assert diagnose_broken_module(ORDINARY_FAILURE) is None


def test_silence_when_nothing_ran() -> None:
    assert diagnose_broken_module("") is None


def test_a_traceback_without_a_collection_error_is_not_a_broken_module() -> None:
    """An ImportError raised *inside* a passing-to-collect test is a test
    failure, not an unloadable file, and must not be dressed up as one."""
    inside_a_test = (
        "FAILED tests/x.py::test_y - ModuleNotFoundError: No module named 'optional_dep'\n"
        "1 failed, 40 passed"
    )

    assert diagnose_broken_module(inside_a_test) is None


def test_a_syntax_error_at_collection_counts_too() -> None:
    broken_syntax = (
        "ERROR collecting src/thing.py\n"
        "src/thing.py:12: in <module>\n"
        "E   SyntaxError: invalid syntax\n"
    )

    assert diagnose_broken_module(broken_syntax) is not None
