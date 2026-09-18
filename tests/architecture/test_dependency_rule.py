"""The dependency rule, enforced mechanically.

Clean Architecture survives only if something checks it. These tests parse the
imports of every module and fail the build when an arrow points the wrong way,
which is cheaper than discovering it during a refactor two months from now.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# interfaces -> application -> domain ; infrastructure -> application/domain ports
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "domain": ("application", "infrastructure", "interfaces", "bootstrap", "worker_agent"),
    "application": ("infrastructure", "interfaces", "bootstrap", "worker_agent"),
    "infrastructure": ("interfaces", "bootstrap"),
}

# Frameworks and drivers that must never reach the inner layers.
FORBIDDEN_THIRD_PARTY: dict[str, tuple[str, ...]] = {
    "domain": (
        "fastapi",
        "starlette",
        "sqlalchemy",
        "alembic",
        "redis",
        "httpx",
        "pydantic",
        "uvicorn",
        "prometheus_client",
    ),
    "application": ("fastapi", "starlette", "sqlalchemy", "alembic", "redis", "uvicorn"),
}


def modules_of(layer: str) -> Iterator[Path]:
    root = SRC / layer
    if not root.exists():
        return
    yield from sorted(root.rglob("*.py"))


def imported_roots(path: Path) -> set[str]:
    """Top-level package of every import in a module, including inside TYPE_CHECKING."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_inner_layers_do_not_import_outer_layers(layer: str) -> None:
    violations: list[str] = []
    for module in modules_of(layer):
        offending = imported_roots(module) & set(FORBIDDEN[layer])
        violations.extend(f"{module.relative_to(SRC)} imports {name}" for name in sorted(offending))
    assert not violations, "dependency rule violated:\n" + "\n".join(violations)


@pytest.mark.parametrize("layer", sorted(FORBIDDEN_THIRD_PARTY))
def test_frameworks_do_not_leak_inwards(layer: str) -> None:
    violations: list[str] = []
    for module in modules_of(layer):
        offending = imported_roots(module) & set(FORBIDDEN_THIRD_PARTY[layer])
        violations.extend(f"{module.relative_to(SRC)} imports {name}" for name in sorted(offending))
    assert not violations, "framework leaked into an inner layer:\n" + "\n".join(violations)


def ambient_calls(path: Path) -> set[str]:
    """Calls that reach for ambient time or identity, found in the AST.

    Parsed rather than grepped: a module that merely *documents* the rule, like
    the clock port, must not be reported as breaking it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("now", "utcnow"):
            target = func.value
            if isinstance(target, ast.Name) and target.id == "datetime":
                found.add(f"datetime.{func.attr}()")
        elif isinstance(func, ast.Name) and func.id in ("uuid4", "uuid1"):
            found.add(f"{func.id}()")
    return found


def test_the_domain_never_reads_the_clock_directly() -> None:
    """Time is injected through the Clock port, so runs stay reproducible."""
    violations = [
        f"{module.relative_to(SRC)} calls {call}"
        for module in modules_of("domain")
        for call in sorted(ambient_calls(module))
        if call.startswith("datetime.")
    ]
    assert not violations, "\n".join(violations)


# Identifier factories are the one legitimate place to mint UUIDs.
ID_FACTORIES = {"identifiers.py", "lease.py", "base.py"}


def test_only_identifier_factories_generate_uuids() -> None:
    violations = [
        f"{module.relative_to(SRC)} calls {call}"
        for module in modules_of("domain")
        if module.name not in ID_FACTORIES
        for call in sorted(ambient_calls(module))
        if call.startswith("uuid")
    ]
    assert not violations, "\n".join(violations)


def test_the_domain_is_not_empty() -> None:
    """Guards against the checks above silently passing on an empty tree."""
    assert len(list(modules_of("domain"))) > 20
