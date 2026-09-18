"""The contract the composition root has to satisfy.

These tests exist so the ``bootstrap`` module, written separately, can be
verified against something executable rather than against prose.
"""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import pytest
from tests.api.conftest import Harness

from application.dto.views import ProjectView
from application.use_cases.projects import ListProjectsUseCase
from interfaces.api.app import create_api
from interfaces.api.dependencies.container import ApiDependencies
from interfaces.api.dependencies.providers import get_dependencies, get_list_projects


async def test_an_unwired_application_fails_loudly() -> None:
    """A wiring mistake must be one clear message, not an AttributeError deeper."""
    app = create_api()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)

    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        with pytest.raises(RuntimeError, match=r"app\.state\.dependencies"):
            await client.get("/v1/projects")


async def test_the_container_can_be_attached_after_construction(harness: Harness) -> None:
    """``create_api()`` then ``app.state.dependencies = ...`` is a supported order."""
    app = create_api()
    app.state.dependencies = harness.app.state.dependencies
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        response = await client.get("/v1/projects")

    assert response.status_code == 200


async def test_a_single_provider_can_be_overridden(harness: Harness) -> None:
    """Fine-grained overrides are the seam a test uses to stub one use case."""

    class EmptyListing(ListProjectsUseCase):
        def __init__(self) -> None: ...

        async def execute(self, *, limit: int = 100, offset: int = 0) -> list[ProjectView]:
            return []

    harness.app.dependency_overrides[get_list_projects] = EmptyListing
    transport = httpx.ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        await client.post("/v1/projects", json={"name": "ghost", "local_path": "/tmp/ghost"})
        response = await client.get("/v1/projects")

    assert response.json() == []
    harness.app.dependency_overrides.clear()


async def test_the_whole_container_can_be_overridden(harness: Harness) -> None:
    container: ApiDependencies = harness.app.state.dependencies
    app = create_api()
    app.dependency_overrides[get_dependencies] = lambda: container
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        response = await client.get("/v1/workers")

    assert response.status_code == 200


def test_the_http_layer_never_imports_an_adapter() -> None:
    """The dependency rule, checked where ``tests/architecture`` does not reach.

    ``tests/architecture/test_dependency_rule.py`` constrains ``domain``,
    ``application`` and ``infrastructure``, but declares nothing for
    ``interfaces``. This layer must still only know ``application`` and
    ``domain``: adapters arrive injected.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "interfaces"
    forbidden = {"infrastructure", "bootstrap", "worker_agent"}
    violations: list[str] = []

    for module in sorted(root.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        violations.extend(f"{module.name} imports {name}" for name in sorted(roots & forbidden))

    assert not violations, "\n".join(violations)
