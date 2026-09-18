"""Dependency injection seams for the HTTP layer."""

from __future__ import annotations

from interfaces.api.dependencies.container import ApiDependencies
from interfaces.api.dependencies.providers import get_dependencies
from interfaces.api.dependencies.readiness import (
    AlwaysReadyProbe,
    DependencyHealth,
    ReadinessProbe,
    ReadinessReport,
)

__all__ = [
    "AlwaysReadyProbe",
    "ApiDependencies",
    "DependencyHealth",
    "ReadinessProbe",
    "ReadinessReport",
    "get_dependencies",
]
