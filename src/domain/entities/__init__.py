"""Domain entities and aggregates."""

from __future__ import annotations

from domain.entities.base import Entity
from domain.entities.candidate import Candidate
from domain.entities.job import Job
from domain.entities.plan import Plan, PlanTask
from domain.entities.project import Project, ToolchainConfig
from domain.entities.review import Review, ReviewFinding, Severity
from domain.entities.run import Run
from domain.entities.worker import Worker

__all__ = [
    "Candidate",
    "Entity",
    "Job",
    "Plan",
    "PlanTask",
    "Project",
    "Review",
    "ReviewFinding",
    "Run",
    "Severity",
    "ToolchainConfig",
    "Worker",
]
