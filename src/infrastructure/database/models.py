"""SQLAlchemy 2.x table definitions (spec section 13).

These classes are *rows*, not aggregates: they hold no invariant, no
transition and no default that the domain does not already own. Translation in
both directions lives in :mod:`infrastructure.database.mappers`, so a schema
change never silently alters behaviour.

Two conventions worth knowing before editing:

* Enumerations are stored as ``VARCHAR`` holding the ``StrEnum`` value rather
  than as PostgreSQL ``ENUM`` types. Adding a member to a domain enum is then a
  code change, not a migration with an exclusive lock on the table.
* Many-to-one relationships (``JobModel.run``, ``CandidateModel.run``...) are
  declared with ``lazy="raise"`` and never traversed. They exist because
  SQLAlchemy orders a flush by *mapper* dependencies, not by foreign keys: a
  candidate and its run added in one transaction would otherwise be inserted in
  an arbitrary order and hit the foreign key.
* JSONB columns hold value objects that are read and written as a whole
  (``RunLimits``, ``WorkerCapabilities``, ``Patch``, ``ValidationReport``...).
  They are never mutated in place, so no ``MutableDict`` tracking is needed —
  the mappers always assign a freshly built dictionary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "CandidateModel",
    "JobModel",
    "PlanModel",
    "PlanTaskModel",
    "ProjectModel",
    "ReviewFindingModel",
    "ReviewModel",
    "RunEventModel",
    "RunModel",
    "ToolResultModel",
    "WorkerModel",
]

# Short enough for every ``StrEnum`` value in ``domain.enums`` with room to grow.
_ENUM_LEN = 32
_ID_LEN = 255


class Base(DeclarativeBase):
    """Declarative base carrying the naming convention used by Alembic.

    Explicit constraint names matter: without them PostgreSQL invents names and
    a later ``ALTER TABLE ... DROP CONSTRAINT`` in a migration cannot be written
    deterministically.
    """

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )

    # How a Python annotation becomes a column type. Declared once so no model
    # has to repeat ``JSONB`` or the timezone-aware timestamp.
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: postgresql.JSONB,
        list[Any]: postgresql.JSONB,
        datetime: DateTime(timezone=True),
        UUID: postgresql.UUID(as_uuid=True),
    }


class ProjectModel(Base):
    """A source repository the platform works on."""

    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(_ID_LEN), unique=True)
    repository_url: Mapped[str | None] = mapped_column(Text, default=None)
    local_path: Mapped[str | None] = mapped_column(Text, default=None)
    default_branch: Mapped[str] = mapped_column(String(_ID_LEN), default="main")
    created_at: Mapped[datetime]
    toolchain: Mapped[dict[str, Any]] = mapped_column(default=dict)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)


class RunModel(Base):
    """One agentic run. Every counter the domain bounds is a column here.

    ``plan_id`` and ``selected_candidate_id`` are deliberately *not* foreign
    keys: plans and candidates point back at the run, and a mutual foreign key
    would force an insertion order that no unit of work can honour in a single
    flush.
    """

    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_status", "status"),
        Index("ix_runs_project_id_created_at", "project_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"))
    objective: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(_ENUM_LEN))
    candidate_count: Mapped[int] = mapped_column(Integer)
    limits: Mapped[dict[str, Any]] = mapped_column(default=dict)
    plan_id: Mapped[UUID | None] = mapped_column(default=None)
    plan_revisions: Mapped[int] = mapped_column(Integer, default=0)
    repair_iterations: Mapped[int] = mapped_column(Integer, default=0)
    review_iterations: Mapped[int] = mapped_column(Integer, default=0)
    selected_candidate_id: Mapped[UUID | None] = mapped_column(default=None)
    input_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    output_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    # Unique so that a replayed "create run" request can never produce a second
    # run (spec section 35). NULL keys stay distinct, which is what we want:
    # internally created runs have no key.
    idempotency_key: Mapped[str | None] = mapped_column(String(_ID_LEN), unique=True, default=None)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)
    created_at: Mapped[datetime]
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    updated_at: Mapped[datetime]
    failure_kind: Mapped[str | None] = mapped_column(String(_ENUM_LEN), default=None)
    failure_reason: Mapped[str | None] = mapped_column(Text, default=None)
    # Optimistic concurrency: two orchestrator replicas reporting on the same
    # run must not silently overwrite each other's counters.
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    # RUF012 wants ``ClassVar`` here, but SQLAlchemy declares ``__mapper_args__``
    # as an instance attribute and mypy then rejects the override; the dict is
    # read once at mapper configuration time and never mutated.
    __mapper_args__ = {"version_id_col": version}  # noqa: RUF012

    project: Mapped[ProjectModel] = relationship(lazy="raise")


class PlanModel(Base):
    """One plan revision. Revisions are append-only: a plan is never updated."""

    __tablename__ = "plans"
    __table_args__ = (
        UniqueConstraint("run_id", "revision", name="uq_plans_run_id_revision"),
        Index("ix_plans_run_id", "run_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    revision: Mapped[int] = mapped_column(Integer)
    objective: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]
    assumptions: Mapped[list[Any]] = mapped_column(default=list)
    constraints: Mapped[list[Any]] = mapped_column(default=list)
    risk_areas: Mapped[list[Any]] = mapped_column(default=list)
    validation_requirements: Mapped[list[Any]] = mapped_column(default=list)
    prompt_version: Mapped[str] = mapped_column(String(_ENUM_LEN), default="v1")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)

    run: Mapped[RunModel] = relationship(lazy="raise")
    tasks: Mapped[list[PlanTaskModel]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="PlanTaskModel.position",
        lazy="selectin",
    )


class PlanTaskModel(Base):
    """One task of a plan.

    ``position`` preserves the planner's ordering, which the dependency graph
    alone does not determine: tasks inside one execution layer are unordered by
    the graph but must come back in a stable order.
    """

    __tablename__ = "plan_tasks"
    __table_args__ = (
        UniqueConstraint("plan_id", "key", name="uq_plan_tasks_plan_id_key"),
        Index("ix_plan_tasks_plan_id", "plan_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    plan_id: Mapped[UUID] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String(_ID_LEN))
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    target_paths: Mapped[list[Any]] = mapped_column(default=list)
    depends_on: Mapped[list[Any]] = mapped_column(default=list)
    validation: Mapped[list[Any]] = mapped_column(default=list)

    plan: Mapped[PlanModel] = relationship(back_populates="tasks")


class JobModel(Base):
    """A schedulable unit of work and its lease (spec sections 12 and 36).

    The lease is flattened into columns instead of a JSONB blob: reclaiming
    expired leases is an indexed range scan on ``lease_expires_at``, and that
    query is what keeps jobs from getting stuck forever.

    ``candidate_id`` carries no foreign key on purpose — a job may be created in
    the same flush as the candidate it serves, and the domain already guarantees
    the reference.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_status_priority", "status", "priority"),
        Index("ix_jobs_lease_expires_at", "lease_expires_at"),
        Index("ix_jobs_run_id", "run_id"),
        Index("ix_jobs_candidate_id", "candidate_id"),
        CheckConstraint("attempt >= 0", name="attempt_not_negative"),
        CheckConstraint("max_attempts >= 1", name="max_attempts_positive"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"))
    type: Mapped[str] = mapped_column(String(_ENUM_LEN))
    role: Mapped[str | None] = mapped_column(String(_ENUM_LEN), default=None)
    candidate_id: Mapped[UUID | None] = mapped_column(default=None)
    priority: Mapped[str] = mapped_column(String(_ENUM_LEN))
    status: Mapped[str] = mapped_column(String(_ENUM_LEN))
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    requirements: Mapped[dict[str, Any] | None] = mapped_column(postgresql.JSONB, default=None)
    idempotency_key: Mapped[str | None] = mapped_column(String(_ID_LEN), unique=True, default=None)
    lease_token: Mapped[str | None] = mapped_column(String(_ID_LEN), default=None)
    lease_holder_id: Mapped[UUID | None] = mapped_column(default=None)
    lease_acquired_at: Mapped[datetime | None] = mapped_column(default=None)
    lease_expires_at: Mapped[datetime | None] = mapped_column(default=None)
    assigned_worker_id: Mapped[UUID | None] = mapped_column(default=None)
    created_at: Mapped[datetime]
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    result: Mapped[dict[str, Any] | None] = mapped_column(postgresql.JSONB, default=None)
    failure_kind: Mapped[str | None] = mapped_column(String(_ENUM_LEN), default=None)
    failure_reason: Mapped[str | None] = mapped_column(Text, default=None)
    # A worker reporting on a job it no longer holds must lose the race.
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    # RUF012 wants ``ClassVar`` here, but SQLAlchemy declares ``__mapper_args__``
    # as an instance attribute and mypy then rejects the override; the dict is
    # read once at mapper configuration time and never mutated.
    __mapper_args__ = {"version_id_col": version}  # noqa: RUF012

    run: Mapped[RunModel] = relationship(lazy="raise")
    project: Mapped[ProjectModel] = relationship(lazy="raise")


class WorkerModel(Base):
    """Durable worker registration.

    Redis owns the live registry and the heartbeat TTL (spec section 14); this
    table is the durable record that survives a Redis flush, so the control
    plane can still answer "which workers did we know about?".
    """

    __tablename__ = "workers"
    __table_args__ = (
        Index("ix_workers_status", "status"),
        Index("ix_workers_last_heartbeat_at", "last_heartbeat_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    endpoint: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(_ENUM_LEN))
    capabilities: Mapped[dict[str, Any]] = mapped_column(default=dict)
    active_jobs: Mapped[int] = mapped_column(Integer, default=0)
    queued_jobs: Mapped[int] = mapped_column(Integer, default=0)
    registered_at: Mapped[datetime]
    last_heartbeat_at: Mapped[datetime]
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)


class CandidateModel(Base):
    """One competing implementation attempt (spec section 9)."""

    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint("run_id", "candidate_index", name="uq_candidates_run_id_candidate_index"),
        Index("ix_candidates_run_id", "run_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    candidate_index: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(_ENUM_LEN))
    workspace_id: Mapped[UUID | None] = mapped_column(default=None)
    # The whole ``Patch`` value object, diff included; NULL until the coder
    # produced one. Large diffs are TOASTed by PostgreSQL, which is fine.
    patch: Mapped[dict[str, Any] | None] = mapped_column(postgresql.JSONB, default=None)
    validation: Mapped[dict[str, Any]] = mapped_column(default=dict)
    summary: Mapped[str] = mapped_column(Text, default="")
    uncertainties: Mapped[list[Any]] = mapped_column(default=list)
    coder_iterations: Mapped[int] = mapped_column(Integer, default=0)
    repair_iterations: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[UUID | None] = mapped_column(default=None)
    last_job_id: Mapped[UUID | None] = mapped_column(default=None)
    review_verdict: Mapped[str | None] = mapped_column(String(_ENUM_LEN), default=None)
    created_at: Mapped[datetime]
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    failure_kind: Mapped[str | None] = mapped_column(String(_ENUM_LEN), default=None)
    failure_reason: Mapped[str | None] = mapped_column(Text, default=None)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    # RUF012 wants ``ClassVar`` here, but SQLAlchemy declares ``__mapper_args__``
    # as an instance attribute and mypy then rejects the override; the dict is
    # read once at mapper configuration time and never mutated.
    __mapper_args__ = {"version_id_col": version}  # noqa: RUF012

    run: Mapped[RunModel] = relationship(lazy="raise")


class ReviewModel(Base):
    """A reviewer verdict. Reviews are append-only evidence, never updated."""

    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("candidate_id", "iteration", name="uq_reviews_candidate_id_iteration"),
        Index("ix_reviews_run_id", "run_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    candidate_id: Mapped[UUID] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"))
    verdict: Mapped[str] = mapped_column(String(_ENUM_LEN))
    iteration: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime]
    summary: Mapped[str] = mapped_column(Text, default="")
    prompt_version: Mapped[str] = mapped_column(String(_ENUM_LEN), default="v1")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)

    run: Mapped[RunModel] = relationship(lazy="raise")
    candidate: Mapped[CandidateModel] = relationship(lazy="raise")
    findings: Mapped[list[ReviewFindingModel]] = relationship(
        back_populates="review",
        cascade="all, delete-orphan",
        order_by="ReviewFindingModel.position",
        lazy="selectin",
    )


class ReviewFindingModel(Base):
    """One defect reported by a review.

    A row per finding rather than a JSONB array: findings are queried on their
    own ("how many blockers did this run produce?") and that is a report, not a
    document read as a whole.
    """

    __tablename__ = "review_findings"
    __table_args__ = (
        UniqueConstraint("review_id", "position", name="uq_review_findings_review_id_position"),
        Index("ix_review_findings_review_id", "review_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    review_id: Mapped[UUID] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(_ENUM_LEN))
    file_path: Mapped[str | None] = mapped_column(Text, default=None)
    line: Mapped[int | None] = mapped_column(Integer, default=None)
    repair_instruction: Mapped[str | None] = mapped_column(Text, default=None)

    review: Mapped[ReviewModel] = relationship(back_populates="findings")


class ToolResultModel(Base):
    """Deterministic evidence: what actually ran, and what it returned.

    ``position`` keeps the execution order, because a build failing before the
    tests even started is a different story from the reverse.
    """

    __tablename__ = "tool_results"
    __table_args__ = (
        Index("ix_tool_results_candidate_id_position", "candidate_id", "position"),
        Index("ix_tool_results_run_id", "run_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    candidate_id: Mapped[UUID | None] = mapped_column(default=None)
    position: Mapped[int] = mapped_column(Integer)
    tool: Mapped[str] = mapped_column(String(_ID_LEN))
    kind: Mapped[str] = mapped_column(String(_ENUM_LEN))
    command: Mapped[str] = mapped_column(Text)
    exit_code: Mapped[int] = mapped_column(Integer)
    stdout: Mapped[str] = mapped_column(Text, default="")
    stderr: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    artifacts: Mapped[list[Any]] = mapped_column(default=list)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)
    # Stamped by PostgreSQL: this is when the evidence was *stored*, which is
    # the infrastructure's business, unlike the durations the tool reported.
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)

    run: Mapped[RunModel] = relationship(lazy="raise")


class RunEventModel(Base):
    """The durable event log (spec sections 15 and 21).

    ``sequence`` is monotonic *per run*, not global: an SSE client reconnecting
    with ``Last-Event-ID`` asks for "everything after N for this run", and a
    global counter would make that cursor jump unpredictably.

    Worker lifecycle events belong to no run, so ``run_id`` is nullable and two
    partial unique indexes keep the sequence unique in both buckets — a plain
    ``UNIQUE (run_id, sequence)`` would not, because PostgreSQL considers NULLs
    distinct.
    """

    __tablename__ = "run_events"
    __table_args__ = (
        Index(
            "uq_run_events_run_id_sequence",
            "run_id",
            "sequence",
            unique=True,
            postgresql_where=text("run_id IS NOT NULL"),
        ),
        Index(
            "uq_run_events_global_sequence",
            "sequence",
            unique=True,
            postgresql_where=text("run_id IS NULL"),
        ),
        Index("ix_run_events_name", "name"),
        Index("ix_run_events_occurred_at", "occurred_at"),
    )

    # The event's own identifier: appending the same event twice is a primary
    # key violation rather than a duplicate line in the audit trail.
    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), default=None
    )
    sequence: Mapped[int] = mapped_column(BigInteger)
    name: Mapped[str] = mapped_column(String(_ID_LEN))
    # ``occurred_at`` is when the fact happened (domain clock); ``recorded_at``
    # is when the row was written, stamped by PostgreSQL so that a clock skew
    # between orchestrator replicas cannot reorder the audit trail.
    occurred_at: Mapped[datetime]
    recorded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)

    run: Mapped[RunModel | None] = relationship(lazy="raise")
