"""Initial schema: projects, runs, plans, jobs, workers, candidates, reviews,
tool results and the run event log.

Generated from ``infrastructure.database.models``; the two must stay in step,
and ``tests/infrastructure/test_database_migrations.py`` fails the build when
they drift.

Revision ID: 2ce29cf68432
Revises:
Created: 2026-09-18 02:12:39.587652
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "2ce29cf68432"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("repository_url", sa.Text(), nullable=True),
        sa.Column("local_path", sa.Text(), nullable=True),
        sa.Column("default_branch", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("toolchain", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
        sa.UniqueConstraint("name", name=op.f("uq_projects_name")),
    )
    op.create_table(
        "workers",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("active_jobs", sa.Integer(), nullable=False),
        sa.Column("queued_jobs", sa.Integer(), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_workers")),
    )
    op.create_index("ix_workers_last_heartbeat_at", "workers", ["last_heartbeat_at"], unique=False)
    op.create_index("ix_workers_status", "workers", ["status"], unique=False)
    op.create_table(
        "runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("limits", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("plan_id", sa.UUID(), nullable=True),
        sa.Column("plan_revisions", sa.Integer(), nullable=False),
        sa.Column("repair_iterations", sa.Integer(), nullable=False),
        sa.Column("review_iterations", sa.Integer(), nullable=False),
        sa.Column("selected_candidate_id", sa.UUID(), nullable=True),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failure_kind", sa.String(length=32), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_runs_project_id_projects")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_runs_idempotency_key")),
    )
    op.create_index(
        "ix_runs_project_id_created_at", "runs", ["project_id", "created_at"], unique=False
    )
    op.create_index("ix_runs_status", "runs", ["status"], unique=False)
    op.create_table(
        "candidates",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("candidate_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=True),
        sa.Column("patch", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("validation", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("uncertainties", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("coder_iterations", sa.Integer(), nullable=False),
        sa.Column("repair_iterations", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.UUID(), nullable=True),
        sa.Column("last_job_id", sa.UUID(), nullable=True),
        sa.Column("review_verdict", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_kind", sa.String(length=32), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_candidates_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_candidates")),
        sa.UniqueConstraint(
            "run_id", "candidate_index", name="uq_candidates_run_id_candidate_index"
        ),
    )
    op.create_index("ix_candidates_run_id", "candidates", ["run_id"], unique=False)
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=True),
        sa.Column("candidate_id", sa.UUID(), nullable=True),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("requirements", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("lease_token", sa.String(length=255), nullable=True),
        sa.Column("lease_holder_id", sa.UUID(), nullable=True),
        sa.Column("lease_acquired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("assigned_worker_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("failure_kind", sa.String(length=32), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.CheckConstraint("attempt >= 0", name=op.f("ck_jobs_attempt_not_negative")),
        sa.CheckConstraint("max_attempts >= 1", name=op.f("ck_jobs_max_attempts_positive")),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_jobs_project_id_projects")
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_jobs_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_jobs_idempotency_key")),
    )
    op.create_index("ix_jobs_candidate_id", "jobs", ["candidate_id"], unique=False)
    op.create_index("ix_jobs_lease_expires_at", "jobs", ["lease_expires_at"], unique=False)
    op.create_index("ix_jobs_run_id", "jobs", ["run_id"], unique=False)
    op.create_index("ix_jobs_status_priority", "jobs", ["status", "priority"], unique=False)
    op.create_table(
        "plans",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("assumptions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("constraints", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("risk_areas", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "validation_requirements", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_plans_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_plans")),
        sa.UniqueConstraint("run_id", "revision", name="uq_plans_run_id_revision"),
    )
    op.create_index("ix_plans_run_id", "plans", ["run_id"], unique=False)
    op.create_table(
        "run_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_run_events_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_events")),
    )
    op.create_index("ix_run_events_name", "run_events", ["name"], unique=False)
    op.create_index("ix_run_events_occurred_at", "run_events", ["occurred_at"], unique=False)
    op.create_index(
        "uq_run_events_global_sequence",
        "run_events",
        ["sequence"],
        unique=True,
        postgresql_where=sa.text("run_id IS NULL"),
    )
    op.create_index(
        "uq_run_events_run_id_sequence",
        "run_events",
        ["run_id", "sequence"],
        unique=True,
        postgresql_where=sa.text("run_id IS NOT NULL"),
    )
    op.create_table(
        "tool_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("candidate_id", sa.UUID(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("tool", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("command", sa.Text(), nullable=False),
        sa.Column("exit_code", sa.Integer(), nullable=False),
        sa.Column("stdout", sa.Text(), nullable=False),
        sa.Column("stderr", sa.Text(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("artifacts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_tool_results_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_results")),
    )
    op.create_index(
        "ix_tool_results_candidate_id_position",
        "tool_results",
        ["candidate_id", "position"],
        unique=False,
    )
    op.create_index("ix_tool_results_run_id", "tool_results", ["run_id"], unique=False)
    op.create_table(
        "plan_tasks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("plan_id", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("target_paths", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("depends_on", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("validation", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["plan_id"], ["plans.id"], name=op.f("fk_plan_tasks_plan_id_plans"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_plan_tasks")),
        sa.UniqueConstraint("plan_id", "key", name="uq_plan_tasks_plan_id_key"),
    )
    op.create_index("ix_plan_tasks_plan_id", "plan_tasks", ["plan_id"], unique=False)
    op.create_table(
        "reviews",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("candidate_id", sa.UUID(), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["candidates.id"],
            name=op.f("fk_reviews_candidate_id_candidates"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_reviews_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reviews")),
        sa.UniqueConstraint("candidate_id", "iteration", name="uq_reviews_candidate_id_iteration"),
    )
    op.create_index("ix_reviews_run_id", "reviews", ["run_id"], unique=False)
    op.create_table(
        "review_findings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("review_id", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("line", sa.Integer(), nullable=True),
        sa.Column("repair_instruction", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_review_findings_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_review_findings")),
        sa.UniqueConstraint("review_id", "position", name="uq_review_findings_review_id_position"),
    )
    op.create_index("ix_review_findings_review_id", "review_findings", ["review_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_review_findings_review_id", table_name="review_findings")
    op.drop_table("review_findings")
    op.drop_index("ix_reviews_run_id", table_name="reviews")
    op.drop_table("reviews")
    op.drop_index("ix_plan_tasks_plan_id", table_name="plan_tasks")
    op.drop_table("plan_tasks")
    op.drop_index("ix_tool_results_run_id", table_name="tool_results")
    op.drop_index("ix_tool_results_candidate_id_position", table_name="tool_results")
    op.drop_table("tool_results")
    op.drop_index(
        "uq_run_events_run_id_sequence",
        table_name="run_events",
        postgresql_where=sa.text("run_id IS NOT NULL"),
    )
    op.drop_index(
        "uq_run_events_global_sequence",
        table_name="run_events",
        postgresql_where=sa.text("run_id IS NULL"),
    )
    op.drop_index("ix_run_events_occurred_at", table_name="run_events")
    op.drop_index("ix_run_events_name", table_name="run_events")
    op.drop_table("run_events")
    op.drop_index("ix_plans_run_id", table_name="plans")
    op.drop_table("plans")
    op.drop_index("ix_jobs_status_priority", table_name="jobs")
    op.drop_index("ix_jobs_run_id", table_name="jobs")
    op.drop_index("ix_jobs_lease_expires_at", table_name="jobs")
    op.drop_index("ix_jobs_candidate_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_candidates_run_id", table_name="candidates")
    op.drop_table("candidates")
    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_runs_project_id_created_at", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_workers_status", table_name="workers")
    op.drop_index("ix_workers_last_heartbeat_at", table_name="workers")
    op.drop_table("workers")
    op.drop_table("projects")
