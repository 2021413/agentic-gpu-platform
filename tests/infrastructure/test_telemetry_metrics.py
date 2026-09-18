"""Metrics must never be load-bearing (spec section 32)."""

from __future__ import annotations

from infrastructure.telemetry import (
    NullMetricsRecorder,
    PlatformMetrics,
    PrometheusMetricsRecorder,
    render_metrics,
)


def recorder() -> PrometheusMetricsRecorder:
    return PrometheusMetricsRecorder(PlatformMetrics.create())


def test_the_spec_metric_names_are_exposed() -> None:
    exposition = render_metrics(PlatformMetrics.create()).decode()
    for name in (
        "active_runs",
        "queued_jobs",
        "registered_workers",
        "healthy_workers",
        "llm_requests_total",
        "llm_request_latency_seconds",
        "llm_tokens_input_total",
        "tool_executions_total",
        "review_failures_total",
        "job_retries_total",
    ):
        assert name in exposition, f"{name} is missing from the exposition"


def test_counters_and_gauges_record() -> None:
    rec = recorder()
    rec.increment("review_failures_total", 2)
    rec.gauge("active_runs", 7)
    rec.observe("llm_request_latency_seconds", 1.5, role="CODER", model="m")

    exposition = render_metrics(rec.metrics).decode()
    assert "review_failures_total 2.0" in exposition
    assert "active_runs 7.0" in exposition


def test_an_unknown_metric_name_is_ignored_not_fatal() -> None:
    """A renamed counter must never be able to kill a run."""
    rec = recorder()
    rec.increment("metric_that_does_not_exist")
    rec.gauge("another_missing_one", 1.0)
    rec.observe("still_missing", 1.0)


def test_the_null_recorder_satisfies_the_same_calls() -> None:
    null = NullMetricsRecorder()
    null.increment("llm_requests_total", role="CODER", model="m", outcome="ok")
    null.gauge("active_runs", 3)
    null.observe("tool_execution_latency_seconds", 0.2, tool="build")


def test_each_registry_is_isolated() -> None:
    """Two containers in one process must not fight over a global registry."""
    first = PlatformMetrics.create()
    second = PlatformMetrics.create()
    first.active_runs.set(1)
    second.active_runs.set(9)
    assert "active_runs 1.0" in render_metrics(first).decode()
    assert "active_runs 9.0" in render_metrics(second).decode()
