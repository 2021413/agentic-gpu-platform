"""Prometheus metrics.

Telemetry is never mandatory: every recorder here is safe to replace with the
null one, and no unit test needs a registry. That is a deliberate constraint of
the spec — a platform whose tests depend on its metrics has metrics in the wrong
place.

The metric names are the ones the spec enumerates, so dashboards written against
it keep working.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

__all__ = [
    "METRICS_CONTENT_TYPE",
    "NullMetricsRecorder",
    "PlatformMetrics",
    "PrometheusMetricsRecorder",
    "render_metrics",
]

METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Latencies that matter here span three orders of magnitude: a tool that fails
# in milliseconds and a 30B model answering in minutes share one histogram.
_LATENCY_BUCKETS = (0.05, 0.25, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)


@dataclass(frozen=True, slots=True)
class PlatformMetrics:
    """The instruments named by the spec, bound to one registry."""

    registry: CollectorRegistry

    active_runs: Gauge
    queued_jobs: Gauge
    active_jobs: Gauge
    registered_workers: Gauge
    healthy_workers: Gauge

    llm_requests_total: Counter
    llm_request_latency_seconds: Histogram
    llm_tokens_input_total: Counter
    llm_tokens_output_total: Counter

    tool_executions_total: Counter
    tool_execution_latency_seconds: Histogram

    candidate_pass_rate: Gauge
    review_failures_total: Counter
    worker_failures_total: Counter
    job_retries_total: Counter

    @classmethod
    def create(cls, registry: CollectorRegistry | None = None) -> PlatformMetrics:
        reg = registry if registry is not None else CollectorRegistry()
        return cls(
            registry=reg,
            active_runs=Gauge("active_runs", "Runs not in a terminal state", registry=reg),
            queued_jobs=Gauge(
                "queued_jobs", "Jobs waiting to be claimed", ["job_type"], registry=reg
            ),
            active_jobs=Gauge("active_jobs", "Jobs currently leased", registry=reg),
            registered_workers=Gauge(
                "registered_workers", "Workers known to the registry", registry=reg
            ),
            healthy_workers=Gauge(
                "healthy_workers", "Workers able to accept new jobs", registry=reg
            ),
            llm_requests_total=Counter(
                "llm_requests_total",
                "Inference requests issued",
                ["role", "model", "outcome"],
                registry=reg,
            ),
            llm_request_latency_seconds=Histogram(
                "llm_request_latency_seconds",
                "Inference latency",
                ["role", "model"],
                buckets=_LATENCY_BUCKETS,
                registry=reg,
            ),
            llm_tokens_input_total=Counter(
                "llm_tokens_input_total", "Prompt tokens", ["role", "model"], registry=reg
            ),
            llm_tokens_output_total=Counter(
                "llm_tokens_output_total",
                "Generated tokens",
                ["role", "model"],
                registry=reg,
            ),
            tool_executions_total=Counter(
                "tool_executions_total",
                "Deterministic tool executions",
                ["tool", "outcome"],
                registry=reg,
            ),
            tool_execution_latency_seconds=Histogram(
                "tool_execution_latency_seconds",
                "Tool execution latency",
                ["tool"],
                buckets=_LATENCY_BUCKETS,
                registry=reg,
            ),
            candidate_pass_rate=Gauge(
                "candidate_pass_rate",
                "Share of candidates surviving deterministic validation",
                registry=reg,
            ),
            review_failures_total=Counter(
                "review_failures_total", "Reviews returning FAIL", registry=reg
            ),
            worker_failures_total=Counter(
                "worker_failures_total",
                "Workers declared unavailable",
                ["reason"],
                registry=reg,
            ),
            job_retries_total=Counter(
                "job_retries_total", "Job attempts beyond the first", ["job_type"], registry=reg
            ),
        )


class PrometheusMetricsRecorder:
    """Generic recorder used where a caller knows only a metric name.

    Unknown names are ignored rather than raised: a missing dashboard line is a
    far better outcome than a run that dies because someone renamed a counter.
    """

    __slots__ = ("_metrics",)

    def __init__(self, metrics: PlatformMetrics) -> None:
        self._metrics = metrics

    @property
    def metrics(self) -> PlatformMetrics:
        return self._metrics

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        instrument = getattr(self._metrics, name, None)
        if isinstance(instrument, Counter):
            self._bind(instrument, labels).inc(value)

    def observe(self, name: str, value: float, **labels: str) -> None:
        instrument = getattr(self._metrics, name, None)
        if isinstance(instrument, Histogram):
            self._bind(instrument, labels).observe(value)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        instrument = getattr(self._metrics, name, None)
        if isinstance(instrument, Gauge):
            self._bind(instrument, labels).set(value)

    @staticmethod
    def _bind(instrument: Counter | Gauge | Histogram, labels: dict[str, str]):  # type: ignore[no-untyped-def]
        return instrument.labels(**labels) if labels else instrument


class NullMetricsRecorder:
    """Records nothing. The default, so telemetry is never load-bearing."""

    __slots__ = ()

    def increment(self, name: str, value: int = 1, **labels: str) -> None: ...
    def observe(self, name: str, value: float, **labels: str) -> None: ...
    def gauge(self, name: str, value: float, **labels: str) -> None: ...


def render_metrics(metrics: PlatformMetrics) -> bytes:
    """The exposition payload served at ``/metrics``."""
    return generate_latest(metrics.registry)
