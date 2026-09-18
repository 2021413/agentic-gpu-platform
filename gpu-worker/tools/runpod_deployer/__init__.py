"""Operator tooling to run the GPU worker on RunPod.

Three modules, kept apart so that the interesting parts are testable without a
network: :mod:`models` is pure data, :mod:`client` is the only place that opens
a socket, and :mod:`cli` is the operator's interface.

The RunPod credential is read from ``RUNPOD_API_KEY`` and from nowhere else.
"""

from __future__ import annotations

from .client import (
    ApiError,
    AuthenticationError,
    GpuUnavailableError,
    MissingApiKeyError,
    NotReadyError,
    QuotaError,
    RunPodClient,
    RunPodError,
    VolumeNotFoundError,
    redact,
)
from .models import (
    AccessEndpoint,
    ExposedPort,
    GpuType,
    NetworkVolume,
    PodEnvironment,
    PodSpec,
    PodState,
    SecretValue,
    SpecError,
    WorkerSettings,
    build_pod_environment,
    choose_access_url,
)

__all__ = [
    "AccessEndpoint",
    "ApiError",
    "AuthenticationError",
    "ExposedPort",
    "GpuType",
    "GpuUnavailableError",
    "MissingApiKeyError",
    "NetworkVolume",
    "NotReadyError",
    "PodEnvironment",
    "PodSpec",
    "PodState",
    "QuotaError",
    "RunPodClient",
    "RunPodError",
    "SecretValue",
    "SpecError",
    "VolumeNotFoundError",
    "WorkerSettings",
    "build_pod_environment",
    "choose_access_url",
    "redact",
]
