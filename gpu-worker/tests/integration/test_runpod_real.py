"""Provision a real RunPod GPU, serve the real model, and measure it.

This test rents hardware. An H100 hour is real money and a leaked Pod bills
until someone notices, so two things are non-negotiable:

* it is skipped unless ``RUNPOD_API_KEY`` *and* ``RUNPOD_REAL_TEST=1`` are both
  set. Nothing about a normal ``pytest`` run, in CI or on a laptop, may start a
  Pod;
* every Pod it creates is terminated in a ``finally``, including when an
  assertion fails halfway through.

What it proves, in order: a Pod can be provisioned with the network volume
attached, the image downloads and loads the model, the worker becomes ready,
``/v1/models`` lists it, a completion comes back valid, a *second* Pod on the
same volume reuses the weights instead of downloading 31 GB again, and
inference still works there. Timings are written to a JSON artifact.

Run it with::

    RUNPOD_API_KEY=... RUNPOD_REAL_TEST=1 \
    RUNPOD_TEST_IMAGE=registry/gpu-worker:1.0.0 \
    RUNPOD_TEST_VOLUME_ID=... \
    .venv/bin/python -m pytest tests/integration/test_runpod_real.py -m runpod -s
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from runpod_deployer import client as api
from runpod_deployer.models import (
    PROXY_TIMEOUT_SECONDS,
    PodSpec,
    PodState,
    SecretValue,
    WorkerSettings,
    choose_access_url,
    default_ports,
)

pytestmark = [pytest.mark.runpod, pytest.mark.integration, pytest.mark.gpu]

OPT_IN_ENV = "RUNPOD_REAL_TEST"
IMAGE_ENV = "RUNPOD_TEST_IMAGE"
VOLUME_ENV = "RUNPOD_TEST_VOLUME_ID"

DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
WORKER_PORT = 8000

# 31 GB of weights over whatever link the host has, then a load into HBM.
POD_TIMEOUT_SECONDS = 1200.0
COLD_READY_TIMEOUT_SECONDS = 3600.0
WARM_READY_TIMEOUT_SECONDS = 1200.0
POLL_SECONDS = 15.0


def _missing_requirements() -> str | None:
    if not os.environ.get("RUNPOD_API_KEY"):
        return "RUNPOD_API_KEY is not set"
    if os.environ.get(OPT_IN_ENV) != "1":
        return f"{OPT_IN_ENV}=1 is not set: this test rents a real GPU and costs money"
    if not os.environ.get(IMAGE_ENV):
        return f"{IMAGE_ENV} is not set (the worker image to deploy)"
    if not os.environ.get(VOLUME_ENV):
        return f"{VOLUME_ENV} is not set (a network volume is what the warm restart proves)"
    return None


_SKIP_REASON = _missing_requirements()
pytestmark.append(
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "requirements met")
)


def _spec(*, name: str, volume_id: str, data_center: str | None) -> PodSpec:
    worker = WorkerSettings(
        model_id=os.environ.get("MODEL_ID") or WorkerSettings().model_id,
        served_model_name=os.environ.get("SERVED_MODEL_NAME"),
        port=WORKER_PORT,
        hf_token=SecretValue(os.environ.get("HF_TOKEN")),
        vllm_api_key=SecretValue(os.environ.get("VLLM_API_KEY")),
    )
    return PodSpec(
        image_name=os.environ[IMAGE_ENV],
        name=name,
        gpu_type_ids=(os.environ.get("RUNPOD_TEST_GPU_TYPE") or DEFAULT_GPU_TYPE,),
        cloud_type="SECURE",
        data_center_ids=(data_center,) if data_center else (),
        container_disk_in_gb=int(os.environ.get("RUNPOD_TEST_CONTAINER_DISK_GB") or 80),
        network_volume_id=volume_id,
        volume_mount_path=worker.persistent_root,
        ports=default_ports(WORKER_PORT, expose="both"),
        worker=worker,
    )


@contextmanager
def _pod(client: api.RunPodClient, spec: PodSpec, metrics: dict[str, Any]) -> Iterator[PodState]:
    """Create a Pod and guarantee its termination.

    The ``finally`` is the whole point: an assertion that fails between here and
    the end of the block must not leave an H100 running.
    """
    state = client.create_pod(spec)
    metrics.setdefault("pod_ids", []).append(state.id)
    print(f"\n[runpod] created pod {state.id}")
    try:
        yield state
    finally:
        try:
            client.terminate_pod(state.id)
            print(f"[runpod] terminated pod {state.id}")
        except api.RunPodError as exc:  # pragma: no cover - only on a RunPod outage
            print(f"[runpod] FAILED TO TERMINATE {state.id}: {exc}. Terminate it by hand NOW.")
            raise


def _bring_up(
    client: api.RunPodClient,
    spec: PodSpec,
    state: PodState,
    *,
    ready_timeout: float,
) -> tuple[PodState, float, float]:
    """Wait for placement, then for the model. Returns the two durations."""
    wait = api.wait_for_pod(
        client,
        state.id,
        timeout_seconds=POD_TIMEOUT_SECONDS,
        poll_seconds=POLL_SECONDS,
        require_network=True,
        log=lambda message: print(f"[runpod] {message}"),
    )
    assert wait.reached, f"pod {state.id} never became addressable: {wait.detail}"
    running = wait.state

    endpoint = choose_access_url(running, worker_port=WORKER_PORT)
    print(f"[runpod] {endpoint.kind} endpoint {endpoint.base_url}")
    ready = api.wait_until_ready(
        endpoint.base_url,
        spec.worker.public_model_name,
        timeout_seconds=ready_timeout,
        poll_seconds=POLL_SECONDS,
        api_key=spec.worker.vllm_api_key,
        log=lambda message: print(f"[runpod] {message}"),
    )
    assert ready.ready, f"the worker never served the model: {ready.detail}"
    return running, wait.waited_seconds, ready.waited_seconds


def _assert_inference(state: PodState, spec: PodSpec, *, label: str) -> float:
    """One real completion over the direct route. Failure here fails the test."""
    endpoint = choose_access_url(state, worker_port=WORKER_PORT)
    models = api.list_models(endpoint.base_url, api_key=spec.worker.vllm_api_key)
    assert spec.worker.public_model_name in models, f"{label}: /v1/models lists {models}"

    smoke = api.smoke_test(
        endpoint.base_url,
        spec.worker.public_model_name,
        api_key=spec.worker.vllm_api_key,
        timeout=300.0,
    )
    assert smoke.passed, f"{label}: inference failed: {smoke.detail}"
    assert smoke.output.strip(), f"{label}: the model returned nothing"
    print(f"[runpod] {label}: {smoke.latency_seconds:.1f}s, {smoke.completion_tokens} tokens")
    return smoke.latency_seconds


def _write_artifact(metrics: dict[str, Any]) -> Path:
    path = Path(os.environ.get("RUNPOD_TEST_ARTIFACT") or "artifacts/runpod_real_test.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[runpod] metrics written to {path}")
    return path


def test_real_pod_serves_the_model_and_reuses_the_volume() -> None:
    metrics: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image": os.environ[IMAGE_ENV],
        "gpu_type": os.environ.get("RUNPOD_TEST_GPU_TYPE") or DEFAULT_GPU_TYPE,
        "outcome": "incomplete",
    }
    volume_id = os.environ[VOLUME_ENV]

    try:
        with api.RunPodClient(
            environ=os.environ, log=lambda message: print(f"[runpod] {message}")
        ) as client:
            volume = client.get_network_volume(volume_id)
            metrics["network_volume"] = {
                "id": volume.id,
                "size_gb": volume.size,
                "data_center": volume.data_center_id,
            }

            # -- cold: nothing on the volume is assumed ---------------------
            cold_spec = _spec(
                name="gpu-worker-itest-cold",
                volume_id=volume_id,
                data_center=volume.data_center_id or None,
            )
            with _pod(client, cold_spec, metrics) as created:
                cold_started = time.monotonic()
                running, placement, cold_ready = _bring_up(
                    client, cold_spec, created, ready_timeout=COLD_READY_TIMEOUT_SECONDS
                )
                metrics["cold_start_seconds"] = placement
                metrics["cold_ready_seconds"] = cold_ready
                metrics["cold_total_seconds"] = time.monotonic() - cold_started
                metrics["cold_inference_seconds"] = _assert_inference(
                    running, cold_spec, label="cold"
                )
                metrics.update(_measure_proxy_first_byte(running, cold_spec))

            # -- warm: a new Pod, the same volume, no download expected -----
            warm_spec = _spec(
                name="gpu-worker-itest-warm",
                volume_id=volume_id,
                data_center=volume.data_center_id or None,
            )
            with _pod(client, warm_spec, metrics) as created:
                warm_started = time.monotonic()
                running, placement, warm_ready = _bring_up(
                    client, warm_spec, created, ready_timeout=WARM_READY_TIMEOUT_SECONDS
                )
                metrics["warm_start_seconds"] = placement
                metrics["warm_restart_seconds"] = time.monotonic() - warm_started
                # The warm run does not download: what it spends before serving
                # is the load. The difference with the cold run is what the
                # download cost, which is the number that justifies the volume.
                metrics["model_load_seconds"] = warm_ready
                metrics["model_download_seconds"] = max(cold_ready - warm_ready, 0.0)
                metrics["warm_inference_seconds"] = _assert_inference(
                    running, warm_spec, label="warm"
                )

                assert metrics["warm_restart_seconds"] < metrics["cold_total_seconds"], (
                    "the warm restart was not faster than the cold one: the volume is "
                    "not being reused and the weights were downloaded again"
                )
        metrics["outcome"] = "passed"
    finally:
        _write_artifact(metrics)


def _measure_proxy_first_byte(state: PodState, spec: PodSpec) -> dict[str, Any]:
    """Time the first streamed byte through the Cloudflare proxy.

    This is what decides whether the 100 s ceiling matters to the orchestrator:
    a first token in a couple of seconds means a streaming client survives a
    long generation; a slow first token means the proxy is only good for short
    answers and the direct TCP port is mandatory.
    """
    proxy_url = state.proxy_url(WORKER_PORT)
    report = api.measure_first_byte(
        proxy_url,
        spec.worker.public_model_name,
        api_key=spec.worker.vllm_api_key,
        timeout=float(PROXY_TIMEOUT_SECONDS + 20),
    )
    result: dict[str, Any] = {
        "proxy_url": proxy_url,
        "first_token_latency_seconds": report.seconds if report.ok else None,
        "proxy_stream_total_seconds": report.total_seconds,
        "proxy_first_byte_ok": report.ok,
        "proxy_detail": report.detail,
        "proxy_timeout_seconds": PROXY_TIMEOUT_SECONDS,
    }
    print(f"[runpod] proxy first byte: {result['first_token_latency_seconds']}s")
    assert report.ok, f"the proxy route did not stream: {report.detail}"
    return result
