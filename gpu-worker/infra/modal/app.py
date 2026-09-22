"""The Modal application: one GPU Server that speaks the OpenAI API.

    modal serve  infra/modal/app.py     # development, with hot reload
    modal deploy infra/modal/app.py     # a persistent deployment

Both from the `gpu-worker/` directory.

## What this replaces

On RunPod the worker was a Pod: provisioned by hand or by `tools/runpod_deployer`,
billed from the moment it existed until someone destroyed it, reachable at a TCP
address that changed on every reset. Here it is a Modal Server: billed per second
of container life, scaled to zero when idle, and reachable at a stable HTTPS URL
that survives redeployment.

## The one thing a caller must handle

A Modal Server does not queue. When no container is running, the proxy answers
**503** immediately rather than holding the request open until a GPU is ready.
That is not an error to retry blindly at the job level — it means "the GPU is
booting, ask again shortly", and the control plane has to know the difference.
See `docs/modal.md` and the cold-start handling in the control plane's
`OpenAICompatibleSettings`.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import replace
from typing import cast

import modal

from infra.modal.config import active_config
from infra.modal.image import worker_image
from infra.modal.runtime import (
    ColdModelError,
    StartupRecord,
    append_startup_record,
    compile_cache_environment,
    gpu_slug,
    launch_vllm,
    monotonic_ms,
    new_record,
    resolve_model,
    scratch_environment,
)
from infra.modal.secrets import worker_secrets
from infra.modal.volumes import model_cache_volume, volume_mounts
from worker.config import PersistentLayout, WorkerConfig
from worker.readiness import wait_until_ready

__all__ = ["APP_NAME", "VLLMServer", "app"]

APP_NAME = os.environ.get("MODAL_APP_NAME", "agentic-gpu-worker")

CONFIG = active_config()

app = modal.App(
    APP_NAME,
    # Tags surface in `modal billing report --show-resources`, which is the only
    # way to answer "what did the worker cost this month" without guessing.
    tags={
        "service": "gpu-worker",
        "project": "agentic",
        "profile": CONFIG.profile,
    },
)


@app.server(
    image=worker_image,
    volumes=volume_mounts(),
    secrets=worker_secrets(),
    port=8000,
    **CONFIG.as_server_kwargs(),
)
class VLLMServer:
    """vLLM, started once per container, serving until the container dies.

    The model is loaded in `@modal.enter()` and never in a request path. That is
    the entire reason this is a Server class and not a Function: a container that
    reloaded 31 GB per request would be slower than no cache at all.
    """

    _process: subprocess.Popen[bytes] | None = None

    @modal.enter()
    def start(self) -> None:
        started = time.monotonic()
        config = WorkerConfig.from_env()
        layout = config.layout
        for directory in layout.all_directories():
            directory.mkdir(parents=True, exist_ok=True)

        # A container mounts the Volume as it stood when it started. If the
        # populate job finished while this container was being scheduled, its
        # commit is invisible until we ask for it — and the symptom would be a
        # refusal to start with the weights sitting right there.
        mark = time.monotonic()
        model_cache_volume.reload()
        reload_ms = monotonic_ms(mark)

        gpu = gpu_slug()
        record = new_record(config, gpu=gpu, cold_download=False)
        record = replace(record, volume_reload_ms=reload_ms)
        print(f"gpu={gpu} volume reloaded in {reload_ms}ms")

        try:
            mark = time.monotonic()
            snapshot, downloaded = resolve_model(
                config, allow_cold_download=CONFIG.allow_cold_download
            )
            record = replace(
                record,
                cold_download=downloaded,
                model_resolve_ms=monotonic_ms(mark),
            )

            compile_environment = compile_cache_environment(layout, gpu=gpu)
            for value in compile_environment.values():
                os.makedirs(value, exist_ok=True)
            print(f"compiled artifacts: {compile_environment['VLLM_CACHE_ROOT']}")

            # The scratch override has to be applied here too: `hub_environment`
            # rebuilds TMPDIR from the layout, which would put vLLM's ZeroMQ
            # sockets back on a filesystem that cannot hold them.
            child_environment = {**compile_environment, **scratch_environment()}

            mark = time.monotonic()
            self._process = launch_vllm(
                config, snapshot, extra_environment=child_environment
            )
            record = replace(record, vllm_launch_ms=monotonic_ms(mark))

            # Modal routes traffic once this method returns and the port is
            # listening. vLLM binds the port before the weights are resident, so
            # returning here would advertise a worker that answers 500s. The
            # readiness check the RunPod image already uses — /v1/models must
            # list the model we asked for — is what makes "ready" mean "serving".
            mark = time.monotonic()
            budget = max(60.0, CONFIG.startup_timeout - (time.monotonic() - started) - 30)
            result = wait_until_ready(config, timeout_seconds=budget, log=print)
            record = replace(
                record,
                readiness_ms=monotonic_ms(mark),
                total_ms=monotonic_ms(started),
                ready=result.ready,
            )
            if not result.ready:
                raise RuntimeError(f"vLLM never became ready: {result.render()}")
            print(result.render())
        except BaseException as exc:
            record = replace(
                record,
                total_ms=monotonic_ms(started),
                ready=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._write_record(layout, record)
            self._terminate()
            if isinstance(exc, ColdModelError):
                # Printed as well as raised: the traceback Modal shows is a wall
                # of frames, and this message contains the command to run.
                print(str(exc))
            raise
        self._write_record(layout, record)

    @modal.exit()
    def stop(self) -> None:
        self._terminate()

    # -- helpers ---------------------------------------------------------
    def _write_record(self, layout: PersistentLayout, record: StartupRecord) -> None:
        """Persist the boot breakdown, never failing the boot over it."""
        try:
            append_startup_record(layout, record)
            model_cache_volume.commit()
        except Exception as exc:
            print(f"could not record startup timings: {type(exc).__name__}: {exc}")

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except Exception:
            process.kill()
        self._process = None


@app.local_entrypoint()
def main() -> None:
    """Print the deployment this profile describes, then the Server's URL.

    `modal run infra/modal/app.py` is the cheapest possible check that the
    configuration is the one intended: it never starts a GPU.
    """
    print(f"app: {APP_NAME}")
    for line in CONFIG.describe():
        print(f"  {line}")
    # `@app.server()` turns the class into a `modal.Server` at import time, but
    # the decorator is not typed as doing so, so the cast states what is true.
    print(f"\nurl: {cast('modal.Server', VLLMServer).get_url()}")
    if not CONFIG.unauthenticated:
        print(
            "\nthis Server requires a proxy token:\n"
            "  modal workspace proxy-tokens create\n"
            "  export INFERENCE_API_KEY=wk-<id>.ws-<secret>"
        )
