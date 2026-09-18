"""Command line entry points.

The shell layer is deliberately thin: it sequences these commands and finally
``exec``s vLLM. Anything that needs a decision lives in Python, where it can be
tested without a GPU.

Exit codes are a contract with ``entrypoint.sh``:

    0  success
    2  configuration error        — operator mistake, retrying will not help
    3  persistent storage error   — volume missing, read-only or full
    4  model preparation failed
    5  not ready within deadline
    6  smoke test failed
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from importlib import metadata

from worker.config import ConfigError, WorkerConfig
from worker.filesystem import StorageError, inspect_storage
from worker.gpu import GpuSurvey, survey_gpus
from worker.model_state import (
    ModelPreparationError,
    prepare_model,
    read_marker,
)
from worker.readiness import health, smoke_test, wait_until_ready

__all__ = [
    "health_main",
    "preflight_main",
    "prepare_main",
    "ready_main",
    "serve_args_main",
    "smoke_main",
]

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_STORAGE = 3
EXIT_MODEL = 4
EXIT_NOT_READY = 5
EXIT_SMOKE = 6

_ALLOW_EPHEMERAL = "WORKER_ALLOW_EPHEMERAL_STORAGE"
_TRACKED_PACKAGES = ("vllm", "torch", "transformers", "huggingface-hub", "httpx")
_MIN_ARGV = 2


def _err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _out(message: str = "") -> None:
    print(message, flush=True)


def _rule(title: str) -> None:
    _out(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def _load_config() -> WorkerConfig:
    try:
        return WorkerConfig.from_env()
    except ConfigError as exc:
        _err(exc.render())
        raise SystemExit(EXIT_CONFIG) from exc


def _require_mount() -> bool:
    """Whether an unmounted volume is fatal.

    The escape hatch exists for throwaway experiments, and says so loudly: a
    worker running on ephemeral storage re-downloads 31 GB on every restart.
    """
    return os.environ.get(_ALLOW_EPHEMERAL, "").strip().lower() not in ("1", "true", "yes")


def _versions() -> list[str]:
    lines = []
    for name in _TRACKED_PACKAGES:
        try:
            lines.append(f"{name:18} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name:18} not installed")
    image = os.environ.get("VLLM_IMAGE_TAG")
    if image:
        lines.append(f"{'base image':18} {image}")
    return lines


def _gpu_warnings(survey: GpuSurvey, config: WorkerConfig) -> list[str]:
    """Conditions that will not stop the boot but will disappoint someone."""
    warnings: list[str] = []
    if not survey.gpus:
        warnings.append("no GPU is visible: vLLM will fail to start")
        return warnings
    if not survey.all_support_fp8:
        warnings.append(
            "at least one GPU lacks hardware FP8 (compute < 8.9); this model is "
            "FP8-quantised and will be dequantised at a large speed cost"
        )
    if not survey.homogeneous:
        warnings.append("GPUs are not identical, which tensor parallelism handles badly")
    if config.tensor_parallel_size > survey.count:
        warnings.append(
            f"TENSOR_PARALLEL_SIZE={config.tensor_parallel_size} exceeds the "
            f"{survey.count} visible GPU(s)"
        )
    if survey.count % max(config.tensor_parallel_size, 1) != 0:
        warnings.append(
            f"{survey.count} GPU(s) do not divide evenly by "
            f"TENSOR_PARALLEL_SIZE={config.tensor_parallel_size}"
        )
    return warnings


# ----------------------------------------------------------------------
def _report_model(config: WorkerConfig) -> object | None:
    _rule("model")
    marker = read_marker(config.layout)
    if marker is None:
        _out("  not prepared on this volume yet")
    else:
        _out(f"  {marker.model_id}@{marker.revision[:12]}")
        _out(f"  {marker.total_gb:,.1f} GB, prepared {marker.prepared_at}")
    return marker


def preflight_main(argv: Sequence[str] | None = None) -> int:
    """Report the environment and refuse to continue if it cannot work."""
    parser = argparse.ArgumentParser(description="Inspect the worker environment")
    parser.add_argument(
        "--skip-storage", action="store_true", help="report only hardware and versions"
    )
    args = parser.parse_args(argv)

    config = _load_config()

    _rule("configuration")
    for line in config.describe():
        _out(f"  {line}")

    _rule("software")
    for line in _versions():
        _out(f"  {line}")

    _rule("hardware")
    survey = survey_gpus()
    for line in survey.render():
        _out(f"  {line}")
    if survey.gpus:
        _out(f"  {'total GPU memory':18} {survey.total_memory_gb:,.0f} GB")

    if args.skip_storage:
        return EXIT_OK

    _rule("persistent storage")
    try:
        report = inspect_storage(config, require_mount=_require_mount())
    except StorageError as exc:
        _err("")
        _err(exc.render())
        return EXIT_STORAGE
    for line in report.render():
        _out(f"  {line}")

    marker = _report_model(config)

    for warning in _gpu_warnings(survey, config):
        _out(f"  ! {warning}")

    # Free space only blocks a cold start. A volume that already holds the model
    # is allowed to be nearly full: nothing large is about to be written.
    if marker is None and not report.has_enough_free:
        _err(
            f"\n  x only {report.disk.free_gb:,.1f} GB free, "
            f"{config.min_free_disk_gb:g} GB required to download the model"
        )
        return EXIT_STORAGE

    _out("\npreflight passed")
    return EXIT_OK


def prepare_main(argv: Sequence[str] | None = None) -> int:
    """Make the model available on the volume, resuming if needed."""
    parser = argparse.ArgumentParser(description="Download and verify the model")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report whether the model is ready without downloading",
    )
    args = parser.parse_args(argv)

    config = _load_config()
    try:
        layout = config.layout
        if args.check_only:
            marker = read_marker(layout)
            if marker is None or not marker.matches(config.model_id, config.model_revision):
                _out("model is not prepared")
                return EXIT_MODEL
            _out(f"model is prepared: {marker.snapshot_path}")
            return EXIT_OK

        from worker.filesystem import ensure_layout  # noqa: PLC0415 - after config parse

        ensure_layout(config, require_mount=_require_mount())
        outcome = prepare_model(config, log=_out)
    except StorageError as exc:
        _err(exc.render())
        return EXIT_STORAGE
    except ModelPreparationError as exc:
        _err(exc.render())
        return EXIT_MODEL

    verb = "downloaded" if outcome.downloaded else "reused"
    _out(
        f"model {verb}: {outcome.state.total_gb:,.1f} GB in "
        f"{outcome.duration_seconds:.0f}s at {outcome.state.snapshot_path}"
    )
    return EXIT_OK


def serve_args_main(argv: Sequence[str] | None = None) -> int:
    """Print the exact vLLM argument vector, one word per line.

    The entrypoint reads these into an array and ``exec``s them. Passing a
    string through a shell instead would let any environment variable inject a
    command, which is why nothing here is ever concatenated.
    """
    parser = argparse.ArgumentParser(description="Print the vLLM argument vector")
    parser.add_argument("--redacted", action="store_true", help="replace the API key, for logging")
    args = parser.parse_args(argv)

    config = _load_config()
    marker = read_marker(config.layout)
    snapshot = None
    if marker is not None and marker.matches(config.model_id, config.model_revision):
        # Point vLLM at the verified local snapshot so it does not resolve the
        # repository again, which would need the network on every cold start.
        snapshot = marker.snapshot_path

    words = config.redacted_argv(snapshot) if args.redacted else config.vllm_argv(snapshot)
    for word in words:
        _out(word)
    return EXIT_OK


def ready_main(argv: Sequence[str] | None = None) -> int:
    """Block until the server serves the expected model."""
    parser = argparse.ArgumentParser(description="Wait for vLLM to become ready")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = _load_config()
    result = wait_until_ready(
        config,
        timeout_seconds=args.timeout,
        log=(lambda _m: None) if args.quiet else _out,
    )
    _out(result.render())
    return EXIT_OK if result.ready else EXIT_NOT_READY


def health_main(argv: Sequence[str] | None = None) -> int:
    """Liveness only. Used by the container HEALTHCHECK."""
    parser = argparse.ArgumentParser(description="Probe liveness")
    parser.parse_args(argv)
    config = _load_config()
    summary = health(config)
    _out(summary.detail)
    return EXIT_OK if summary.alive else EXIT_NOT_READY


def smoke_main(argv: Sequence[str] | None = None) -> int:
    """Run one real completion and validate it."""
    parser = argparse.ArgumentParser(description="Run one real completion")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    config = _load_config()
    result = smoke_test(
        config,
        timeout=args.timeout,
        **({"max_tokens": args.max_tokens} if args.max_tokens else {}),
    )
    for line in result.render():
        (_out if result.passed else _err)(line)
    return EXIT_OK if result.passed else EXIT_SMOKE


def _entry(function: object) -> None:  # pragma: no cover - thin shim
    raise SystemExit(function())  # type: ignore[operator]


def main() -> int:  # pragma: no cover - convenience for `python -m worker.cli`
    commands = {
        "preflight": preflight_main,
        "prepare-model": prepare_main,
        "serve-args": serve_args_main,
        "ready": ready_main,
        "health": health_main,
        "smoke-test": smoke_main,
    }
    if len(sys.argv) < _MIN_ARGV or sys.argv[1] not in commands:
        _err(f"usage: python -m worker.cli {{{'|'.join(commands)}}} [options]")
        return EXIT_CONFIG
    return commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
