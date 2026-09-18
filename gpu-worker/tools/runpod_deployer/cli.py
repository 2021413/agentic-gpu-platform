"""``python -m tools.runpod_deployer <command>`` -- deploy the worker on RunPod.

Commands
--------
``deploy``      create a Pod, wait for it, wait for the model, smoke test it
``status``      show one Pod and the URL to reach it
``smoke-test``  run one real completion against an existing Pod
``destroy``     terminate a Pod, only when explicitly asked
``gpu-types``   list the GPU types RunPod currently offers

Exit codes
----------
0   success
1   unexpected internal error (a bug here, not on RunPod)
2   usage error (argparse)
3   credential problem: RUNPOD_API_KEY missing, invalid or refused
4   RunPod refused the request: quota, no GPU, unknown volume, outage
5   the Pod never became addressable within --pod-timeout
6   the Pod is up but never served the model within --ready-timeout
7   the smoke test failed: the worker answered, but not correctly
8   the operation was refused by the operator (destroy without confirmation)

Nothing here destroys a Pod on its own. A failed deployment leaves the Pod
running, on purpose: the 31 GB of weights on its volume are the expensive part,
and an operator who wants them gone can say so in one command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TextIO

from . import client as api
from .models import (
    PROXY_TIMEOUT_SECONDS,
    AccessEndpoint,
    PodSpec,
    PodState,
    SecretValue,
    SpecError,
    WorkerSettings,
    choose_access_url,
    default_ports,
    sequence_to_tuple,
)

__all__ = ["main"]

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_CREDENTIALS = 3
EXIT_API = 4
EXIT_POD_TIMEOUT = 5
EXIT_NOT_READY = 6
EXIT_SMOKE_FAILED = 7
EXIT_ABORTED = 8

DEFAULT_IMAGE_ENV = "WORKER_IMAGE"


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    prompt: Callable[[str], str] = input,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Entry point. Every failure mode maps to one documented exit code."""
    env = os.environ if environ is None else environ
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        handler: Callable[..., int] = args.handler
        return handler(args, env=env, out=out, err=err, prompt=prompt)
    except api.MissingApiKeyError as exc:
        print(exc.render(), file=err)
        return EXIT_CREDENTIALS
    except api.AuthenticationError as exc:
        print(exc.render(), file=err)
        return EXIT_CREDENTIALS
    except (api.RunPodError, SpecError) as exc:
        render = getattr(exc, "render", None)
        print(render() if callable(render) else f"error: {exc}", file=err)
        return EXIT_API
    except KeyboardInterrupt:
        print("interrupted; no Pod was destroyed", file=err)
        return EXIT_ABORTED


# ----------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.runpod_deployer",
        description="Deploy and validate the GPU worker on RunPod.",
        epilog="The API key is read from RUNPOD_API_KEY only; there is no flag for it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    deploy = sub.add_parser("deploy", help="create a Pod and validate it end to end")
    _add_pod_arguments(deploy)
    _add_worker_arguments(deploy)
    deploy.add_argument("--pod-timeout", type=float, default=900.0)
    deploy.add_argument("--ready-timeout", type=float, default=3600.0)
    deploy.add_argument("--poll", type=float, default=10.0)
    deploy.add_argument("--smoke-timeout", type=float, default=300.0)
    deploy.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact request body (secrets masked) and stop",
    )
    deploy.add_argument("--json", action="store_true", help="also print a machine-readable summary")
    deploy.set_defaults(handler=_cmd_deploy)

    status = sub.add_parser("status", help="show one Pod and its access URL")
    status.add_argument("pod_id")
    status.add_argument("--worker-port", type=int, default=8000)
    status.add_argument("--prefer-proxy", action="store_true", help="report the proxy URL instead")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_cmd_status)

    smoke = sub.add_parser("smoke-test", help="one real completion against an existing Pod")
    smoke.add_argument("pod_id")
    smoke.add_argument("--worker-port", type=int, default=8000)
    smoke.add_argument("--model", default=None, help="defaults to MODEL_ID / --served-model-name")
    smoke.add_argument("--served-model-name", default=None)
    smoke.add_argument("--prefer-proxy", action="store_true")
    smoke.add_argument("--timeout", type=float, default=300.0)
    smoke.add_argument(
        "--first-byte",
        action="store_true",
        help="also stream one completion to time the first byte on both routes",
    )
    smoke.add_argument("--json", action="store_true")
    smoke.set_defaults(handler=_cmd_smoke)

    destroy = sub.add_parser("destroy", help="terminate a Pod (irreversible)")
    destroy.add_argument("pod_id")
    destroy.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    destroy.set_defaults(handler=_cmd_destroy)

    gpus = sub.add_parser("gpu-types", help="list RunPod GPU types")
    gpus.add_argument("--filter", default="", help="substring match on the id")
    gpus.add_argument("--min-memory", type=int, default=0, help="minimum GPU memory in GB")
    gpus.add_argument("--json", action="store_true")
    gpus.set_defaults(handler=_cmd_gpu_types)

    return parser


def _add_pod_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", default="gpu-worker")
    parser.add_argument(
        "--image",
        default=None,
        help=f"container image; defaults to ${DEFAULT_IMAGE_ENV}",
    )
    parser.add_argument(
        "--gpu-type",
        action="append",
        dest="gpu_types",
        help="repeatable, in preference order (see `gpu-types`)",
    )
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
    parser.add_argument("--data-center", action="append", dest="data_centers")
    parser.add_argument("--container-disk-gb", type=int, default=60)
    parser.add_argument(
        "--network-volume",
        default=None,
        help="network volume id; without one nothing survives a restart",
    )
    parser.add_argument(
        "--expose",
        choices=("http", "tcp", "both"),
        default="both",
        help=(
            "http = Cloudflare proxy only (100s ceiling), tcp = direct public port, "
            "both = ask for each and prefer the direct one"
        ),
    )
    parser.add_argument(
        "--prefer-proxy",
        action="store_true",
        help="use the proxy URL even when a direct TCP port exists",
    )
    parser.add_argument("--global-networking", action="store_true")


def _add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="MODEL_ID for the worker")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--worker-port", type=int, default=8000)
    parser.add_argument("--mount-path", default="/runpod-volume")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--tensor-parallel", type=int, default=None)
    parser.add_argument("--vllm-extra-arg", action="append", dest="vllm_extra_args")


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def _cmd_deploy(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str],
    out: TextIO,
    err: TextIO,
    prompt: Callable[[str], str],
) -> int:
    del prompt
    spec = _build_spec(args, env)
    if args.dry_run:
        _print_block(out, "plan", spec.describe())
        body = spec.to_create_body()
        body["env"] = spec.environment().redacted()
        print(json.dumps(body, indent=2, sort_keys=True), file=out)
        print("\ndry run: nothing was created", file=out)
        return EXIT_OK

    vllm_key = SecretValue(env.get("VLLM_API_KEY"))
    timings: dict[str, float] = {}
    with api.RunPodClient(environ=env, log=lambda message: print(f"  {message}", file=err)) as cli:
        spec = _resolve_volume(cli, spec, out)
        _print_block(out, "plan", spec.describe())

        started = time.monotonic()
        state = cli.create_pod(spec)
        timings["create_seconds"] = time.monotonic() - started
        print(f"\ncreated pod {state.id} ({timings['create_seconds']:.1f}s)", file=out)
        print(f"if anything below fails: destroy it with `destroy {state.id} --yes`", file=out)

        wait = api.wait_for_pod(
            cli,
            state.id,
            timeout_seconds=args.pod_timeout,
            poll_seconds=args.poll,
            require_network=bool(spec.tcp_ports) and not args.prefer_proxy,
            log=lambda message: print(f"  {message}", file=err),
        )
        timings["cold_start_seconds"] = wait.waited_seconds
        state = wait.state
        if not wait.reached:
            print(f"\npod {state.id} never came up: {wait.detail}", file=err)
            _print_block(out, "pod", state.describe())
            return EXIT_POD_TIMEOUT
        print(f"pod running after {wait.waited_seconds:.0f}s", file=out)
        _print_block(out, "pod", state.describe())

    endpoint = choose_access_url(
        state, worker_port=args.worker_port, prefer_direct_tcp=not args.prefer_proxy
    )
    _print_endpoint(out, endpoint, spec.worker.public_model_name)

    ready = api.wait_until_ready(
        endpoint.base_url,
        spec.worker.public_model_name,
        timeout_seconds=args.ready_timeout,
        poll_seconds=args.poll,
        api_key=vllm_key,
        log=lambda message: print(f"  {message}", file=err),
    )
    timings["model_ready_seconds"] = ready.waited_seconds
    print(f"\n{ready.render()}", file=out)
    if not ready.ready:
        print(
            "the Pod is alive but never served the model: check its logs at "
            f"https://console.runpod.io/pods (pod {state.id}, still running and billing)",
            file=err,
        )
        _maybe_json(args, out, state, endpoint, timings, ready=False, smoke=None)
        return EXIT_NOT_READY

    smoke = api.smoke_test(
        endpoint.base_url,
        spec.worker.public_model_name,
        api_key=vllm_key,
        timeout=args.smoke_timeout,
    )
    timings["smoke_seconds"] = smoke.latency_seconds
    _print_block(out, "", smoke.render())
    _print_block(
        out,
        "timings",
        [f"{name:<22} {value:.1f}s" for name, value in timings.items()],
    )
    _maybe_json(args, out, state, endpoint, timings, ready=True, smoke=smoke)
    if not smoke.passed:
        return EXIT_SMOKE_FAILED
    print(f"\nworker ready at {endpoint.base_url}", file=out)
    return EXIT_OK


def _cmd_status(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str],
    out: TextIO,
    err: TextIO,
    prompt: Callable[[str], str],
) -> int:
    del err, prompt
    with api.RunPodClient(environ=env) as cli:
        state = cli.get_pod(args.pod_id)
    _print_block(out, "pod", state.describe())
    endpoint = choose_access_url(
        state, worker_port=args.worker_port, prefer_direct_tcp=not args.prefer_proxy
    )
    _print_endpoint(out, endpoint, None)
    if args.json:
        print(
            json.dumps(
                {
                    "pod_id": state.id,
                    "status": state.desired_status,
                    "url": endpoint.base_url,
                    "url_kind": endpoint.kind,
                    "max_request_seconds": endpoint.max_request_seconds,
                },
                indent=2,
            ),
            file=out,
        )
    return EXIT_OK


def _cmd_smoke(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str],
    out: TextIO,
    err: TextIO,
    prompt: Callable[[str], str],
) -> int:
    del prompt
    with api.RunPodClient(environ=env) as cli:
        state = cli.get_pod(args.pod_id)
    model = args.model or args.served_model_name or env.get("MODEL_ID") or ""
    if not model:
        print(
            "no model name: pass --model (it must match SERVED_MODEL_NAME or MODEL_ID)",
            file=err,
        )
        return EXIT_USAGE
    endpoint = choose_access_url(
        state, worker_port=args.worker_port, prefer_direct_tcp=not args.prefer_proxy
    )
    _print_endpoint(out, endpoint, model)
    vllm_key = SecretValue(env.get("VLLM_API_KEY"))
    smoke = api.smoke_test(endpoint.base_url, model, api_key=vllm_key, timeout=args.timeout)
    _print_block(out, "", smoke.render())

    first_byte = None
    if args.first_byte:
        first_byte = api.measure_first_byte(endpoint.base_url, model, api_key=vllm_key)
        if first_byte.ok:
            print(
                f"first byte in {first_byte.seconds:.2f}s "
                f"(full stream {first_byte.total_seconds:.2f}s) over {endpoint.kind}",
                file=out,
            )
        else:
            print(f"first-byte measurement failed: {first_byte.detail}", file=err)
    if args.json:
        print(
            json.dumps(
                {
                    "pod_id": state.id,
                    "url": endpoint.base_url,
                    "url_kind": endpoint.kind,
                    "passed": smoke.passed,
                    "latency_seconds": smoke.latency_seconds,
                    "first_byte_seconds": first_byte.seconds if first_byte else None,
                    "detail": smoke.detail,
                },
                indent=2,
            ),
            file=out,
        )
    return EXIT_OK if smoke.passed else EXIT_SMOKE_FAILED


def _cmd_destroy(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str],
    out: TextIO,
    err: TextIO,
    prompt: Callable[[str], str],
) -> int:
    if not args.yes:
        answer = ""
        try:
            answer = prompt(
                f"terminate pod {args.pod_id}? its volume data and the 31 GB download "
                "are not recoverable [type the pod id to confirm]: "
            )
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip() != args.pod_id:
            print("not confirmed; nothing was terminated", file=err)
            return EXIT_ABORTED
    with api.RunPodClient(environ=env) as cli:
        cli.terminate_pod(args.pod_id)
    print(f"pod {args.pod_id} terminated", file=out)
    return EXIT_OK


def _cmd_gpu_types(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str],
    out: TextIO,
    err: TextIO,
    prompt: Callable[[str], str],
) -> int:
    del err, prompt
    with api.RunPodClient(environ=env) as cli:
        types = cli.list_gpu_types()
    wanted = args.filter.lower()
    selected = [
        gpu for gpu in types if wanted in gpu.id.lower() and gpu.memory_in_gb >= args.min_memory
    ]
    selected.sort(key=lambda gpu: (-gpu.memory_in_gb, gpu.id))
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": gpu.id,
                        "displayName": gpu.display_name,
                        "memoryInGb": gpu.memory_in_gb,
                        "secureCloud": gpu.secure_cloud,
                        "communityCloud": gpu.community_cloud,
                        "securePrice": gpu.secure_price,
                        "communityPrice": gpu.community_price,
                    }
                    for gpu in selected
                ],
                indent=2,
            ),
            file=out,
        )
        return EXIT_OK
    print(f"{'gpu type id':<34} {'memory':>7}  {'clouds':<17} secure $/hr", file=out)
    for gpu in selected:
        print(gpu.render(), file=out)
    print(f"\n{len(selected)} of {len(types)} types shown", file=out)
    return EXIT_OK


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _build_spec(args: argparse.Namespace, env: Mapping[str, str]) -> PodSpec:
    image = args.image or env.get(DEFAULT_IMAGE_ENV)
    if not image:
        raise SpecError(
            f"no image: pass --image or export {DEFAULT_IMAGE_ENV}=<registry>/<name>:<tag>"
        )
    worker = WorkerSettings(
        model_id=args.model or WorkerSettings().model_id,
        model_revision=args.model_revision,
        served_model_name=args.served_model_name,
        persistent_root=args.mount_path,
        port=args.worker_port,
        extra_args=sequence_to_tuple(args.vllm_extra_args),
        hf_token=SecretValue(env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")),
        vllm_api_key=SecretValue(env.get("VLLM_API_KEY")),
    )
    if args.max_model_len is not None:
        worker = replace(worker, max_model_len=args.max_model_len)
    if args.gpu_memory_utilization is not None:
        worker = replace(worker, gpu_memory_utilization=args.gpu_memory_utilization)
    if args.tensor_parallel is not None:
        worker = replace(worker, tensor_parallel_size=args.tensor_parallel)
    return PodSpec(
        image_name=image,
        name=args.name,
        gpu_type_ids=sequence_to_tuple(args.gpu_types),
        gpu_count=args.gpu_count,
        cloud_type=args.cloud,
        data_center_ids=sequence_to_tuple(args.data_centers),
        container_disk_in_gb=args.container_disk_gb,
        network_volume_id=args.network_volume,
        volume_mount_path=args.mount_path,
        ports=default_ports(args.worker_port, expose=args.expose),
        global_networking=args.global_networking,
        worker=worker,
    )


def _resolve_volume(cli: api.RunPodClient, spec: PodSpec, out: TextIO) -> PodSpec:
    """Check the volume exists and pin the Pod to its data center.

    A network volume lives in exactly one data center, so a Pod that wants it
    has to be created there. Doing this before ``POST /pods`` turns a confusing
    capacity error into a clear one, and costs one cheap GET.
    """
    if not spec.network_volume_id:
        print(
            "warning: no --network-volume. Nothing persists: every restart "
            "re-downloads the 31 GB of weights.",
            file=out,
        )
        return spec
    volume = cli.get_network_volume(spec.network_volume_id)
    print(
        f"volume {volume.id} ({volume.name}, {volume.size} GB) in {volume.data_center_id}",
        file=out,
    )
    if spec.data_center_ids:
        return spec
    return replace(spec, data_center_ids=(volume.data_center_id,))


def _print_block(out: TextIO, title: str, lines: Sequence[str]) -> None:
    if title:
        print(f"\n{title}", file=out)
    for line in lines:
        print(f"  {line}" if title else line, file=out)


def _print_endpoint(out: TextIO, endpoint: AccessEndpoint, model: str | None) -> None:
    print(f"\naccess url        {endpoint.base_url}  [{endpoint.kind}]", file=out)
    if model:
        print(f"model name        {model}", file=out)
    if endpoint.is_proxied:
        print(f"WARNING: {endpoint.note}", file=out)
        print(
            f"         an orchestrator sending prompts that generate for more than "
            f"{PROXY_TIMEOUT_SECONDS}s must stream, or this Pod must be recreated with "
            "--expose tcp.",
            file=out,
        )
    else:
        print(f"note: {endpoint.note}", file=out)


def _maybe_json(
    args: argparse.Namespace,
    out: TextIO,
    state: PodState,
    endpoint: AccessEndpoint,
    timings: Mapping[str, float],
    *,
    ready: bool,
    smoke: api.SmokeReport | None,
) -> None:
    if not args.json:
        return
    print(
        json.dumps(
            {
                "pod_id": state.id,
                "status": state.desired_status,
                "url": endpoint.base_url,
                "url_kind": endpoint.kind,
                "max_request_seconds": endpoint.max_request_seconds,
                "ready": ready,
                "smoke_passed": bool(smoke and smoke.passed),
                "timings": dict(timings),
            },
            indent=2,
            sort_keys=True,
        ),
        file=out,
    )


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    raise SystemExit(main())
