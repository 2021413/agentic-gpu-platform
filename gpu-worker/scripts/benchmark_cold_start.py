"""Measure what a cold start actually costs, split into its parts.

    python scripts/benchmark_cold_start.py --cold 10 --warm 10

**This spends money.** Each cold iteration boots an H100 and serves one real
completion. Ten of them is roughly ten cold starts' worth of GPU time; at
$0.001097/second that is a few dollars, not a few cents.

## Why not one `cold_start_seconds`

Section 24 of the specification asks for the breakdown, and the reason is that
the number everyone quotes is almost never the number that matters. A slow first
request is usually not Modal being slow to find a GPU — it is 31 GB moving off
the Volume, or `torch.compile` and CUDA graph capture running again because the
compiled artifacts were keyed to a different device. Those have different fixes,
and a single figure tells you which one to attempt: none of them.

So this script reports two clocks side by side:

* **client-side** — time to first token and total completion time, which is what
  the control plane experiences;
* **container-side** — the `StartupRecord` each container writes to the Volume,
  which splits the same wall-clock into volume reload, model resolution, vLLM
  launch and readiness.

## How a cold start is forced

There is no "kill the container" API, and there should not be. The honest way is
the one the autoscaler already offers: shrink `scaledown_window` to its floor,
wait for the pool to empty (the Server answers 503 when it has), then restore the
deployed value and send a request. That is the same path a real idle period
takes, which is the point.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import modal

DEFAULT_APP = "agentic-gpu-worker"
DEFAULT_SERVER = "VLLMServer"
H100_DOLLARS_PER_SECOND = 0.001097
# Modal's proxy answers this when the pool is empty; it means "booting".
SERVICE_UNAVAILABLE = 503
BAD_REQUEST = 400


@dataclass
class Attempt:
    """One request, from the client's side of the wire."""

    cold: bool
    waited_for_capacity_s: float
    first_token_s: float | None
    total_s: float
    ok: bool
    detail: str = ""


@dataclass
class Summary:
    """The distribution of one measurement, as section 36 asks for it."""

    name: str
    unit: str
    count: int
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    samples: list[float] = field(default_factory=list)


def summarise(name: str, values: Sequence[float], *, unit: str = "s") -> Summary:
    ordered = sorted(values)
    if not ordered:
        return Summary(name=name, unit=unit, count=0)

    def percentile(fraction: float) -> float:
        # Nearest-rank. With ten samples, interpolating invents precision the
        # measurement does not have.
        index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
        return ordered[index]

    return Summary(
        name=name,
        unit=unit,
        count=len(ordered),
        p50=statistics.median(ordered),
        p90=percentile(0.90),
        p95=percentile(0.95),
        minimum=ordered[0],
        maximum=ordered[-1],
        samples=list(ordered),
    )


def render(summary: Summary) -> str:
    if not summary.count:
        return f"{summary.name:<26} no samples"
    return (
        f"{summary.name:<26} n={summary.count:<3} "
        f"p50={summary.p50:7.1f}{summary.unit}  p90={summary.p90:7.1f}{summary.unit}  "
        f"p95={summary.p95:7.1f}{summary.unit}  "
        f"min={summary.minimum:7.1f}{summary.unit}  max={summary.maximum:7.1f}{summary.unit}"
    )


# -- the wire -----------------------------------------------------------
def complete_once(
    client: httpx.Client,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    capacity_timeout: float,
) -> Attempt:
    """One streamed completion, waiting out 503s and timing the parts.

    Streaming is not a preference here: a non-streaming completion has no
    time-to-first-token to measure, because its first byte *is* its last.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    began = time.monotonic()
    became_available: float | None = None
    deadline = began + capacity_timeout

    while True:
        try:
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code == SERVICE_UNAVAILABLE:
                    response.read()
                    if time.monotonic() >= deadline:
                        return Attempt(
                            cold=True,
                            waited_for_capacity_s=time.monotonic() - began,
                            first_token_s=None,
                            total_s=time.monotonic() - began,
                            ok=False,
                            detail=f"no container within {capacity_timeout:g}s",
                        )
                    time.sleep(1.0)
                    continue
                if response.status_code >= BAD_REQUEST:
                    body = response.read().decode("utf-8", "replace")[:200]
                    return Attempt(
                        cold=became_available is not None,
                        waited_for_capacity_s=(became_available or began) - began,
                        first_token_s=None,
                        total_s=time.monotonic() - began,
                        ok=False,
                        detail=f"HTTP {response.status_code}: {body}",
                    )

                became_available = became_available or time.monotonic()
                first_token: float | None = None
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    if first_token is None:
                        first_token = time.monotonic()
                finished = time.monotonic()
                return Attempt(
                    cold=False,
                    waited_for_capacity_s=became_available - began,
                    first_token_s=(first_token - began) if first_token else None,
                    total_s=finished - began,
                    ok=first_token is not None,
                    detail="" if first_token else "stream carried no content",
                )
        except httpx.HTTPError as exc:
            return Attempt(
                cold=False,
                waited_for_capacity_s=0.0,
                first_token_s=None,
                total_s=time.monotonic() - began,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )


def drain_to_zero(server: modal.Server, client: httpx.Client, *, timeout: float) -> float:
    """Shrink the idle window until the pool empties. Returns seconds waited.

    Restoring the deployed configuration is the caller's job, in a `finally`.
    """
    server.update_autoscaler(scaledown_window=2, min_containers=0)
    began = time.monotonic()
    while time.monotonic() - began < timeout:
        try:
            response = client.get("/health", timeout=10.0)
        except httpx.HTTPError:
            time.sleep(2.0)
            continue
        if response.status_code == SERVICE_UNAVAILABLE:
            return time.monotonic() - began
        time.sleep(2.0)
    raise TimeoutError(
        f"the pool still had a container after {timeout:g}s; "
        "something is holding it warm (min_containers, or traffic from elsewhere)"
    )


# -- container-side records ---------------------------------------------
def container_records(volume_name: str, since_epoch: float) -> list[dict[str, object]]:
    """Startup breakdowns written by the containers this run started.

    Read from the Volume rather than from logs: a log line is a string someone
    has to parse, and this is the same JSON the container wrote.
    """
    with tempfile.TemporaryDirectory() as workdir:
        target = Path(workdir) / "startup.jsonl"
        completed = subprocess.run(
            ["modal", "volume", "get", volume_name, "/logs/startup.jsonl", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not target.is_file():
            print(f"  (no startup records: {completed.stderr.strip()[:200]})")
            return []
        records = []
        for line in target.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(entry)
    # `started_at` is ISO-8601 UTC; comparing strings would be fragile across
    # the boundary, so anything written after the run began is close enough:
    # the benchmark is the only thing starting containers.
    return records[-64:] if since_epoch else records



def collect(
    server: modal.Server,
    url: str,
    headers: dict[str, str],
    args: argparse.Namespace,
) -> list[Attempt]:
    """Run the cold and warm iterations, restoring the autoscaler whatever happens.

    The `finally` is not politeness: leaving `scaledown_window` at two seconds
    on a deployed Server turns every subsequent request into a cold start, and
    nothing in the dashboard says why.
    """
    attempts: list[Attempt] = []
    with httpx.Client(base_url=url, headers=headers, timeout=httpx.Timeout(600.0)) as client:
        try:
            for index in range(args.cold):
                drained = drain_to_zero(server, client, timeout=args.capacity_timeout)
                server.update_autoscaler(scaledown_window=args.restore_scaledown)
                attempt = complete_once(
                    client,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    capacity_timeout=args.capacity_timeout,
                )
                attempt.cold = True
                attempts.append(attempt)
                print(
                    f"cold {index + 1:>2}/{args.cold}  drained in {drained:5.0f}s  "
                    f"capacity {attempt.waited_for_capacity_s:6.1f}s  "
                    f"first token {attempt.first_token_s or float('nan'):6.1f}s  "
                    f"total {attempt.total_s:6.1f}s  {attempt.detail}"
                )
            for index in range(args.warm):
                attempt = complete_once(
                    client,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    capacity_timeout=args.capacity_timeout,
                )
                attempts.append(attempt)
                print(
                    f"warm {index + 1:>2}/{args.warm}  "
                    f"first token {attempt.first_token_s or float('nan'):6.1f}s  "
                    f"total {attempt.total_s:6.1f}s  {attempt.detail}"
                )
        finally:
            server.update_autoscaler(scaledown_window=args.restore_scaledown)
            print(f"\nscaledown_window restored to {args.restore_scaledown}s")
    return attempts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default=DEFAULT_APP)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--cold", type=int, default=10)
    parser.add_argument("--warm", type=int, default=10)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    parser.add_argument("--prompt", default="Write a Python function that reverses a list.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--token", default=None, help="proxy token; defaults to $INFERENCE_API_KEY")
    parser.add_argument("--volume", default="agentic-gpu-cache")
    parser.add_argument("--capacity-timeout", type=float, default=1200.0)
    parser.add_argument("--restore-scaledown", type=int, default=60)
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("INFERENCE_API_KEY", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if not token:
        print("warning: no proxy token; this only works on an unauthenticated Server\n")

    server = modal.Server.from_name(args.app, args.server)
    url = server.get_url()
    print(f"server: {url}\ncold: {args.cold}  warm: {args.warm}\n")

    began_epoch = time.time()
    attempts = collect(server, url, headers, args)

    cold = [a for a in attempts if a.cold and a.ok]
    warm = [a for a in attempts if not a.cold and a.ok]
    failures = [a for a in attempts if not a.ok]

    summaries = [
        summarise("cold: capacity wait", [a.waited_for_capacity_s for a in cold]),
        summarise("cold: first token", [a.first_token_s or 0.0 for a in cold]),
        summarise("cold: full completion", [a.total_s for a in cold]),
        summarise("warm: first token", [a.first_token_s or 0.0 for a in warm]),
        summarise("warm: full completion", [a.total_s for a in warm]),
    ]

    records = container_records(args.volume, began_epoch)
    for field_name, label in (
        ("volume_reload_ms", "container: volume reload"),
        ("model_resolve_ms", "container: model resolve"),
        ("vllm_launch_ms", "container: vllm launch"),
        ("readiness_ms", "container: readiness"),
        ("total_ms", "container: total startup"),
    ):
        values = [
            float(r[field_name]) / 1000
            for r in records
            if isinstance(r.get(field_name), int)
        ]
        summaries.append(summarise(label, values))

    print()
    for summary in summaries:
        print(render(summary))

    if cold:
        mean_cold = statistics.mean(a.total_s for a in cold)
        print(
            f"\ncost of one cold request at H100 rates: "
            f"${mean_cold * H100_DOLLARS_PER_SECOND:.3f} "
            f"({mean_cold:.0f} GPU-seconds)"
        )
    if failures:
        print(f"\n{len(failures)} attempt(s) failed:")
        for attempt in failures[:10]:
            print(f"  {attempt.detail}")

    if args.as_json:
        print(
            json.dumps(
                {
                    "attempts": [asdict(a) for a in attempts],
                    "summaries": [asdict(s) for s in summaries],
                    "container_records": records,
                },
                indent=2,
            )
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
