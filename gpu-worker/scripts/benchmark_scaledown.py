"""Find the `scaledown_window` that costs least for how this project is used.

    python scripts/benchmark_scaledown.py --windows 30,60,120,300 \
        --gaps 20,45,90,240

**This spends money**, and unlike the cold-start benchmark it spends it on
deliberate idleness. Read the estimate it prints before running it.

## The trade-off, stated as arithmetic

    scaledown_window small  ->  little idle GPU, many cold starts
    scaledown_window large  ->  few cold starts, much idle GPU

Both sides are GPU-seconds at the same $0.001097. So the question is not "how
long should the worker stay warm" but "for this pattern of requests, which
window buys fewer total GPU-seconds". A developer who tests once a minute and a
nightly batch job have opposite answers, and neither can be guessed.

## What it does

For each candidate window, it replays the same sequence of inter-request gaps
and counts what happened: how many requests found a warm container, how many
paid a cold start, and how many GPU-seconds were burnt idle in between. The idle
figure is computed, not measured — Modal bills container lifetime, and the
script knows exactly when it stopped sending requests.

A dry run (`--estimate`) prints the cost of the benchmark itself and exits,
because it is easy to ask for a sweep that costs more than a month of the
setting it is trying to choose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import httpx
import modal

H100_DOLLARS_PER_SECOND = 0.001097
# Modal's proxy answers this when the pool is empty; it means "booting".
SERVICE_UNAVAILABLE = 503
BAD_REQUEST = 400


@dataclass
class WindowResult:
    """What one candidate window cost over the replayed pattern."""

    window_seconds: int
    requests: int
    cold_starts: int
    warm_hits: int
    idle_gpu_seconds: float
    cold_start_seconds: float
    serving_seconds: float
    failures: int

    @property
    def gpu_seconds(self) -> float:
        return self.idle_gpu_seconds + self.cold_start_seconds + self.serving_seconds

    @property
    def dollars(self) -> float:
        return self.gpu_seconds * H100_DOLLARS_PER_SECOND

    @property
    def warm_hit_rate(self) -> float:
        return self.warm_hits / self.requests if self.requests else 0.0

    def render(self) -> str:
        return (
            f"{self.window_seconds:>5}s  "
            f"cold={self.cold_starts:>3}/{self.requests:<3} "
            f"warm={self.warm_hit_rate * 100:5.1f}%  "
            f"idle={self.idle_gpu_seconds:7.0f}s  "
            f"boot={self.cold_start_seconds:7.0f}s  "
            f"serve={self.serving_seconds:7.0f}s  "
            f"total={self.gpu_seconds:7.0f}s  ${self.dollars:6.2f}"
            + (f"  ({self.failures} failed)" if self.failures else "")
        )


def one_request(
    client: httpx.Client, *, model: str, prompt: str, max_tokens: int, capacity_timeout: float
) -> tuple[bool, float, float, bool]:
    """Returns (was_cold, seconds_waiting_for_capacity, seconds_serving, ok)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    began = time.monotonic()
    cold = False
    deadline = began + capacity_timeout
    while True:
        try:
            response = client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError:
            return cold, time.monotonic() - began, 0.0, False
        if response.status_code == SERVICE_UNAVAILABLE:
            cold = True
            if time.monotonic() >= deadline:
                return True, time.monotonic() - began, 0.0, False
            time.sleep(1.0)
            continue
        served_from = time.monotonic()
        ok = response.status_code < BAD_REQUEST
        # The capacity wait ends when the request is accepted; everything after
        # is generation, which the window does not influence.
        return cold, served_from - began, time.monotonic() - served_from, ok


def sweep(
    server: modal.Server,
    client: httpx.Client,
    *,
    window: int,
    gaps: Sequence[float],
    model: str,
    prompt: str,
    max_tokens: int,
    capacity_timeout: float,
) -> WindowResult:
    server.update_autoscaler(scaledown_window=window, min_containers=0)
    cold_starts = warm = failures = 0
    idle = boot = serving = 0.0

    for index, gap in enumerate([0.0, *gaps]):
        if gap:
            # Whatever the gap, the container stays alive for at most `window`
            # seconds of it, and that is what Modal bills.
            idle += min(gap, window)
            time.sleep(gap)
        was_cold, waited, served, ok = one_request(
            client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            capacity_timeout=capacity_timeout,
        )
        boot += waited if was_cold else 0.0
        serving += served
        cold_starts += 1 if was_cold else 0
        warm += 0 if was_cold else 1
        failures += 0 if ok else 1
        print(
            f"  [{window:>4}s] request {index + 1:>2}  "
            f"{'COLD' if was_cold else 'warm'}  wait={waited:6.1f}s serve={served:6.1f}s"
            + ("  FAILED" if not ok else "")
        )
    return WindowResult(
        window_seconds=window,
        requests=len(gaps) + 1,
        cold_starts=cold_starts,
        warm_hits=warm,
        idle_gpu_seconds=idle,
        cold_start_seconds=boot,
        serving_seconds=serving,
        failures=failures,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="agentic-gpu-worker")
    parser.add_argument("--server", default="VLLMServer")
    parser.add_argument("--windows", default="30,60,120,300")
    parser.add_argument(
        "--gaps",
        default="20,45,90,240",
        help="seconds between requests, replayed identically for every window",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    parser.add_argument("--prompt", default="Write a Python function that reverses a list.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--token", default=None)
    parser.add_argument("--capacity-timeout", type=float, default=1200.0)
    parser.add_argument("--restore-scaledown", type=int, default=60)
    parser.add_argument("--estimate", action="store_true", help="print the cost and exit")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    windows = [int(value) for value in args.windows.split(",") if value.strip()]
    gaps = [float(value) for value in args.gaps.split(",") if value.strip()]

    # A cold start is assumed to cost about three minutes of GPU time. It is a
    # guess, which is why this is labelled an estimate and why the cold-start
    # benchmark exists to replace the guess with a measurement.
    assumed_cold = 180.0
    worst_case = sum(min(sum(gaps), len(gaps) * window) + assumed_cold for window in windows)
    print(
        f"windows: {windows}\ngaps: {gaps}\n"
        f"upper bound on this benchmark: ~{worst_case / 60:.0f} GPU-minutes, "
        f"about ${worst_case * H100_DOLLARS_PER_SECOND:.2f}\n"
    )
    if args.estimate:
        return 0

    token = args.token or os.environ.get("INFERENCE_API_KEY", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    server = modal.Server.from_name(args.app, args.server)
    url = server.get_url()
    print(f"server: {url}\n")

    results: list[WindowResult] = []
    with httpx.Client(base_url=url, headers=headers, timeout=httpx.Timeout(600.0)) as client:
        try:
            for window in windows:
                results.append(
                    sweep(
                        server,
                        client,
                        window=window,
                        gaps=gaps,
                        model=args.model,
                        prompt=args.prompt,
                        max_tokens=args.max_tokens,
                        capacity_timeout=args.capacity_timeout,
                    )
                )
                print(f"  -> {results[-1].render()}\n")
        finally:
            server.update_autoscaler(scaledown_window=args.restore_scaledown)
            print(f"scaledown_window restored to {args.restore_scaledown}s\n")

    print("window  outcome")
    for result in results:
        print(result.render())

    if results:
        best = min(results, key=lambda r: r.gpu_seconds)
        print(
            f"\ncheapest for this pattern: scaledown_window={best.window_seconds}s "
            f"at {best.gpu_seconds:.0f} GPU-seconds (${best.dollars:.2f})"
        )
        print(
            "This is one pattern. Replay the gaps your own day actually has "
            "before writing the number into a profile."
        )
    if args.as_json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
