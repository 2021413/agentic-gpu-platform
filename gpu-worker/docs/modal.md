# Deploying on Modal

This worker runs as a **Modal Server**: a container that speaks HTTP natively,
billed per second of container life, scaled to zero when idle, behind a stable
HTTPS URL.

```text
control plane
   │  Authorization: Bearer wk-<id>.ws-<secret>
   ▼
https://<workspace>--agentic-gpu-worker-vllmserver.us-east.modal.direct
   │  (Modal proxy — 401 before any container on a bad token,
   │                 503 when the pool is empty)
   ▼
Modal Server, H100, min_containers=0
   ├── vLLM subprocess, OpenAI-compatible, port 8000
   └── Modal Volume at /data
         ├── huggingface/hub/     31.2 GB of FP8 weights
         ├── compiled/<gpu>/<image>/  CUDA graphs, Triton kernels
         ├── state/model-ready.json   the marker
         └── logs/startup.jsonl       one line per container boot
```

The invariant is the one the RunPod image already enforced; only the mount point
and the lifecycle owner change:

```text
immutable image + ephemeral GPU + persistent external volume + runtime config
```

---

## What is different from a Pod, and what it costs you

| | RunPod Pod | Modal Server |
|---|---|---|
| Billing | from creation to destruction | per second of container life |
| Idle GPU | paid until someone destroys it | paid for `scaledown_window` seconds after the last request |
| Address | TCP host:port, changes on every reset | stable HTTPS URL |
| Auth | `VLLM_API_KEY` on a public port | Modal proxy token, checked before any container |
| No worker available | connection refused | **HTTP 503, immediately** |
| Long generations | 100 s ceiling via the HTTP proxy, none over direct TCP | no platform ceiling; the client sets the timeout |

**The 503 is the one thing a caller must be taught.** A Modal Server uses a
stateless reverse proxy and does not queue: with an empty pool it rejects the
request and starts a container in response to it. The control plane therefore
needs `INFERENCE_SCALE_TO_ZERO=true`, which makes its adapter wait for capacity
instead of failing, and makes a scaled-to-zero worker report as healthy rather
than being evicted from the registry for being idle.

Getting that wrong does not look like a bug. It looks like a bill: every first
request after an idle period fails, the job is requeued, and the retried job
runs on the container the failed one paid to start.

---

## First-time setup

```bash
cd gpu-worker
uv pip install -e '.[modal]'        # the Modal client is a deploy-time tool
modal setup                         # opens a browser, writes ~/.modal.toml
modal token info                    # confirm the workspace
```

Then the three resources the deployment expects:

```bash
# 1. the Volume (created on first use, but make it explicit)
modal volume create agentic-gpu-cache

# 2. the Secret — HF_TOKEN only; there is no VLLM_API_KEY any more
modal secret create agentic-gpu-worker HF_TOKEN=hf_...

# 3. the proxy token the control plane will present
modal workspace proxy-tokens create
#    -> wk-<id> and ws-<secret>
#    the control plane wants them joined by a period:
#    INFERENCE_API_KEY=wk-<id>.ws-<secret>
```

**Populate the Volume before the first deployment.** This is not optional: a
container that finds no prepared model refuses to start rather than download
31.2 GB with an H100 on the meter.

```bash
modal run scripts/populate_modal_volume.py          # ~30-60 min, CPU only
modal run scripts/populate_modal_volume.py --verify # check, download nothing
```

---

## Day to day

```bash
modal run    infra/modal/app.py     # print the profile and the URL; starts no GPU
modal serve  infra/modal/app.py     # development, hot reload on save
modal deploy infra/modal/app.py     # a persistent deployment
```

All three from the `gpu-worker/` directory: Modal puts the working directory on
`sys.path`, which is what makes `infra.modal.*` and `worker.*` importable.

The development loop this replaces:

```text
before                          after
──────                          ─────
docker build (20 min)           edit a file
docker push (24 GB)             modal serve reloads
create pod                      request
wait for the image pull         GPU boots, or is already warm
test                            GPU stays warm for scaledown_window
stop the pod                    scales to zero on its own
```

A code change rebuilds nothing. `add_local_python_source(..., copy=False)`
attaches the sources at container start instead of baking them into a layer, so
CUDA, PyTorch and vLLM are never reinstalled for an edit to `runtime.py`.

---

## Profiles

Defined in `infra/modal/config.py`, selected by `MODAL_PROFILE`:

| | dev | prod |
|---|---|---|
| `min_containers` | 0 | 0 |
| `max_containers` | 1 | 4 |
| `scaledown_window` | 130 s | 130 s |
| `startup_timeout` | 900 s | 900 s |
| `exit_grace_period` | 120 s | 300 s |
| `unauthenticated` | allowed | **refused** |

`max_containers=1` in dev is a cost guard, not a capacity statement: a loop that
fans out ten requests would otherwise provision ten H100s, and the mistake is
only visible on the bill. Modal's own guidance prefers leaving
`target_concurrency` unset over `max_containers=1` for a singleton, because the
cap also prevents a replacement container during a rolling redeployment. That is
the trade accepted here — a few seconds of 503 during a redeploy, against an
accidental fleet.

Any single field can be overridden for an experiment without editing the file:

```bash
MODAL_SCALEDOWN_WINDOW=300 modal deploy infra/modal/app.py
```

The profile refuses configurations that would waste money or expose the worker:
a `startup_timeout` too short for a cold vLLM start (the container is killed
mid-load and the loop only shows up as 503s), an idle window longer than an
hour, `unauthenticated=True` in prod, and GPU snapshots without CPU snapshots.

---

## Cost

H100 SXM at **$0.001097/second**, about $3.95/hour (confirmed by `modal billing rates`; an H200 is $4.54/hour and costs the same as an H100 when Modal substitutes one). Volume storage at
$0.09/GiB/month, so 40 GiB of weights and caches is roughly $3.60/month — about
fifty-five minutes of H100 time, which is why the Volume is never the thing to
economise on.

What to watch, per `modal billing report`:

```bash
modal billing report --for today --show-resources
modal billing report --for "this month" --show-resources --tag-names service,profile
```

The apps are tagged `service=gpu-worker`, `project=agentic` and
`profile=dev|prod`, so the report separates them.

The figure that matters is **cost per successful inference**, not $/GPU-hour.
Three things move it, in this order:

1. **Cold starts you did not need.** Measured by `scripts/benchmark_cold_start.py`,
   which splits the boot into volume reload, model resolution, vLLM launch and
   readiness — because "the cold start is slow" has at least three different
   fixes and one number tells you none of them.
2. **Idle GPU you did not want.** Once the cold start is measured this stops
   being a search and becomes arithmetic. Idle GPU and booting GPU bill at the
   same rate, so the window that minimises GPU-seconds is the one where waiting
   costs what booting costs — the cold start's own duration, here ~130 s. Below
   it, a request arriving inside the window is strictly cheaper served warm;
   above it, you are paying more to avoid a boot than the boot costs. That is
   the ski-rental bound, and it caps the worst case at twice optimal without
   predicting anything. `scripts/benchmark_scaledown.py` exists to check that
   reasoning against a real day's traffic, not to discover the number.
3. **Anything on the critical path that is not inference.** A download, a hub
   lookup, a recompilation. Hence the preload script, `HF_HUB_OFFLINE=1` once
   the snapshot is local, and a compile cache keyed by the GPU actually
   attached.

```bash
python scripts/benchmark_scaledown.py --estimate    # what the benchmark itself costs
python scripts/benchmark_cold_start.py --cold 10 --warm 10
```

Both spend real money. Both say so before they start.

---

## Why the compile cache is keyed by GPU

`gpu="H100"` may be served by an H200 — Modal upgrades for free when H100s are
scarce, which is exactly the availability problem this migration exists to
solve. But CUDA graphs and Triton kernels compiled on one are not valid on the
other, and reusing them produces a *slower* worker rather than a failing one,
which is much harder to notice.

So `infra/modal/runtime.py` reads the device name from `nvidia-smi` at startup
and points `VLLM_CACHE_ROOT` at `/data/compiled/<gpu>/<image-tag>/`. Benchmarks
should use `gpu="H100!"`, which refuses the upgrade: a distribution measured
across both is two distributions reported as one.

---

## Troubleshooting

**`ColdModelError` on startup, container marked failed.** The Volume holds no
prepared model for this `MODEL_ID`. Run the populate script. The refusal is
deliberate and costs a few seconds of GPU time instead of forty minutes.

**Every request returns 503 and nothing boots.** Check `startup_timeout` and the
app logs: a container killed mid-load is replaced by another that loads from
scratch, and the client only ever sees 503. `modal app logs agentic-gpu-worker`.

**401 from the proxy.** The token is wrong, or it is an API token (`ak-`/`as-`)
rather than a proxy token (`wk-`/`ws-`). They are not interchangeable.

**The model loads but `/v1/models` lists something else.** The Server refuses to
report ready in that case, by design — it is the same check the RunPod image
uses. Usually a `MODEL_REVISION` mismatch between the populate run and the
deployment.

**Startup timings.** Every container appends one JSON line to
`/data/logs/startup.jsonl`:

```bash
modal volume get agentic-gpu-cache /logs/startup.jsonl -
```

---

## Measured, 22 September 2026

One workspace, one H100 (`nvidia-h100-80gb-hbm3`), `Qwen3-Coder-30B-A3B-Instruct-FP8`
at 16384 context. These are readings, not estimates; the breakdown comes from
the `StartupRecord` each container writes to `/data/logs/startup.jsonl` and from
vLLM's own log lines.

| | first boot | after both fixes |
|---|---|---|
| volume reload | 0.18 s | 0.18 s |
| model resolve (marker + verify) | 0.53 s | 0.53 s |
| vLLM launch (fork) | 0.002 s | 0.002 s |
| weight load | **338.1 s** (82 s/shard) | **54.2 s** (7.7 s/shard) |
| init engine | 215.4 s (compile 52.6 s) | 56.7 s (compile 6.1 s) |
| **readiness, total** | **643 s — $0.71** | **178 s — $0.19** |
| warm completion (16 tokens) | — | 3.5 – 4.0 s |

Everything outside vLLM comes to **0.7 seconds**. That is the whole argument for
splitting the measurement: a single `cold_start_seconds` of 643 would have sent
somebody looking at Modal's scheduler, the Volume, or the image, and all three
were already fast.

Two changes account for the difference, and the logs attribute them:

* **`--safetensors-load-strategy=prefetch` — about −284 s.** A Modal Volume is
  mounted over 9P and vLLM does not recognise that as a network filesystem, so
  it turns its own read-ahead off and loads the four shards serially. It says so
  in the log, and naming the strategy is what that message asks for.
* **A warm compile cache on the Volume — about −159 s.** Keyed by the GPU
  actually attached, so an H200 substitution cannot silently reuse an H100's
  graphs.

**Third cold start, compile cache fully warm** — `scripts/benchmark_cold_start.py --cold 1 --warm 2`:

```text
cold  1/1   capacity 138.0s   first token 138.0s   total 139.0s
warm  1/2                     first token   0.7s   total   1.0s
warm  2/2                     first token   0.3s   total   0.9s

container: volume reload    0.1s
container: model resolve    0.1s
container: vllm launch      0.0s
container: readiness      129.3s
container: total startup  129.6s

cost of one cold request at H100 rates: $0.152 (139 GPU-seconds)
```

So the arc is 643s → 178s → **129.6s**, and $0.71 → $0.19 → **$0.15**. A warm
request answers its first token in **0.3–0.7 s**.

**The wiring, end to end.** Not curl: the control plane's own
`OpenAICompatibleLLMProvider`, built from the same `Settings` the orchestrator
uses, against an empty pool.

```text
scale_to_zero             = True
cold_start_max_wait       = 900.0s
health() on an empty pool = True      <- or the registry evicts the only worker
completing through the adapter, pool empty...
waited 220s, cost ~$0.24 of H100
finish_reason = stop
usage         = TokenUsage(input_tokens=28, output_tokens=16)
```

The 503s were waited out inside the adapter. No job failed, nothing was
requeued, and the completion that paid for the cold start is the one that got
the answer.

Other measurements worth keeping:

* **Populating the Volume: 31.2 GB in 76 seconds**, on a CPU container, for
  about one cent. Doing the same download on the H100 would cost roughly $0.08
  in GPU time alone — and forty times that if it happened on every cold start.
* **A refusal costs 0.6 – 1.4 s.** A container that finds no prepared model dies
  in about a second. Modal retried it three times for one request: four seconds
  of H100 in total, against the forty minutes an accidental download would have
  billed.
* **`max_containers=1` holds.** Twenty `/health` polls against an empty pool,
  every one answered 503, produced exactly one container.
* **Redeploying a code change takes 15 seconds**, against 240 for the first
  deploy that had to pull and build the image.

## Memory snapshots: measured, and turned back off

Modal's GPU memory snapshots promise up to a 10× faster cold start, and the
code to use them is in this repository behind `MODAL_ENABLE_MEMORY_SNAPSHOT`
and `MODAL_ENABLE_GPU_SNAPSHOT`. **They are off, because on this Server they
made cold starts slower.**

The pattern implemented is Modal's own: warm the engine with three real
completions so the lazily-built CUDA graphs are captured, put vLLM to sleep so
the weights move to host memory, let Modal snapshot, then wake on restore. All
of it works — the logs show `fall asleep` in 10.0 s and `wake up` in 1.7 s. What
does not work is the saving:

```text
Restoring Function from memory snapshot.
launching: vllm serve ...              <- a full boot, every time
Model loading took 28.3 s
init engine took 32.8 s
ready after 123 s
warmup 1/3 ok ... warmup 3/3 ok
It took 10.0 seconds to fall asleep.
warmed and asleep in 10430ms; snapshotting
restored and serving in 1714ms
```

Every container creates a snapshot and no container is ever spared a boot by
one. `@modal.enter(snap=True)` captures the Modal runtime's own process; vLLM
runs in a subprocess, and on `@app.server` that subprocess is not in the
snapshot. Modal's published example that does benefit uses the older
`@app.cls` + `@modal.web_server` pair.

Measured cost of enabling them anyway: **+20 s on every cold start** (13 s of
warmup, 10 s falling asleep, 1.7 s waking) for nothing. Worse, the first boot
after enabling took 474 s, because `--enable-sleep-mode` changes vLLM's
configuration hash and therefore invalidated the compiled-artifact cache:
`init engine took 322.17 s (compilation: 261.46 s)`.

One real effect is worth recording: after the warmup, the first completion came
back in **1.43 s** instead of ~4 s, because the first request no longer pays for
the lazily-built graphs. That is a 2.6 s saving bought with 13 s of boot, so it
does not pay for itself here either.

Revisit if Modal extends snapshot capture to subprocesses, or if this worker
moves to `@app.cls`.

## What is still unproven

* No cold start has been measured against a *cold* compile cache **with**
  prefetch on, so the two savings above are attributed from vLLM's own timings
  rather than isolated experimentally.
* `scripts/benchmark_cold_start.py` and `scripts/benchmark_scaledown.py` have
  not been run. `scaledown_window` is still 60 s in dev and 120 s in prod
  because those were reasonable starting points, not because anything measured
  them against a real day's request pattern.
* Nothing has run under concurrent load. `target_concurrency` is unset, so one
  container serves one request at a time.
