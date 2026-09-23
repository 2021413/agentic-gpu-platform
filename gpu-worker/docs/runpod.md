# Deploying on RunPod

This worker runs as an ordinary **RunPod Pod**. Not Serverless — see
[why](#why-not-serverless) at the end.

## What you need

| | |
|---|---|
| GPU | 1× H100 80GB or H200 (compute capability 9.0, required for hardware FP8) |
| Container disk | **50 GB** — the image alone unpacks to ~24 GB; no weights ever land here |
| Network volume | **60 GB**, mounted at `/runpod-volume` ([sizing](persistent-storage.md#sizing)) |
| Exposed port | 8000 |
| Image | `<registry>/agentic-gpu-worker:<semver>` — a pinned tag, never `latest` |

The container disk looks large for an image that carries no weights, and it is
not a mistake: `vllm/vllm-openai:v0.28.0-cu129` is 9.7 GB in the registry but
**24.2 GB unpacked** (measured, not estimated — CUDA, PyTorch, the kernels and
the Python stack). RunPod unpacks it onto the container disk, so anything under
about 30 GB fails the pull. 50 GB leaves room for logs and a layer change.

A single H100 80GB fits this model comfortably: 31.2 GB of FP8 weights leave
ample room for KV cache at `MAX_MODEL_LEN=16384` and
`GPU_MEMORY_UTILIZATION=0.90`.

## Environment

Minimum:

```bash
PERSISTENT_ROOT=/runpod-volume
MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
MODEL_REVISION=dcaee4d4dfc5ee71ad501f01f530e5652438fde0   # pin this in production
VLLM_API_KEY=<generate one>                               # mandatory if the port is public
```

Every variable is documented in [`.env.example`](../.env.example). Secrets are
injected at runtime only: nothing is baked into the image, and nothing is
logged.

**Pin `MODEL_REVISION` in production.** Without it the worker resolves `main`,
which means a new upstream commit can change what a Pod serves on its next cold
start. An already-prepared volume is reused regardless, so the drift appears
only where it is hardest to notice.

## First boot, and every boot after

```text
BOOT → PREFLIGHT → VOLUME VALIDATION → MODEL PREPARATION
     → MODEL VERIFICATION → vLLM START → READINESS → READY
```

Cold start downloads 31.2 GB. **Warm restart does not**: the marker on the
volume is checked, the snapshot is verified against its recorded file sizes,
and vLLM starts against the local path. That is the entire reason the volume
exists.

Check what a Pod is doing:

```bash
worker-preflight       # hardware, volume, versions, model state
worker-ready           # blocks until /v1/models lists the served model
worker-smoke-test      # one real completion, validated
worker-health          # liveness only, used by HEALTHCHECK
```

## Reaching the worker — read this before choosing

A Pod can be reached three ways, and **they are not equivalent for inference**.

### 1. HTTP proxy — convenient, and limited

```text
https://<POD_ID>-8000.proxy.runpod.net
```

The route is `client → Cloudflare → RunPod load balancer → Pod`, and RunPod
documents the consequence plainly:

> **100-second timeout**: Cloudflare enforces a maximum connection time of 100
> seconds. If your service doesn't respond within this time, the connection
> closes with a 524 error.

A non-streaming completion returns nothing until generation finishes, so its
time-to-first-byte *is* the generation time. Short answers are fine; a long
patch or a large-context repair is not. The failure mode is the worst kind:
**intermittent** 524s that look like infrastructure failures, get retried on
another worker, and burn GPU time before the run fails for no legible reason.

Use the proxy for health checks, smoke tests and short completions. Do not use
it as the inference endpoint for long generations unless the client streams.

### 2. Direct TCP — the recommended inference path

Add 8000 to **Expose TCP Ports**. RunPod assigns a public IP and an external
port; there is no Cloudflare in the path and therefore no 100-second ceiling.

```text
TCP port 213.173.109.39:13007 -> :8000
```

Two things to know:

* The external port changes whenever the Pod resets, and **this image does
  nothing about that**: re-registering an endpoint belongs to the control
  plane's worker agent, a separate component. Any client holding a hard-coded
  address will break on a reset.
* IPs are stable on **Secure Cloud** and may change on Community Cloud if a Pod
  is migrated.
* The port is genuinely public, so `VLLM_API_KEY` stops being optional.

Ports numbered above 70000 request symmetrical mapping if you need the external
port to match the internal one.

### 3. Global networking — best when the control plane is also on RunPod

Every Pod in the account gets a private IP reachable only from your other Pods.
No public exposure, no proxy, no timeout ceiling. 100 Mbps between Pods, which
is ample for JSON request bodies. NVIDIA GPU Pods only.

If the orchestrator runs on RunPod too, this is the right answer.

### Summary

| Path | Long generations | Public exposure | Stable address |
|---|---|---|---|
| Proxy | only if the client streams | yes (HTTPS) | yes |
| Direct TCP | ✓ | yes (needs API key) | IP yes on Secure Cloud, port no |
| Global networking | ✓ | no | yes |

## Streaming, and why it fixes the proxy

Cloudflare documents 524 precisely, and the precision matters:

> Error 524 indicates that Cloudflare successfully connected to the origin web
> server, but the origin did not provide an HTTP response before the default
> 125 seconds **Proxy Read Timeout**. [...] The error 524 occurs if the origin
> web server acknowledges the resource request after the connection has been
> established, but does not send a timely response within the Proxy Read
> Timeout delay.

It is a **read** timeout, not a cap on total connection time. A proxy read
timeout is rearmed by every chunk the origin sends. So:

* A **non-streaming** completion sends nothing until generation ends. Its
  time-to-first-byte is its generation time, and past the window it becomes a
  524.
* A **streaming** completion sends headers immediately and then a token every
  few tens of milliseconds. The read timer never expires, and total duration
  stops mattering.

(RunPod documents 100 seconds where Cloudflare documents 125; RunPod presumably
configures it lower. The mechanism is identical either way.)

**Consequence for a client of this worker:** request `"stream": true` and
accumulate the deltas, and the proxy becomes viable for long generations. A
client that sends `"stream": false` must use direct TCP or global networking
instead.

This is a mechanism, not a measurement. RunPod's load balancer sits between
Cloudflare and the Pod and may impose limits of its own, so the opt-in real-GPU
validation records `first_token_latency_seconds` and exercises both paths.
Until that has run on real hardware, direct TCP remains the recommendation for
non-streaming clients.

## Multiple GPUs

The same image serves 1× or 2× H100/H200. `TENSOR_PARALLEL_SIZE` defaults to 1
and is never inferred: `AUTO_TENSOR_PARALLEL=true` opts into using every
visible GPU.

That default is deliberate. Combining two GPUs into one wider worker halves the
number of independently schedulable workers, which is the opposite of how this
fleet scales — the orchestrator adds Pods, it does not widen them. Widen only
when a model does not fit, or when single-stream latency matters more than
throughput.

## Scaling

Add a Pod, and it serves inference. That is all this image does.

**Registration, heartbeats and draining are not part of it.** Those belong to
the control plane's worker agent, which runs alongside and is documented with
the orchestrator. Stated plainly because the two are easy to conflate:
`grep -r heartbeat src/` in this project returns nothing.

What this image guarantees is narrower and worth having on its own — a Pod that
comes up serving the right model, on a volume that does not re-download 31 GB,
and that refuses to claim it is ready when it is not.

## Automated deployment

`tools/runpod_deployer` drives all of the above through the RunPod API instead
of the console:

```bash
export RUNPOD_API_KEY=...          # read from the environment only, no flag
python -m tools.runpod_deployer gpu-types      # what is available, and at what price
python -m tools.runpod_deployer deploy         # create, wait for ready, smoke test
python -m tools.runpod_deployer status
python -m tools.runpod_deployer smoke-test
python -m tools.runpod_deployer destroy --yes  # never destroys without this
```

`deploy` waits for real readiness — `/v1/models` must list the served model —
and then runs one real completion, so a Pod that comes up broken is reported as
broken rather than as deployed. `WORKER_IMAGE` selects the image to run.

The key is read from `RUNPOD_API_KEY` and nowhere else: there is no flag, no
file path and no default, and it is redacted from every log and error.

## Why not RunPod Serverless

RunPod Serverless runs a **handler** that pulls jobs from RunPod's own queue.
It has no stable, self-registering HTTP endpoint, and its lifecycle is owned by
RunPod rather than by our control plane — which schedules at the job level and
needs to reach a worker directly. Serverless also cold-starts from scratch,
which for a 31 GB model is exactly what the persistent volume exists to avoid.

On RunPod, use Pods.

### This is not an argument against serverless GPUs in general

Both objections above are specific to RunPod's implementation, and
[Modal](modal.md) answers both: a Modal Server is an ordinary HTTP server at a
stable HTTPS URL, reached directly by the control plane, and it mounts a
persistent Volume so a cold start reads 31.2 GB off a disk rather than off the
internet.

`infra/modal/` is that deployment. It scales to zero, which a Pod cannot, and it
does not go looking for capacity in one region at a time — which is what made
this document's advice expensive on the days EU-FR-1 had none.
