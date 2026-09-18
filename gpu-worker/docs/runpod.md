# Deploying on RunPod

This worker runs as an ordinary **RunPod Pod**. Not Serverless — see
[why](#why-not-serverless) at the end.

## What you need

| | |
|---|---|
| GPU | 1× H100 80GB or H200 (compute capability 9.0, required for hardware FP8) |
| Container disk | 20 GB — the image only; no weights ever land here |
| Network volume | **150 GB**, mounted at `/runpod-volume` ([sizing](persistent-storage.md#sizing)) |
| Exposed port | 8000 |
| Image | `<registry>/agentic-gpu-worker:<semver>` — a pinned tag, never `latest` |

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

* The external port changes whenever the Pod resets. That is handled by design
  here — the worker agent re-registers its endpoint on every boot — but any
  client holding a hard-coded address will break.
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
| Proxy | ✗ 100 s ceiling | yes (HTTPS) | yes |
| Direct TCP | ✓ | yes (needs API key) | IP yes on Secure Cloud, port no |
| Global networking | ✓ | no | yes |

## Streaming, and what it would change

Cloudflare's 524 fires when the origin fails to return **headers** in time. A
streaming completion sends headers immediately and then tokens, which should
keep the proxy viable for long generations. The wording RunPod publishes —
"maximum connection time" — is ambiguous enough that this must be **measured**
rather than assumed, which is what `first_token_latency_seconds` in the real
GPU validation is for.

Until it is measured on a real Pod, treat direct TCP or global networking as
the supported inference path.

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

Add a Pod. The worker registers itself with the control plane, starts
heartbeating, and receives jobs. Remove one by draining it: it stops accepting
work, finishes what it holds, and deregisters. Nothing restarts.

## Why not Serverless

RunPod Serverless runs a **handler** that pulls jobs from RunPod's own queue.
It has no stable, self-registering HTTP endpoint, and its lifecycle is owned by
RunPod rather than by our control plane — which schedules at the job level and
needs to reach a worker directly. Serverless also cold-starts from scratch,
which for a 31 GB model is exactly what the persistent volume exists to avoid.

Use Pods.
