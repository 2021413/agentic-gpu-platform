# Agentic GPU Worker

A reproducible RunPod GPU worker image that serves
`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` through vLLM's OpenAI-compatible API.

```text
RunPod GPU Pod
  ├── Docker image  (immutable, version-pinned, no weights)
  │     ├── bootstrap ─ preflight ─ model preparation
  │     ├── vLLM
  │     └── health / readiness / smoke test
  └── Network volume at /runpod-volume  (weights, caches, state)
```

**The invariant everything else serves:**

```text
immutable image + ephemeral GPU + persistent external volume + runtime config
```

No model weight is ever baked into a layer, downloaded during the build, or
written to the container filesystem. Destroying and recreating a Pod with the
same volume attached does **not** re-download 31 GB.

---

## Quick start

```bash
docker build -t agentic-gpu-worker:1.0.0 .
```

Deploy as a RunPod Pod with a 150 GB network volume mounted at
`/runpod-volume`, port 8000 exposed, and:

```bash
MODEL_REVISION=dcaee4d4dfc5ee71ad501f01f530e5652438fde0
VLLM_API_KEY=<generate one>
```

Then:

```bash
worker-preflight     # hardware, volume, versions, model state
worker-ready         # blocks until /v1/models lists the served model
worker-smoke-test    # one real completion, validated end to end
```

Full deployment guide: [`docs/runpod.md`](docs/runpod.md).

---

## What it actually verifies

The bootstrap sequence is a series of refusals, each one a real deployment
mistake caught before it becomes expensive:

```text
BOOT → PREFLIGHT → VOLUME VALIDATION → MODEL PREPARATION
     → MODEL VERIFICATION → vLLM START → READINESS → READY
```

* **The volume is a real mount**, proven by comparing device numbers. A volume
  that failed to attach looks exactly like an empty directory — which is how
  31 GB ends up in a layer that evaporates on the next restart.
* **There is room before the first byte**, and a full disk is reported as a full
  disk rather than retried. The second attempt fills the same volume, slower.
* **One downloader at a time**, behind an advisory lock the kernel releases if
  the holder dies. Two Pods on one volume would otherwise corrupt the blob store
  and pay twice.
* **Downloads resume, they do not restart.** Deleting partial blobs to "start
  clean" is how a flaky network becomes an unbounded bill.
* **The model is verified before it is trusted.** A snapshot directory existing
  proves nothing; the marker records every file and its size, so a warm boot
  verifies offline in seconds.
* **Ready means serving.** vLLM accepts connections long before the weights are
  resident, so readiness checks that `/v1/models` lists the expected model.
* **The smoke test rejects an empty answer.** A 200 carrying no content is a
  broken worker, and letting it pass advertises capacity that produces nothing.

---

## Configuration

Environment-driven, validated as a whole at boot. Every key is documented in
[`.env.example`](.env.example); the defaults matter:

| Variable | Default | Why |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | 31.2 GB, FP8 e4m3 block-quantised |
| `MODEL_REVISION` | unset | **pin it in production**, or `main` can move under you |
| `PERSISTENT_ROOT` | `/runpod-volume` | where RunPod mounts network volumes |
| `MIN_FREE_DISK_GB` | `60` | weights + a resuming download + caches |
| `MAX_MODEL_LEN` | `16384` | the model supports 262144; the KV cache is what costs |
| `GPU_MEMORY_UTILIZATION` | `0.90` | leaves room for the CUDA context |
| `TENSOR_PARALLEL_SIZE` | `1` | never inferred — see below |
| `VLLM_API_KEY` | unset | mandatory once the port is public |

`VLLM_EXTRA_ARGS` is split with `shlex` and passed as an argument vector. It is
never evaluated by a shell, so no environment variable can inject a command.

### Why tensor parallelism is opt-in

`AUTO_TENSOR_PARALLEL=true` uses every visible GPU. It is off by default because
combining two GPUs into one wider worker halves the number of independently
schedulable workers — the opposite of how this fleet scales. The orchestrator
adds Pods; it does not widen them. Widen only when a model does not fit, or when
single-stream latency matters more than throughput.

---

## Reaching the worker

**This decides whether long generations work at all.** RunPod's HTTP proxy sits
behind Cloudflare and enforces a 100-second ceiling, returning 524 beyond it. A
non-streaming completion returns nothing until it finishes, so its
time-to-first-byte *is* its generation time.

Use direct TCP exposure or global networking for inference. The full comparison,
and what streaming would change, is in
[`docs/runpod.md`](docs/runpod.md#reaching-the-worker--read-this-before-choosing).

---

## Safety

Tool execution is not this component's concern, but exposure is:

* Secrets come from the runtime environment only. Nothing is baked into a layer,
  passed as a build argument, or written to a log — `Secret` refuses to render
  itself precisely because printing a config object is what people do at 3am.
* vLLM's API key stops casual access, not a determined one. A public TCP port in
  front of a GPU worth several dollars an hour deserves a private network or a
  reverse proxy. Said plainly rather than implied.

---

## Tests

```bash
pytest tests -q                    # no GPU, no network, no container
pytest -m integration tests -q     # reaches huggingface.co with a 300 KB model
pytest -m runpod tests -q          # provisions a real Pod; costs money; opt-in
```

The default selection really is offline: `addopts` in `pyproject.toml`
deselects `integration`, `gpu` and `runpod`, so the promise above is enforced by
configuration rather than by convention.

CI lives at the repository root, in
[`.github/workflows/gpu-worker-ci.yml`](../.github/workflows/gpu-worker-ci.yml),
because GitHub only reads workflows from there. It never needs a GPU and never
downloads the model, and it asserts the invariant mechanically rather than
trusting it: the built image is exported and searched for weight files, the
cache variables are checked to point at the volume, and booting without a
volume must exit 3.

The real-GPU validation is deliberately opt-in behind both `RUNPOD_API_KEY` and
an explicit flag, and records `cold_start_seconds`, `model_download_seconds`,
`model_load_seconds`, `warm_restart_seconds` and `first_token_latency_seconds`.

---

## Deploying from the command line

`tools/runpod_deployer` creates a Pod, waits for genuine readiness, runs a real
completion, and refuses to destroy anything without `--yes`. The RunPod key is
read from `RUNPOD_API_KEY` and nowhere else — no flag, no file, no default — and
is redacted from every log and error path. See
[`docs/runpod.md`](docs/runpod.md#automated-deployment).

## Documentation

| | |
|---|---|
| [`docs/runpod.md`](docs/runpod.md) | deployment, networking, scaling, why not Serverless |
| [`docs/persistent-storage.md`](docs/persistent-storage.md) | layout, sizing, the marker, the lock |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | every failure mode and its diagnostic |

---

## Releases

Images are tagged with an immutable `<semver>` and the `<git-sha>` they were
built from. `latest` is never used in a deployment example: a Pod that silently
picks up a new image is a Pod that silently changes what it serves.
