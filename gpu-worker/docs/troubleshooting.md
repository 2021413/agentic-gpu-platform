# Troubleshooting

Every failure below is detected deliberately and reported with a diagnostic
naming the cause. If you hit one that is not here and the message was not
useful, that is a bug worth reporting.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | configuration error — an operator mistake; retrying will not help |
| 3 | persistent storage error — volume missing, read-only, or full |
| 4 | model preparation failed |
| 5 | not ready within the deadline |
| 6 | smoke test failed |

## Storage

### `does not exist, so no persistent volume is attached`

The Pod has no network volume at `PERSISTENT_ROOT`. Attach one with mount path
`/runpod-volume`, or point `PERSISTENT_ROOT` at where yours is mounted.

### `is on the container filesystem, not a mounted volume`

The directory exists but is not a separate mount — the volume silently failed
to attach. This is caught by comparing device numbers, because an unattached
volume looks exactly like an empty directory, and downloading into it would
lose 31 GB on the next restart while filling the container disk.

`WORKER_ALLOW_EPHEMERAL_STORAGE=1` overrides this for throwaway tests only.

### `is not writable`

A read-only volume, or a quota already exhausted. Both pass `os.access`, which
is why the check writes, flushes and fsyncs a probe file instead.

### `the volume ran out of space while downloading`

Reported with the actual capacity and never retried: a second attempt fills the
same disk more slowly. Grow the volume — see [sizing](persistent-storage.md#sizing) —
or remove an old revision from the hub cache.

### `only N GB free, M GB required`

The pre-download check. Lower `MIN_FREE_DISK_GB` only if you know the model is
smaller than the default assumes.

## Model preparation

### `another process has held the model download lock`

Two Pods share this volume and one is already downloading. That is the lock
doing its job. Wait, raise `MODEL_LOCK_TIMEOUT_SECONDS`, or stop the other Pod.
A lock whose holder died is released by the kernel, so this is never stale.

### `the cached model failed verification and will be completed`

A shard is missing or truncated, usually after a killed download. The download
resumes; nothing is deleted.

### `download failed after N attempts`

Transient network failures were retried with exponential backoff and ran out.
Check outbound connectivity. For a gated repository, check `HF_TOKEN` is set
and authorised.

### `marker describes X@Y, which is not what was requested`

`MODEL_ID` or `MODEL_REVISION` changed. The new model is prepared; the old one
is left on the volume. Remove it yourself if you need the space.

### Xet transfer problems

`HF_HUB_DISABLE_XET=1` falls back to the classic transfer path. Try it if
downloads stall or fail in a way that mentions Xet.

## GPU

### `no GPU is visible: vLLM will fail to start`

`nvidia-smi` is absent or reported nothing. On RunPod this means the Pod was
created without a GPU, or the container lacks the NVIDIA runtime.

### `at least one GPU lacks hardware FP8`

Compute capability below 8.9. This model is FP8-quantised; on older hardware it
is dequantised at a large speed cost. Use H100, H200 or L40S class GPUs.

### `TENSOR_PARALLEL_SIZE exceeds the visible GPU(s)`

vLLM will refuse to start. Either lower it or deploy a Pod with more GPUs.

### CUDA initialisation failure / driver mismatch

The pinned base image is CUDA 12.9 and expects a driver in the 535–570
branches. A host outside that range needs the CUDA 13 variant instead — swap
`VLLM_IMAGE` to `vllm/vllm-openai:v0.28.0` and rebuild. `worker-preflight`
prints the driver version it found.

### Not enough GPU memory / vLLM exits during model load

Lower `GPU_MEMORY_UTILIZATION` (0.85 is a reasonable retry) or lower
`MAX_MODEL_LEN`: the KV cache is what grows with context length. The model
weights alone are 31.2 GB, so an 80 GB card has room, but the margin shrinks
quickly at long contexts.

## Serving

### `not ready after N s: serving (), expected '...'`

vLLM is up but lists no model, or lists a different one. Usually a wrong
snapshot path or a stale `SERVED_MODEL_NAME`. `worker-serve-args --redacted`
prints the exact vector it was launched with.

### Readiness timeout

Loading 31 GB takes minutes, and the first run also compiles CUDA graphs.
`READINESS_TIMEOUT_SECONDS` defaults to 1800 for that reason. A genuine timeout
usually means the load failed — check the vLLM logs rather than raising the
deadline.

### `port already in use`

Something else is on `PORT` inside the container. On RunPod this normally means
the entrypoint ran twice; only one vLLM process should ever exist, started with
`exec`.

### Smoke test fails with `the model produced no output`

The server answered 200 with an empty string. The worker is broken and must not
be advertised as ready — this check exists precisely to catch it.

### HTTP 524 from `*.proxy.runpod.net`

Cloudflare's 100-second ceiling. Not a worker fault. See
[reaching the worker](runpod.md#reaching-the-worker--read-this-before-choosing).

## Security

### `api key    NOT SET (open port)` in the preflight banner

`VLLM_API_KEY` is unset. **The worker starts anyway** — it has no way to know
whether its port is reachable from the internet, and refusing to boot would
break every private-network deployment. The banner says so on every start, and
it is the operator's job to act on it.

If the Pod exposes a public TCP port, treat this line as a defect and set the
key. The key is passed to vLLM through the environment and never appears in the
argument vector, so it does not reach `/proc/1/cmdline` or any log.

Note honestly what an API key buys you: it stops casual access, not a
determined one. A public TCP port on a GPU worth several dollars an hour
deserves a private network or a reverse proxy in front of it.
