# Persistent storage

The volume is the only thing in this design that is allowed to survive. The
image is immutable, the GPU is ephemeral, and the container filesystem is
disposable — so every byte that would be expensive to fetch again lives on a
RunPod network volume mounted at `/runpod-volume`.

## Layout

```text
/runpod-volume/
  huggingface/          HF_HOME
    hub/                HUGGINGFACE_HUB_CACHE — the blob store and snapshots
  models/               reserved for manually placed weights
  vllm/                 VLLM_CACHE_ROOT — compiled graphs and kernels
    triton/             TRITON_CACHE_DIR
  torch/                TORCH_HOME
  tmp/                  TMPDIR — scratch space, kept off the container disk
  xdg/                  XDG_CACHE_HOME
  state/
    model-ready.json    the readiness marker
    model-download.lock the advisory download lock
  logs/
```

Every path is derived from `PERSISTENT_ROOT` in `worker/config.py`. If a
directory is not derived from it, that is a bug: it means something large is
being written to a filesystem that disappears with the Pod.

### Why the cache variables are set in the image, not the entrypoint

`huggingface_hub` and `torch` read `HF_HOME`, `HF_HUB_CACHE` and `TORCH_HOME`
**at import time**. Exporting them from a script that runs after those modules
are imported silently writes gigabytes into the container filesystem. They are
therefore `ENV` in the Dockerfile as well as part of `PersistentLayout`.

## Sizing

The model is **31.2 GB** across four safetensors shards (verified against the
repository, not estimated). A volume sized to that number alone will fail the
first time anything else needs room.

| What | Size |
|---|---|
| Model weights, one revision | 31.2 GB |
| Peak extra while resuming (partial blobs in `hub/blobs/*.incomplete`) | ~10 GB |
| vLLM / Triton compiled cache | 1–5 GB |
| Temporary files | 1–2 GB |
| **Single revision, comfortable** | **~50 GB** |
| A second revision, for a rolling model upgrade | +31.2 GB |

**Recommended volume: 150 GB.** Absolute minimum for one resident revision:
80 GB. Snapshots are symlinks into the blob store, so a snapshot does *not*
double the cost — which is also why `directory_size_bytes` counts each inode
once, and why a naive `du -L` will tell you the model is twice its real size.

`MIN_FREE_DISK_GB` defaults to **60**: enough for a full cold download plus the
resume headroom plus the caches. It is checked before the first byte is
fetched, and only on a cold start — a volume that already holds a verified
model is allowed to be nearly full, because nothing large is about to be
written.

## The readiness marker

`state/model-ready.json` is written **after** a snapshot has been downloaded
*and* verified, never before. It records the model id, the resolved commit, the
snapshot path, and every file with its size.

```json
{
  "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
  "revision": "dcaee4d4dfc5ee71ad501f01f530e5652438fde0",
  "snapshot_path": "/runpod-volume/huggingface/hub/models--Qwen--...",
  "prepared_at": "2026-09-18T01:12:44+00:00",
  "total_bytes": 33500000000,
  "files": {"model-00001-of-00004.safetensors": 9999999999, "...": 0},
  "verified": true,
  "marker_version": 1
}
```

It is written to a sibling and renamed, so a Pod killed mid-write leaves either
the old marker or the new one — never a truncated file claiming a model is
ready when only half of it is.

Verification compares **sizes**, not hashes. Hashing 31 GB on every boot would
add minutes to a restart, while the failure that actually happens — a shard
truncated by a killed download — changes the size.

A marker that is corrupt, or written by a different version of the format, is
treated as absent. The snapshot is then re-verified, which is cheap compared
with refusing to boot.

## The download lock

`state/model-download.lock` is held with `flock` for the duration of a
download. Two Pods sharing one volume would otherwise race on the same blob
store, producing corrupt files and paying twice.

`flock` is advisory, and that is the point: the kernel releases it when the
holder dies. A Pod killed mid-download does not leave a lock that blocks every
future boot. The holder's hostname and pid are written inside the file for
diagnostics only — never trusted for correctness.

The second process re-checks the marker **after** acquiring the lock, because
while it waited the first one may have finished exactly the download it was
about to start.

## What is never done automatically

A corrupt cache is **reported**, not deleted. `clear_model_cache` exists and is
only ever called by an operator. Silently removing 31 GB because a check failed
is how a flaky disk turns into an hour of downtime and a surprise bill.

## Running without a volume

`WORKER_ALLOW_EPHEMERAL_STORAGE=1` accepts an unmounted root for throwaway
experiments. It is loud about what it costs: the model is re-downloaded on
every restart, and the container disk fills up. Never set it on a real
deployment.
