# Changing the model

A runbook. Follow it in order; each step exists because skipping it produces a
failure that is annoying to diagnose from a Pod.

The short version: **the model is configuration, never code.** No image rebuild
is needed to serve a different model. What does need attention is the
arithmetic — three numbers must agree, and nothing checks them for you until a
Pod fails.

---

## 1. Measure the model you want

Never estimate this. The hub answers exactly:

```bash
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8   # the one you want

# Total download size, in GB
curl -s "https://huggingface.co/api/models/$MODEL/tree/main?recursive=true" \
  | python3 -c "import json,sys; f=json.load(sys.stdin); \
print(round(sum((x.get('size') or (x.get('lfs') or {}).get('size') or 0) for x in f)/1e9, 1), 'GB')"

# Architecture, quantisation and native context length
curl -sL "https://huggingface.co/$MODEL/raw/main/config.json" \
  | python3 -c "import json,sys; c=json.load(sys.stdin); \
print({k: c.get(k) for k in ('model_type','torch_dtype','max_position_embeddings')}); \
print('quant:', (c.get('quantization_config') or {}).get('quant_method', 'none (full precision)'))"

# The commit to pin
curl -s "https://huggingface.co/api/models/$MODEL" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"
```

Write down three things: **weight size in GB**, **quantisation**, **commit sha**.

> A model with `quant: none` is stored in bf16 and weighs roughly **two bytes per
> parameter**. A 30B model is then ~60 GB, not 31 — double everything below.

---

## 2. Check it fits the GPU

Weights are not the whole story: the KV cache grows with `MAX_MODEL_LEN` and
with concurrency, and vLLM reserves `GPU_MEMORY_UTILIZATION` of the card up
front.

| | |
|---|---|
| Rule of thumb | weights + 20-30% of the card for KV cache and activations |
| H100 80 GB | comfortable up to ~50 GB of weights |
| H200 141 GB | comfortable up to ~100 GB of weights |

**FP8 needs compute capability 8.9 or newer** — H100, H200, L40S, Ada, Blackwell.
On older cards vLLM dequantises and the worker becomes far slower than its
advertised capacity suggests. `worker-preflight` warns about this, and prints the
compute capability it found.

If it does not fit: lower `MAX_MODEL_LEN` first (the KV cache is what grows),
then `GPU_MEMORY_UTILIZATION` to 0.85, then use a bigger card or set
`TENSOR_PARALLEL_SIZE` to spread across several.

---

## 3. Size the storage — the part that actually bites

Three numbers must agree:

```text
MIN_FREE_DISK_GB  =  weight size + 10        (room to resume one shard)
volume size       =  weight size + 25        (weights + resume + vLLM cache + tmp)
container disk    =  50 GB                   (the image, unchanged by the model)
```

For the default model (31.2 GB): `MIN_FREE_DISK_GB=40`, volume 60 GB.
For a 60 GB bf16 model: `MIN_FREE_DISK_GB=70`, volume 85 GB.

Two traps, both of which cost a Pod boot:

* **`MIN_FREE_DISK_GB` must stay below what the volume can offer.** A 60 GB
  volume presents about 58 GB free; a floor of 60 fails every first boot on a
  volume that is perfectly adequate.
* **A volume grows but never shrinks.** Over-provisioning is permanent unless you
  delete and recreate it — and deleting loses the downloaded weights.

Storage is billed whether or not a Pod runs. At the high-performance rate
measured in EU-FR-1 (**$0.142/GB/month**), 60 GB is ~$8.50/month and 150 GB is
~$21/month. Standard-tier data centers are about half that.

---

## 4. Decide what happens to the old model

**Nothing happens to it automatically.** When `MODEL_ID` or `MODEL_REVISION`
changes, the worker logs

```text
marker describes acme/old-model@abc123def456, which is not what was requested;
preparing again
```

downloads the new one alongside, and **leaves the old weights on the volume**.
That is deliberate: silently deleting 31 GB because a variable changed is how a
typo becomes an hour of downtime. It also means a volume sized for one model
will fill up.

If both do not fit, either grow the volume first, or remove the old snapshot by
hand from `/runpod-volume/huggingface/hub/models--<org>--<name>/` before
switching.

Both models resident lets you roll back by changing one variable, with no
download. That is the only reason to size a volume for two.

---

## 5. Apply the change

Nothing is rebuilt. Set the variables on the Pod:

```bash
MODEL_ID=<org>/<name>
MODEL_REVISION=<the sha from step 1>     # pin it; see below
MIN_FREE_DISK_GB=<weights + 10>
MAX_MODEL_LEN=16384                      # lower it if VRAM is tight
SERVED_MODEL_NAME=<optional alias clients send as "model">
```

Or through the deployment tool, which takes the same values as flags:

```bash
python -m tools.runpod_deployer deploy \
  --model "$MODEL" \
  --model-revision "$SHA" \
  --max-model-len 16384 \
  --network-volume <volume id>
```

**Always pin `MODEL_REVISION` outside a test.** Unpinned, the worker resolves
`main`, so an upstream commit changes what a Pod serves on its next cold start —
and an already-prepared volume keeps serving the old one, so the drift appears
only on the machines that happen to start fresh. The marker records the
*resolved* commit either way, so you can always read what a volume actually
holds:

```bash
cat /runpod-volume/state/model-ready.json
```

---

## 6. Verify, before trusting it

```bash
worker-preflight      # sizes, GPU, compute capability, volume, model state
worker-ready          # blocks until /v1/models lists the model you asked for
worker-smoke-test     # one real completion, validated
```

`worker-ready` checks the **name**, not just an HTTP 200. That is what catches a
Pod that came up serving something else entirely — a stale `SERVED_MODEL_NAME`, a
wrong snapshot path — which would otherwise surface much later as a puzzling 404
from a client.

Expect a cold start of roughly 6 minutes for a 31 GB model on a high-performance
volume (measured: 345 s total, 272 s to serving). A warm restart on the same
volume skips the download entirely.

---

## What can go wrong, and how it looks

| Symptom | Cause |
|---|---|
| `only N GB free, M GB required` at boot | `MIN_FREE_DISK_GB` above what the volume offers, or a second model resident |
| `the volume ran out of space while downloading` | volume too small for the new weights; it is never retried, on purpose |
| `at least one GPU lacks hardware FP8` | FP8 model on a card older than compute 8.9 |
| vLLM exits during load, no useful message | weights do not fit; lower `MAX_MODEL_LEN`, then `GPU_MEMORY_UTILIZATION` |
| `not ready: serving (), expected '...'` | `SERVED_MODEL_NAME` does not match what clients send |
| Model changed but the Pod serves the old one | the volume still holds a verified marker for it; check `model-ready.json` |

Full list in [`troubleshooting.md`](troubleshooting.md).
