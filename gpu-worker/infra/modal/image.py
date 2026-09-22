"""The container image, and the line it refuses to cross.

    IMAGE  ≠  APPLICATION CODE

Everything below the `add_local_python_source` call changes a few times a year:
CUDA, PyTorch, vLLM, the kernels. Everything at that call changes several times
an hour. Keeping them apart is what turns "edit a file, wait twenty minutes for
a 24 GB image to build and push" into "edit a file, run `modal deploy`".

`copy=False` — the default — is what makes that true: the sources are attached
to the container at startup rather than baked into a layer, so a code change
invalidates no build cache at all.

## Why the upstream vLLM image rather than a base image of our own

Modal's own examples install vLLM into `nvidia/cuda` with `uv_pip_install`, and
that gives excellent layer caching. It also re-resolves the entire Python stack
at build time, which is precisely what `Dockerfile` refuses to do, in a comment
that explains why: the digest of a published, tested vLLM release is the one
thing making this worker reproducible. `vllm/vllm-openai:v0.28.0-cu129` is that
release, it is the base the RunPod image already used, and pulling it directly
means the Modal worker and the RunPod worker run identical bytes below our own
code — with no registry of ours in the path, and nothing to rebuild or push.
"""

from __future__ import annotations

import os
from typing import Final

import modal

from infra.modal.volumes import DATA_ROOT, cache_environment

__all__ = ["VALIDATED_MODEL_REVISION", "VLLM_EXTRA_ARGS", "VLLM_IMAGE", "worker_image"]

# The single place where the runtime is chosen. Never `latest`: a floating tag
# silently changes CUDA, PyTorch and the kernels under a fleet that is supposed
# to be reproducible. CUDA 12.9.1 works with NVIDIA drivers 535 through 570 and
# its TORCH_CUDA_ARCH_LIST covers 9.0, so Hopper H100/H200 are supported.
VLLM_IMAGE: Final = os.environ.get("MODAL_VLLM_IMAGE", "vllm/vllm-openai:v0.28.0-cu129")

# The commit this worker was validated against.
#
# `Dockerfile` deliberately does not bake this, on the grounds that pinning is
# an operator decision. On Modal the calculation is different: the weights are
# fetched by one process (`scripts/populate_modal_volume.py`) and consumed by
# another (the Server), and if the two disagree about the revision the Server
# refuses to start. Leaving it unset means both resolve `main` independently and
# a new upstream commit between them breaks the deployment. A default that can
# be overridden at deploy time is the safer of the two failures.
VALIDATED_MODEL_REVISION: Final = os.environ.get(
    "MODEL_REVISION", "dcaee4d4dfc5ee71ad501f01f530e5652438fde0"
)

# Extra vLLM flags, as a shell-quoted string parsed with `shlex` by
# `WorkerConfig`. One flag is here for a reason worth writing down.
#
# A Modal Volume is mounted over **9P**, and vLLM inspects the checkpoint's
# filesystem to decide whether to overlap reads with GPU transfers:
#
#     Filesystem type for checkpoints: 9P. Checkpoint size: 29.03 GiB.
#     Auto-prefetch is disabled because the filesystem (9P) is not a
#     recognized network FS (NFS/Lustre).
#
# It is a network filesystem; vLLM simply has no case for this one. With
# prefetch off, the four shards loaded serially at about 105 seconds each —
# roughly seven minutes of an H100 billed to read a disk, about $0.46 per cold
# start in pure I/O. Forcing the strategy is what that message asks for.
VLLM_EXTRA_ARGS: Final = os.environ.get(
    "VLLM_EXTRA_ARGS", "--safetensors-load-strategy=prefetch"
)


def _image_environment() -> dict[str, str]:
    env = {
        "PERSISTENT_ROOT": str(DATA_ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "VLLM_IMAGE_TAG": VLLM_IMAGE,
        "MODEL_REVISION": VALIDATED_MODEL_REVISION,
        "VLLM_EXTRA_ARGS": VLLM_EXTRA_ARGS,
        # 16384 was sized for a shared RunPod pod. An H100 holding 29 GB of FP8
        # weights has roughly 45 GB left, so the KV cache is not what is scarce
        # here — and the control plane reserves 4096 tokens for the answer.
        #
        # 32768 was the first attempt and the first real planner call failed
        # against it by exactly one token:
        #
        #     maximum context length is 32768 tokens. However, you requested
        #     4096 output tokens and your prompt contains at least 28673 input
        #     tokens, for a total of at least 32769 tokens.
        #
        # That is not a coincidence, it is the crude estimator: the context
        # budget packs files until `estimate_tokens` says it is full, and that
        # function divides characters by four. When the real tokenizer disagrees
        # upwards the prompt is already built, and the engine refuses it after
        # the GPU has been woken. Widening the window does not fix the estimate
        # — it buys the margin the estimate does not provide.
        "MAX_MODEL_LEN": os.environ.get("MAX_MODEL_LEN", "65536"),
        # Faster hub transfers. It only matters while the Volume is being
        # populated, on a CPU container, but the variable is read at import time
        # so it has to be in the image rather than set by the caller.
        "HF_XET_HIGH_PERFORMANCE": "1",
        # Exposes vLLM's /sleep and /wake_up routes. Without it they 404 and a
        # snapshot would capture a server holding 29 GB of device memory that
        # the restoring container cannot be assumed to reproduce.
        "VLLM_SERVER_DEV_MODE": "1",
        # Modal's own vLLM snapshot example sets this for snapshot
        # compatibility; parallel Inductor workers do not survive the restore.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    env.update(cache_environment())
    return env


worker_image = (
    modal.Image.from_registry(VLLM_IMAGE)
    # The base image's ENTRYPOINT is ["vllm", "serve"], which would swallow
    # Modal's own container entrypoint and try to serve a model named after it.
    # Clearing it is mandatory, not hygiene.
    .entrypoint([])
    .run_commands(
        # Modal requires `python` and `pip` on PATH. The vLLM image ships
        # `python3` and `pip3`, and on some tags only `python3`.
        "command -v python  >/dev/null || ln -s \"$(command -v python3)\" /usr/local/bin/python",
        "command -v pip     >/dev/null || ln -s \"$(command -v pip3)\"    /usr/local/bin/pip",
        # The worker package's only two dependencies, installed only if the base
        # image does not already satisfy them. `--no-deps` is not enough here:
        # the resolver must never be handed a chance to touch torch, vllm or
        # transformers, so it is never invoked when the imports already work.
        "python -c 'import httpx' 2>/dev/null "
        "|| pip install --no-cache-dir 'httpx>=0.28'",
        "python -c 'import huggingface_hub' 2>/dev/null "
        "|| pip install --no-cache-dir 'huggingface-hub>=0.35'",
        # The invariant, enforced by the build rather than promised in a README.
        # A weight file in a layer is pulled onto every cold container, for a
        # model that is supposed to live on the Volume.
        "found=$(find / -xdev \\( -name '*.safetensors' -o -name '*.gguf' -o -name '*.pt' \\) "
        "-size +64M -printf '%s\\t%p\\n' | head -5); "
        'if [ -n "$found" ]; then echo "model weights in a layer:" >&2; '
        'echo "$found" >&2; exit 1; fi',
    )
    .env(_image_environment())
    # Attached at container start, not copied into a layer. This is the line
    # that makes a code change cost seconds instead of a rebuild.
    .add_local_python_source("worker", "infra", copy=False)
)
