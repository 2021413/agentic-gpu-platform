# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Agentic GPU worker — a thin, reproducible layer on top of the official vLLM
# runtime image.
#
# The architectural invariant, restated here because every line below is bound
# by it:
#
#     immutable image + ephemeral GPU + persistent external volume + runtime config
#
# * No model weight is ever COPYed, ADDed or downloaded during the build. The
#   31.2 GB of Qwen3-Coder FP8 lives on the RunPod network volume, never in a
#   layer. A layer holding the weights would have to be pulled on every cold
#   Pod, would blow past registry limits, and would make a revision bump a full
#   image rebuild.
# * CUDA, PyTorch, vLLM, transformers and their transitive pins come from the
#   base image and are NOT reinstalled. Reinstalling any of them would resolve
#   fresh versions at build time and destroy the one property that makes this
#   image reproducible: the digest of a published, tested vLLM release.
# ---------------------------------------------------------------------------

# The single place where the runtime is chosen. Never `latest`: a floating tag
# silently changes CUDA, PyTorch and the kernels under a fleet that is supposed
# to be reproducible.
#
#   vllm/vllm-openai:v0.28.0-cu129  (default)
#       CUDA 12.9.1, works with NVIDIA drivers 535 through 570.
#       TORCH_CUDA_ARCH_LIST contains 9.0, so Hopper H100/H200 are covered.
#       Chosen because the RunPod H100/H200 fleet is still widely on 5xx
#       drivers below 575, where the CUDA 13 build refuses to start.
#
#   vllm/vllm-openai:v0.28.0        (documented alternative)
#       CUDA 13.0.2, requires drivers 535 through 575. Use it only on hosts
#       known to run a 575+ driver, e.g. Blackwell capacity.
#
# Override at build time:  docker build --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.28.0 .
ARG VLLM_IMAGE=vllm/vllm-openai:v0.28.0-cu129

FROM ${VLLM_IMAGE}

# ---------------------------------------------------------------------------
# Build metadata. Declared as ARGs so `docker inspect` can show exactly which
# runtime, which source revision and which default model an image carries —
# without running it, and without trusting a tag that someone may have moved.
# ---------------------------------------------------------------------------
ARG VLLM_IMAGE
ARG VLLM_VERSION=0.28.0
ARG CUDA_VERSION=12.9.1
ARG WORKER_VERSION=1.0.0
ARG GIT_REVISION=unknown
ARG BUILD_DATE=unknown
ARG DEFAULT_MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
# The commit this image was validated against. Deliberately NOT exported as
# MODEL_REVISION: pinning is an operator decision (see .env.example), and a
# baked pin would silently override an unpinned deployment.
ARG VALIDATED_MODEL_REVISION=dcaee4d4dfc5ee71ad501f01f530e5652438fde0

LABEL org.opencontainers.image.title="agentic-gpu-worker" \
      org.opencontainers.image.description="RunPod GPU worker serving Qwen3-Coder-30B-A3B-Instruct-FP8 through vLLM" \
      org.opencontainers.image.version="${WORKER_VERSION}" \
      org.opencontainers.image.revision="${GIT_REVISION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="${VLLM_IMAGE}" \
      org.opencontainers.image.licenses="Proprietary" \
      ai.agentic.worker.vllm-version="${VLLM_VERSION}" \
      ai.agentic.worker.cuda-version="${CUDA_VERSION}" \
      ai.agentic.worker.base-image="${VLLM_IMAGE}" \
      ai.agentic.worker.default-model="${DEFAULT_MODEL_ID}" \
      ai.agentic.worker.validated-model-revision="${VALIDATED_MODEL_REVISION}" \
      ai.agentic.worker.model-weights-baked="false" \
      ai.agentic.worker.persistent-root="/runpod-volume"

# ---------------------------------------------------------------------------
# Cache redirection.
#
# These MUST be in the image ENV and not merely exported by entrypoint.sh:
# `huggingface_hub` and `torch` read them at IMPORT time, so any Python process
# started by an operator (`docker exec ... python -c "import torch"`), by the
# HEALTHCHECK, or by vLLM itself would otherwise write gigabytes of compiled
# kernels and hub blobs into the container filesystem — the exact failure this
# image exists to prevent.
#
# The values are PersistentLayout.environment() evaluated at the default root
# (src/worker/config.py). If PERSISTENT_ROOT is changed at runtime,
# entrypoint.sh re-derives all of them from the same Python source of truth
# before anything heavy is imported.
# ---------------------------------------------------------------------------
ENV PERSISTENT_ROOT=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    VLLM_CACHE_ROOT=/runpod-volume/vllm \
    TORCH_HOME=/runpod-volume/torch \
    TMPDIR=/runpod-volume/tmp \
    XDG_CACHE_HOME=/runpod-volume/xdg \
    TRITON_CACHE_DIR=/runpod-volume/vllm/triton

# Serving defaults, identical to the ones in src/worker/config.py. Repeated
# here only so that `docker inspect` answers "what does this image serve and
# where does it listen" without starting it.
ENV MODEL_ID="${DEFAULT_MODEL_ID}" \
    HOST=0.0.0.0 \
    PORT=8000 \
    VLLM_IMAGE_TAG="${VLLM_IMAGE}" \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# ---------------------------------------------------------------------------
# The worker package. `--no-deps` is load-bearing: the dependency solver must
# never be given a chance to touch torch, vllm or transformers. The two real
# dependencies are installed only if the base image does not already satisfy
# them (it ships huggingface-hub, and httpx via the openai client).
# ---------------------------------------------------------------------------
COPY pyproject.toml README.md /opt/worker/
COPY src /opt/worker/src

RUN set -eux; \
    command -v bash >/dev/null; \
    missing=''; \
    python3 -c 'import httpx' 2>/dev/null || missing="${missing} httpx>=0.28"; \
    python3 -c 'import huggingface_hub' 2>/dev/null || missing="${missing} huggingface-hub>=0.35"; \
    echo "dependencies missing from the base image:${missing:- none}"; \
    if [ -n "${missing}" ]; then pip install --no-cache-dir ${missing}; fi; \
    pip install --no-cache-dir --no-deps /opt/worker; \
    rm -rf /root/.cache/pip; \
    python3 -c 'import worker, httpx, huggingface_hub; print("worker", worker.__version__)'; \
    for command in worker-preflight worker-prepare-model worker-serve-args \
                   worker-ready worker-health worker-smoke-test; do \
        command -v "${command}"; \
    done

# The invariant, enforced by the build itself rather than by a promise in a
# README: fail loudly if any model weight ended up in a layer, and make sure
# the mount point was not accidentally materialised (an existing empty
# /runpod-volume turns "the volume failed to attach" into a silent 31 GB
# download onto the ephemeral container disk).
RUN set -eux; \
    found="$(find / -xdev \( -name '*.safetensors' -o -name '*.gguf' \) -print -quit)"; \
    if [ -n "${found}" ]; then echo "model weights in a layer: ${found}" >&2; exit 1; fi; \
    if [ -e /runpod-volume ]; then echo "/runpod-volume must not exist in the image" >&2; exit 1; fi

# The lifecycle scripts. Installed on PATH so an operator can run any single
# stage by hand: `docker exec <pod> smoke-test.sh`. The pip console scripts
# (worker-*) are also on PATH and remain the lower-level interface; these
# wrappers add exit-code translation and re-export the persistent cache
# environment, which a `docker exec` shell does not inherit from entrypoint.sh.
COPY --chmod=0755 docker/entrypoint.sh \
                  docker/preflight.sh \
                  docker/prepare-model.sh \
                  docker/health.sh \
                  docker/ready.sh \
                  docker/smoke-test.sh \
                  /usr/local/bin/

# Inherited from the base image; stated explicitly so a base change cannot move
# it silently. Deliberately NOT /runpod-volume: creating that directory here
# would turn "the volume failed to attach" into "an empty directory exists",
# which is precisely the mistake that sends 31 GB to an ephemeral disk.
WORKDIR /vllm-workspace

EXPOSE 8000

# Liveness, never readiness. `worker-health` probes /health, which answers as
# soon as the process is up; `worker-ready` (which also checks that the right
# model is actually served) is what an orchestrator should gate traffic on.
#
# --start-period is 45 minutes because a COLD volume must download 31.2 GB over
# four shards before vLLM even starts loading, and the load itself — FP8 MoE
# weights, CUDA graph capture — adds several minutes on top. During that window
# the process is alive and healthy but answers nothing, and a supervisor that
# restarts it would throw away the partial download and start the same 31.2 GB
# again, forever. A WARM restart passes this check in seconds, so the long
# start-period costs nothing in steady state.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45m --retries=3 \
    CMD ["worker-health"]

# The base image's ENTRYPOINT is ["vllm","serve"], which would start a server
# before the volume has been validated and before the model exists locally.
# Overridden unconditionally. No CMD: the argument vector is produced by
# `worker-serve-args` from the environment, and a CMD here would be a second,
# silently diverging source of truth for it.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
