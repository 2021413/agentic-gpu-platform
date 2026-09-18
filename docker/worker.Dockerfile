# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# GPU worker image: vLLM inference server + worker-side agent (spec section 44).
#
# Build from the REPOSITORY ROOT:
#   docker build -f docker/worker.Dockerfile -t agentic-worker-gpu:dev .
#
# Run (requires the NVIDIA Container Toolkit on the host):
#   docker run --gpus all --env-file .env \
#     -v agentic-hf-cache:/models/huggingface agentic-worker-gpu:dev
#
# Design notes:
#   * MODEL WEIGHTS ARE NEVER BAKED IN. Nothing is downloaded at build time;
#     HF_HOME points at a mounted cache volume, so the image stays small and
#     the same image serves any model.
#   * The model is chosen at RUN time through MODEL_ID. No model name is
#     hard-coded anywhere; the value below is only a default for the initial
#     target model and is overridden by the environment.
#   * The worker agent is installed in its OWN virtualenv so that this
#     project's dependency pins can never disturb the ones vLLM was built with.
# ---------------------------------------------------------------------------

# Pin a concrete tag for reproducible builds, e.g. vllm/vllm-openai:v0.10.1.
# `latest` is the default only so that a fresh clone builds at all.
ARG VLLM_IMAGE=vllm/vllm-openai:latest

FROM ${VLLM_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Local inference server: bound to loopback, never exposed. The control
    # plane only ever talks to the worker agent (WORKER_ENDPOINT).
    VLLM_HOST=127.0.0.1 \
    VLLM_PORT=8001 \
    # Initial target model (spec section 3.3); override freely.
    MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct \
    MODEL_CONTEXT_LENGTH=131072 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.90 \
    WORKER_CONCURRENCY=4 \
    WORKER_PORT=9000 \
    LLM_PROVIDER=vllm \
    # Weights live in this mounted volume, outside the image.
    HF_HOME=/models/huggingface \
    PATH="/opt/worker-venv/bin:${PATH}"

# Isolated virtualenv for the worker agent, plus the model cache directory.
RUN set -eux; \
    if ! python3 -m venv /opt/worker-venv; then \
        apt-get update; \
        apt-get install --no-install-recommends -y python3-venv; \
        rm -rf /var/lib/apt/lists/*; \
        python3 -m venv /opt/worker-venv; \
    fi; \
    mkdir -p /models/huggingface

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml README.md ./
RUN /opt/worker-venv/bin/python -c "\
import pathlib, tomllib;\
pyproject = tomllib.loads(pathlib.Path('pyproject.toml').read_text());\
pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(pyproject['project']['dependencies']) + '\n')\
" \
 && /opt/worker-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./src
COPY prompts ./prompts
RUN /opt/worker-venv/bin/pip install --no-cache-dir --no-deps .

COPY --chmod=0755 docker/entrypoint-worker.sh /usr/local/bin/entrypoint-worker.sh

# Unprivileged account. It must own the model cache, otherwise the first
# download fails; a host bind-mount must be chown'd to 10001 as well.
RUN groupadd --gid 10001 worker 2>/dev/null || true; \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin worker 2>/dev/null || true; \
    chown -R 10001:10001 /models/huggingface /app
USER 10001:10001

# Worker agent port (control-plane callbacks / health). The vLLM port stays
# internal on purpose.
EXPOSE 9000

# Readiness = the local inference server answering. Loading a 30B model from a
# cold cache can take many minutes, hence the long start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=5 \
    CMD python3 -c "import os,urllib.request,sys;\
url='http://127.0.0.1:%s/health' % os.environ.get('VLLM_PORT','8001');\
sys.exit(0 if urllib.request.urlopen(url, timeout=8).status == 200 else 1)" || exit 1

ENTRYPOINT ["entrypoint-worker.sh"]
CMD []
