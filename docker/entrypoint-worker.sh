#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GPU worker entry point: supervises the vLLM server and the worker agent.
#
#   vLLM server (system python, GPU)  -- loopback only, OpenAI-compatible
#   worker agent (/opt/worker-venv)   -- register / heartbeat / drain / jobs
#
# If either process exits, the container exits: a worker that lost its
# inference server must disappear from the registry rather than keep
# advertising capacity it no longer has. The control plane then reclaims its
# leases (see docs/worker-lifecycle.md).
#
# Set VLLM_ENABLED=0 to run the agent against an inference server hosted
# elsewhere (INFERENCE_BASE_URL); nothing else changes.
# ---------------------------------------------------------------------------
set -euo pipefail

VLLM_ENABLED="${VLLM_ENABLED:-1}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-1800}"

log() { printf '[entrypoint-worker] %s\n' "$*" >&2; }

# A command passed to `docker run` / compose `command:` wins over everything.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

pids=()

terminate() {
    log "shutting down"
    for pid in "${pids[@]:-}"; do
        [ -n "${pid}" ] && kill -TERM "${pid}" 2>/dev/null || true
    done
    wait || true
    exit 0
}
trap terminate TERM INT

start_vllm() {
    if [ -z "${MODEL_ID:-}" ]; then
        log "FATAL: MODEL_ID is empty. The model is configuration, not code:"
        log "       set MODEL_ID (e.g. Qwen/Qwen3-Coder-30B-A3B-Instruct)."
        exit 2
    fi

    log "starting vLLM for model '${MODEL_ID}' on ${VLLM_HOST}:${VLLM_PORT}"
    log "HF_HOME=${HF_HOME:-<unset>} (weights are read from the mounted cache)"

    # VLLM_EXTRA_ARGS is intentionally word-split: it carries whole flags,
    # e.g. "--enable-auto-tool-choice --tool-call-parser hermes".
    # shellcheck disable=SC2086
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_ID}" \
        --served-model-name "${MODEL_ID}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --max-model-len "${MODEL_CONTEXT_LENGTH:-131072}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
        ${VLLM_EXTRA_ARGS:-} &
    pids+=("$!")
}

wait_for_vllm() {
    local deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
    log "waiting for vLLM readiness (up to ${VLLM_STARTUP_TIMEOUT_SECONDS}s; a cold model cache is slow)"
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if python3 -c "import sys,urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/health', timeout=5)" 2>/dev/null; then
            log "vLLM is ready"
            return 0
        fi
        # Fail fast if the server died instead of waiting for the full timeout.
        if ! kill -0 "${pids[0]}" 2>/dev/null; then
            log "FATAL: vLLM exited during startup; see the log above."
            exit 1
        fi
        sleep 5
    done
    log "FATAL: vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT_SECONDS}s."
    exit 1
}

start_agent() {
    if ! python -c "import worker_agent.main" >/dev/null 2>&1; then
        cat >&2 <<'MSG'
[entrypoint-worker] FATAL: `worker_agent.main` cannot be imported.

The worker-side agent (src/worker_agent/, console script `agentic-worker`)
is not part of this image yet. The inference server above is fine; only the
registration/heartbeat process is missing. Until it lands, run this image
with an explicit command, e.g.:

    docker run --rm --gpus all agentic-worker-gpu:dev \
        python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_ID"
MSG
        exit 3
    fi

    log "registering with control plane ${CONTROL_PLANE_URL:-<unset>} as ${WORKER_ID:-<generated>}"
    agentic-worker &
    pids+=("$!")
}

if [ "${VLLM_ENABLED}" = "1" ]; then
    start_vllm
    wait_for_vllm
    export INFERENCE_BASE_URL="${INFERENCE_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
else
    log "VLLM_ENABLED=0: using external inference at ${INFERENCE_BASE_URL:-<unset>}"
fi

start_agent

# Exit as soon as any supervised process stops.
wait -n
log "a supervised process exited; stopping the container"
terminate
