#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Worker lifecycle.
#
#   BOOT → PREFLIGHT → VOLUME → MODEL PREPARATION → MODEL VALIDATION
#        → vLLM START → (optional) READINESS WATCH → READY
#
# Every decision that needs logic lives in Python (src/worker/), where it is
# tested without a GPU. This script only sequences those commands, translates
# their documented exit codes into something an operator can act on, and
# finally hands the process over to vLLM.
#
# Exit codes, contracted with src/worker/cli.py:
#   0 success   2 configuration   3 storage   4 model   5 not ready   6 smoke
# ---------------------------------------------------------------------------
set -Eeuo pipefail

readonly EXIT_CONFIG=2
readonly EXIT_STORAGE=3
readonly EXIT_MODEL=4
readonly EXIT_NOT_READY=5
readonly EXIT_SMOKE=6

log() { printf '%s  %-5s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" "${*:2}"; }
info() { log INFO "$@"; }
warn() { log WARN "$@" >&2; }
fail() { log ERROR "$@" >&2; }

stage() {
    printf '\n'
    log STAGE "══ $* ══"
}

# Names the line and the command, because "exit 1" on a boot script is not a
# diagnosis. The status, the line and the command are passed as arguments so
# they are captured at the moment of failure rather than read later.
on_error() {
    local code=$1 line=$2
    shift 2
    fail "entrypoint.sh failed at line ${line}: $* (exit ${code})"
    exit "${code}"
}
trap 'on_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR

# ---------------------------------------------------------------------------
# Exit-code translation. The Python layer already printed the technical detail;
# this adds the operator-facing "so what do I do about it".
# ---------------------------------------------------------------------------
explain() {
    case "$1" in
        "${EXIT_CONFIG}")
            fail "configuration error — an environment variable is wrong or missing."
            fail "  Retrying will not help. Fix the Pod template / .env and redeploy."
            fail "  The named variable is printed above; .env.example documents every one."
            ;;
        "${EXIT_STORAGE}")
            fail "persistent storage error — the volume is missing, read-only or full."
            fail "  Check that a RunPod network volume is attached at \${PERSISTENT_ROOT}"
            fail "  (currently '${PERSISTENT_ROOT:-/runpod-volume}') and that it has room"
            fail "  for the weights. Do NOT work around this by running without a volume:"
            fail "  the model would be re-downloaded on every Pod start."
            ;;
        "${EXIT_MODEL}")
            fail "model preparation failed — the weights are not usable on this volume."
            fail "  Most common causes: no outbound network, a gated repository without a"
            fail "  valid HF_TOKEN, a wrong MODEL_ID/MODEL_REVISION, or a volume that filled"
            fail "  mid-download. Partial downloads are kept on purpose and resume on the"
            fail "  next start; deleting them is an explicit operator action."
            ;;
        "${EXIT_NOT_READY}")
            fail "not ready within the deadline — vLLM did not serve the expected model."
            fail "  Raise READINESS_TIMEOUT_SECONDS for a very large model, or read the vLLM"
            fail "  log above: an OOM at load time and a wrong --served-model-name both land"
            fail "  here."
            ;;
        "${EXIT_SMOKE}")
            fail "smoke test failed — the server answered, but not with a usable completion."
            ;;
        *)
            fail "unexpected failure (exit $1)."
            ;;
    esac
}

# Runs a worker command, and on failure explains the code and propagates it
# unchanged. The code is the contract; remapping it would break every caller.
run_stage() {
    local rc=0
    "$@" || rc=$?
    if ((rc != 0)); then
        printf '\n' >&2
        fail "'$*' exited with ${rc}"
        explain "${rc}"
        exit "${rc}"
    fi
}

# ---------------------------------------------------------------------------
# Escape hatch: any argument means "run this instead of the lifecycle".
# There is no CMD, so arguments only ever come from an operator typing them
# (`docker run <image> bash`, `docker run <image> worker-preflight`).
# ---------------------------------------------------------------------------
if (($# > 0)); then
    info "arguments given: bypassing the lifecycle and exec'ing: $*"
    exec "$@"
fi

# ---------------------------------------------------------------------------
stage "BOOT"
# ---------------------------------------------------------------------------
info "host             $(hostname)"
info "base image       ${VLLM_IMAGE_TAG:-unknown}"
info "worker version   $(python3 -c 'import worker; print(worker.__version__)' 2>/dev/null || echo unknown)"
info "persistent root  ${PERSISTENT_ROOT:-/runpod-volume}"
info "pid              $$ (this shell is replaced by vLLM at the end)"

case "${WORKER_ALLOW_EPHEMERAL_STORAGE:-0}" in
    1 | true | yes | on)
        warn "WORKER_ALLOW_EPHEMERAL_STORAGE is set: a missing volume will NOT stop"
        warn "this worker. The weights would then live on the container filesystem"
        warn "and be downloaded again on every Pod start. Intended for throwaway runs."
        ;;
esac

# ---------------------------------------------------------------------------
stage "PREFLIGHT"
# ---------------------------------------------------------------------------
# Reports configuration, installed versions, GPUs and storage, and refuses to
# continue on anything that cannot work. This is also what creates the
# directory tree on the volume.
run_stage worker-preflight

# ---------------------------------------------------------------------------
stage "VOLUME"
# ---------------------------------------------------------------------------
# The cache variables are already in the image ENV for the default root, and
# huggingface_hub and torch read them at import time. If PERSISTENT_ROOT was
# overridden, those baked values now point at the wrong place, so they are
# re-derived from the single source of truth — PersistentLayout.environment()
# — before anything heavy is imported.
#
# Read as key=value lines and exported one by one: never `eval`, so a path
# containing a shell metacharacter stays a path.
layout_env="$(python3 -c '
from worker.config import WorkerConfig
for key, value in WorkerConfig.from_env().layout.environment().items():
    print(f"{key}={value}")
')" || {
    fail "could not derive the persistent layout from the environment"
    explain "${EXIT_CONFIG}"
    exit "${EXIT_CONFIG}"
}

while IFS='=' read -r key value; do
    [[ -n ${key} ]] || continue
    if [[ ${!key-} != "${value}" ]]; then
        info "re-exporting ${key}=${value} (image default was '${!key-unset}')"
    fi
    export "${key}=${value}"
done <<<"${layout_env}"

for key in HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE VLLM_CACHE_ROOT TORCH_HOME \
           TMPDIR XDG_CACHE_HOME TRITON_CACHE_DIR; do
    directory="${!key}"
    # `ensure_layout` created the tree, but TRITON_CACHE_DIR is a subdirectory
    # of it that Triton would create on first use. Creating it here instead
    # means a permission or quota problem surfaces now, in a stage that can
    # explain it, rather than as a kernel-compilation failure minutes later.
    if ! mkdir -p "${directory}" 2>/dev/null; then
        fail "${key}=${directory} could not be created"
        explain "${EXIT_STORAGE}"
        exit "${EXIT_STORAGE}"
    fi
    if [[ ! -w ${directory} ]]; then
        fail "${key}=${directory} is not writable"
        explain "${EXIT_STORAGE}"
        exit "${EXIT_STORAGE}"
    fi
done
info "caches point at the volume: HF_HUB_CACHE=${HF_HUB_CACHE}"

# ---------------------------------------------------------------------------
stage "MODEL PREPARATION"
# ---------------------------------------------------------------------------
skip_preparation=0
case "${WORKER_SKIP_MODEL_PREPARATION:-0}" in
    1 | true | yes | on) skip_preparation=1 ;;
esac

if ((skip_preparation)); then
    warn "WORKER_SKIP_MODEL_PREPARATION is set: the model will NOT be downloaded"
    warn "or verified. This exists for offline bootstrap tests of the image; a"
    warn "real Pod started this way will make vLLM resolve the repository itself,"
    warn "outside the lock and outside the volume accounting."
else
    # Idempotent by design: a volume that already holds a verified snapshot
    # returns in seconds, which is the whole point of the persistent volume.
    run_stage worker-prepare-model
fi

# ---------------------------------------------------------------------------
stage "MODEL VALIDATION"
# ---------------------------------------------------------------------------
# Cheap re-check of the marker: proves that what is on the volume is what this
# worker was asked to serve, and that the next stage will point vLLM at a local
# snapshot rather than at the hub.
validation_rc=0
worker-prepare-model --check-only || validation_rc=$?
if ((validation_rc != 0)); then
    if ((skip_preparation)); then
        warn "no verified snapshot on the volume (expected: preparation was skipped)"
    else
        fail "the model reported as prepared did not validate"
        explain "${EXIT_MODEL}"
        exit "${EXIT_MODEL}"
    fi
fi

# ---------------------------------------------------------------------------
stage "vLLM START"
# ---------------------------------------------------------------------------
# A NUL-separated vector, read into a real array. Never a string: an argument
# vector assembled by concatenation and re-split by the shell is exactly how a
# value in MODEL_ID or VLLM_EXTRA_ARGS becomes a command. `worker-serve-args`
# also resolves the local snapshot path from the marker, so vLLM never needs the
# network on a warm start.
#
# NUL and not newline, and not command substitution either:
#   * `shlex` can produce a word that CONTAINS a newline — a quoted
#     VLLM_EXTRA_ARGS is enough — and a line-oriented read would split it in
#     two, exec'ing vLLM with arguments nobody wrote. NUL is the one byte an
#     argv word cannot contain.
#   * `$(...)` silently drops NUL bytes, so the vector goes through a file.
serve_args_file="$(mktemp)"
trap 'rm -f "${serve_args_file}"' EXIT
serve_args_rc=0
worker-serve-args > "${serve_args_file}" || serve_args_rc=$?
if ((serve_args_rc != 0)); then
    fail "worker-serve-args exited with ${serve_args_rc}"
    explain "${serve_args_rc}"
    exit "${serve_args_rc}"
fi

mapfile -d '' -t vllm_argv < "${serve_args_file}"
if ((${#vllm_argv[@]} < 2)) || [[ -z ${vllm_argv[0]} ]]; then
    fail "worker-serve-args produced no argument vector"
    explain "${EXIT_CONFIG}"
    exit "${EXIT_CONFIG}"
fi

info "vLLM command (secrets redacted):"
while IFS= read -r word; do
    info "    ${word}"
done < <(worker-serve-args --redacted --lines)

# ---------------------------------------------------------------------------
# Optional readiness watch.
#
# Honest description of what this does, because `exec` makes it subtle:
#
#   The watcher is started as a background CHILD of this shell. `exec` then
#   replaces this shell's process IMAGE while keeping its PID, so vLLM becomes
#   PID 1 and the watcher survives — it is not killed, and it keeps the stdout
#   and stderr it inherited, so its lines appear in `docker logs` alongside
#   vLLM's. Its parent is now vLLM itself. Two consequences an operator should
#   know: (1) when the watcher exits, vLLM (as PID 1) does not reap arbitrary
#   children, so it lingers as one harmless zombie entry until the container
#   stops; (2) its exit status is observable only in the log — it CANNOT fail
#   the container, by design. This stage is observability, not a gate.
#
#   The real readiness gate belongs outside: `worker-ready` run from the
#   orchestrator, or `docker exec <pod> ready.sh`.
# ---------------------------------------------------------------------------
case "${WORKER_WAIT_READY:-0}" in
    1 | true | yes | on)
        info "WORKER_WAIT_READY is set: watching readiness in the background"
        (
            if worker-ready; then
                log READY "the worker is serving requests"
            else
                log ERROR "readiness watch gave up (exit ${EXIT_NOT_READY} semantics);" \
                    "vLLM is still running and was NOT stopped by this watcher"
            fi
        ) &
        info "readiness watcher started as pid $!"
        ;;
esac

# ---------------------------------------------------------------------------
stage "EXEC"
# ---------------------------------------------------------------------------
# `exec` is what makes signal handling correct, and it is not a micro-
# optimisation. It replaces this shell with vLLM at the SAME pid, so vLLM is
# PID 1: `docker stop`, a Kubernetes/RunPod drain and a `kill` from a
# supervisor all deliver SIGTERM directly to the process that knows how to
# finish in-flight requests and tear down NCCL and the shared-memory segments.
#
# Without `exec`, bash would stay PID 1 with vLLM as its child. A non-
# interactive bash does not forward signals to a foreground child and would not
# even run a trap until that child exited — so SIGTERM would be absorbed, the
# runtime would wait out its grace period and then SIGKILL the whole container,
# dropping every in-flight request and leaving /dev/shm segments behind.
# Trapping and forwarding by hand can be made to work; not needing to is
# better, which is why nothing is started after this line.
info "handing over to vLLM (pid $$ becomes PID 1 of the container)"
exec "${vllm_argv[@]}"
