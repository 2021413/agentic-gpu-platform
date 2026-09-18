#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Thin wrapper around `worker-ready` — READINESS, the question an orchestrator
# should gate traffic on:
#
#     docker exec <pod> ready.sh
#     docker exec <pod> ready.sh --timeout 300 --quiet
#
# Blocks until the server actually lists the model it was asked to serve, so it
# also catches the case where vLLM came up serving something else.
#
# Exit codes (src/worker/cli.py): 0 ready, 2 configuration, 5 not ready in time.
# ---------------------------------------------------------------------------
set -Eeuo pipefail
on_error() {
    local code=$1 line=$2
    shift 2
    printf 'ready.sh: failed at line %s: %s (exit %s)\n' "${line}" "$*" "${code}" >&2
    exit "${code}"
}
trap 'on_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR

exec worker-ready "$@"
