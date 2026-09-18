#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Thin wrapper around `worker-health` — LIVENESS only:
#
#     docker exec <pod> health.sh
#
# "Alive" means something answers on the port. During a multi-minute model load
# the process is alive and must not be restarted, yet it cannot serve; use
# ready.sh for that question.
#
# No cache variables are exported here on purpose: this path must stay cheap,
# and it only ever speaks HTTP to the local server.
#
# Exit codes (src/worker/cli.py): 0 alive, 2 configuration, 5 not answering.
# ---------------------------------------------------------------------------
set -Eeuo pipefail
on_error() {
    local code=$1 line=$2
    shift 2
    printf 'health.sh: failed at line %s: %s (exit %s)\n' "${line}" "$*" "${code}" >&2
    exit "${code}"
}
trap 'on_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR

exec worker-health "$@"
