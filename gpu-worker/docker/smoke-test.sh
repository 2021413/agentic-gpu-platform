#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Thin wrapper around `worker-smoke-test` — one real completion, validated:
#
#     docker exec <pod> smoke-test.sh
#     docker exec <pod> smoke-test.sh --max-tokens 16 --timeout 60
#
# Costs a few GPU-seconds, so it is an explicit operator/CI action and never
# part of the boot path or the HEALTHCHECK.
#
# Exit codes (src/worker/cli.py): 0 passed, 2 configuration, 6 smoke failed.
# ---------------------------------------------------------------------------
set -Eeuo pipefail
on_error() {
    local code=$1 line=$2
    shift 2
    printf 'smoke-test.sh: failed at line %s: %s (exit %s)\n' "${line}" "$*" "${code}" >&2
    exit "${code}"
}
trap 'on_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR

exec worker-smoke-test "$@"
