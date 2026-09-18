#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Thin wrapper around `worker-preflight`, so an operator can inspect a running
# Pod without knowing where Python lives:
#
#     docker exec <pod> preflight.sh
#     docker exec <pod> preflight.sh --skip-storage
#
# It exists for one substantive reason beyond convenience: a `docker exec`
# shell inherits the IMAGE environment, not the one entrypoint.sh exported, so
# with a custom PERSISTENT_ROOT the baked cache variables would point at the
# wrong tree. They are re-derived here from PersistentLayout.environment().
#
# Exit codes (src/worker/cli.py): 0 ok, 2 configuration, 3 storage.
# ---------------------------------------------------------------------------
set -Eeuo pipefail
on_error() {
    local code=$1 line=$2
    shift 2
    printf 'preflight.sh: failed at line %s: %s (exit %s)\n' "${line}" "$*" "${code}" >&2
    exit "${code}"
}
trap 'on_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR

# key=value lines, exported one at a time. Never `eval`: a path is data.
while IFS='=' read -r key value; do
    [[ -n ${key} ]] || continue
    export "${key}=${value}"
done < <(python3 -c '
from worker.config import WorkerConfig
for key, value in WorkerConfig.from_env().layout.environment().items():
    print(f"{key}={value}")
')

exec worker-preflight "$@"
