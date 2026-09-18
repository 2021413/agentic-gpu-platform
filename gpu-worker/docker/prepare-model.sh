#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Thin wrapper around `worker-prepare-model`:
#
#     docker exec <pod> prepare-model.sh                # download or resume
#     docker exec <pod> prepare-model.sh --check-only   # is it ready?
#
# Safe to run while the worker is serving: the download lock on the volume is
# what serialises it against the boot path.
#
# The cache variables are re-derived before the command runs, because
# huggingface_hub reads them at import time and a `docker exec` shell does not
# inherit what entrypoint.sh exported. Getting this wrong writes 31.2 GB to the
# container filesystem instead of the volume.
#
# Exit codes (src/worker/cli.py): 0 ok, 2 configuration, 3 storage, 4 model.
# ---------------------------------------------------------------------------
set -Eeuo pipefail
on_error() {
    local code=$1 line=$2
    shift 2
    printf 'prepare-model.sh: failed at line %s: %s (exit %s)\n' "${line}" "$*" "${code}" >&2
    exit "${code}"
}
trap 'on_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR

while IFS='=' read -r key value; do
    [[ -n ${key} ]] || continue
    export "${key}=${value}"
done < <(python3 -c '
from worker.config import WorkerConfig
for key, value in WorkerConfig.from_env().layout.environment().items():
    print(f"{key}={value}")
')

exec worker-prepare-model "$@"
