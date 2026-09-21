#!/usr/bin/env bash
# Run an agentic objective against a local project.
#
#   ./run-on.sh /path/to/project "what you want done"
#
# Takes the path as an argument rather than an environment variable, because a
# variable set in one shell invocation does not survive into the next — which
# silently wrote an empty mount path and failed four steps later.
set -Eeuo pipefail

API=${API:-http://localhost:8000}
PY=.venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

project_path="${1:-}"
objective="${2:-}"

die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

[ -n "$project_path" ] || die "usage: ./run-on.sh /path/to/project \"objective\""
[ -n "$objective" ]    || die "usage: ./run-on.sh /path/to/project \"objective\""
[ -d "$project_path" ] || die "$project_path is not a directory"
project_path="$(cd "$project_path" && pwd)"     # absolute, symlinks resolved

# 1. The orchestrator makes git worktrees, so the project must be a repository.
if ! git -C "$project_path" rev-parse --git-dir >/dev/null 2>&1; then
    echo "==> $project_path is not a git repository; initialising one"
    git -C "$project_path" init -b main -q
    git -C "$project_path" add -A
    git -C "$project_path" commit -qm "initial state" || true
fi

# Uncommitted work would block the controlled merge of an accepted patch.
if [ -n "$(git -C "$project_path" status --porcelain)" ]; then
    die "$project_path has uncommitted changes. Commit or stash them first: the
       accepted patch is merged back into this repository, and integrating onto
       a dirty tree is refused."
fi

# 2. Make it visible inside the api container.
echo "==> mounting $project_path at /projects/current"
if grep -q '^LOCAL_PROJECT_PATH=' .env; then
    sed -i "s|^LOCAL_PROJECT_PATH=.*|LOCAL_PROJECT_PATH=$project_path|" .env
else
    printf 'LOCAL_PROJECT_PATH=%s\n' "$project_path" >> .env
fi
docker compose up -d api >/dev/null
for _ in $(seq 1 30); do
    curl -fsS -m2 "$API/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -fsS -m5 "$API/health" >/dev/null || die "the control plane did not come back up"

# 3. Toolchain. Guessed from what is in the repository, never silently: a wrong
#    build command reported as a code defect would poison the repair loop.
name="$(basename "$project_path")"
if [ -f "$project_path/package.json" ]; then
    lang=javascript; build='npm run build --if-present'; test='npm test'
elif [ -f "$project_path/pyproject.toml" ] || [ -f "$project_path/setup.py" ]; then
    lang=python; build='python -m compileall -q .'; test='pytest -q'
elif [ -f "$project_path/Cargo.toml" ]; then
    lang=rust; build='cargo build'; test='cargo test'
elif [ -f "$project_path/go.mod" ]; then
    lang=go; build='go build ./...'; test='go test ./...'
else
    lang=unknown; build=''; test=''
    echo "==> WARNING: no recognised project file. Build and test are left unset,"
    echo "    so validation will be recorded as SKIPPED, never as passed."
fi
echo "==> toolchain: $lang | build: ${build:-none} | test: ${test:-none}"

json_or_null() { [ -n "$1" ] && printf '"%s"' "$1" || printf 'null'; }

pid=$(curl -fsS -X POST "$API/v1/projects" -H 'content-type: application/json' \
  -d "{\"name\":\"$name\",\"local_path\":\"/projects/current\",\"default_branch\":\"main\",
       \"toolchain\":{\"language\":\"$lang\",
                      \"build_command\":$(json_or_null "$build"),
                      \"test_command\":$(json_or_null "$test")}}" \
  | $PY -c 'import json,sys; print(json.load(sys.stdin)["id"])') \
  || die "could not create the project; is the control plane running? (make docker-up)"

rid=$(curl -fsS -X POST "$API/v1/projects/$pid/runs" -H 'content-type: application/json' \
  -d "$($PY -c 'import json,sys; print(json.dumps({"objective": sys.argv[1], "candidate_count": int(sys.argv[2])}))' "$objective" "${CANDIDATES:-1}")" \
  | $PY -c 'import json,sys; print(json.load(sys.stdin)["id"])') \
  || die "could not create the run"

echo
echo "project  $pid"
echo "run      $rid"
echo "patch    curl -s $API/v1/runs/$rid/candidates"
echo
echo "==> following events (ctrl-c to stop; the run keeps going)"
curl -N "$API/v1/runs/$rid/events"
