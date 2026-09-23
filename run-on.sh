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

# 2. Send the project to the control plane. It used to be bind-mounted at one
#    fixed path that every project record pointed to, so picking any project
#    ran the agents on whatever was mounted. Now the files are uploaded and the
#    server keeps them, one directory per project, at a path nobody chooses.
#    The archive is built from `git ls-files`, so ignored artefacts, virtualenvs
#    and build output never leave the machine.
archive="$(mktemp --suffix=.zip)"
trap 'rm -f "$archive"' EXIT
( cd "$project_path" && git ls-files -z | xargs -0 zip -q "$archive" ) \
    || die "could not archive $project_path (is zip installed?)"
echo "==> archived $(git -C "$project_path" ls-files | wc -l) tracked file(s), $(du -h "$archive" | cut -f1)"
for _ in $(seq 1 30); do
    curl -fsS -m2 "$API/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -fsS -m5 "$API/health" >/dev/null || die "the control plane is not up (make docker-up)"

# 3. Toolchain. Guessed from what is in the repository, never silently: a wrong
#    build command reported as a code defect would poison the repair loop.
name="${NAME:-$(basename "$project_path")}"

# Does the Makefile actually define that target? A build command that fails
# because the target does not exist marks every candidate non-viable for a
# reason that has nothing to do with the code.
has_make_target() {
    [ -f "$project_path/Makefile" ] || return 1
    grep -qE "^$1[[:space:]]*:" "$project_path/Makefile"
}

if [ -f "$project_path/package.json" ]; then
    lang=javascript; build='npm run build --if-present'; test='npm test'
elif [ -f "$project_path/pyproject.toml" ] || [ -f "$project_path/setup.py" ]; then
    lang=python; build='python -m compileall -q .'; test='pytest -q'
elif [ -f "$project_path/Cargo.toml" ]; then
    lang=rust; build='cargo build'; test='cargo test'
elif [ -f "$project_path/go.mod" ]; then
    lang=go; build='go build ./...'; test='go test ./...'
elif [ -f "$project_path/CMakeLists.txt" ]; then
    lang=cpp; build='cmake --build build --parallel'; test='ctest --test-dir build --output-on-failure'
elif [ -f "$project_path/Makefile" ]; then
    # C or C++ by Makefile. The build target is whatever `make` does by
    # default; the test target only exists if the project wrote one.
    if ls "$project_path"/**/*.cpp "$project_path"/*.cpp >/dev/null 2>&1; then lang=cpp; else lang=c; fi
    build='make -j4'
    if   has_make_target test;  then test='make test'
    elif has_make_target check; then test='make check'
    else
        test=''
        echo "==> note: the Makefile defines no 'test' or 'check' target, so the"
        echo "    test command is left unset. Validation records SKIPPED rather"
        echo "    than inventing a target that would fail on every candidate."
    fi
else
    lang=unknown; build=''; test=''
    echo "==> WARNING: no recognised project file. Build and test are left unset,"
    echo "    so validation will be recorded as SKIPPED, never as passed."
fi
echo "==> toolchain: $lang | build: ${BUILD-${build:-none}} | test: ${TEST-${test:-none}}"

# Explicit overrides win over detection, for the case where you know better.
build=${BUILD-$build}
test=${TEST-$test}

# ... and only then ask whether they can run. Checking before the override was
# a bug of my own making: it refused a run over a command the caller had
# already replaced.
# Do the commands we just chose actually exist where the tools run?
#
# Guessing this from the language was wrong twice over: it missed that the
# runtime image has no compiler for a C project, and it waved through a Python
# project whose `pytest` is equally absent. So ask, rather than assume — a
# command that cannot start is reported as a tool failure and kills the run
# with a reason that says nothing about the code.
#
# The api container is the right thing to probe, despite TOOL_SANDBOX_IMAGE
# suggesting otherwise: with ENVIRONMENT=local the composition root builds a
# SubprocessSandboxExecutor, and a tool command is then a child process of this
# very container. That image name only starts meaning something once a run is
# not local and the Docker executor is composed instead.
missing=""
for cmd in "$build" "$test"; do
    [ -n "$cmd" ] || continue
    exe=${cmd%% *}
    docker compose exec -T api sh -c "command -v '$exe' >/dev/null 2>&1" 2>/dev/null \
        || missing="$missing $exe"
done

if [ -n "$missing" ]; then
    echo
    echo "==> WARNING: not found where the tools run:$missing"
    echo "    Every candidate would fail on that, for a reason that has nothing"
    echo "    to do with the code the agents write."
    echo
    echo "    Either install them in docker/api.Dockerfile and rebuild"
    echo "    (docker compose up -d --build api), or drop the command so that"
    echo "    validation is recorded as SKIPPED instead of failed."
    echo
    if [ "${STRICT_TOOLS:-1}" = "1" ]; then
        die "refusing to spend a run on a toolchain that cannot execute.
       Re-run with STRICT_TOOLS=0 to proceed anyway (validation will fail),
       or set the commands yourself:
           BUILD='...' TEST='' ./run-on.sh \"$project_path\" \"$objective\""
    fi
fi

# Upload. The server detects the toolchain from the files it receives; the
# guess above is sent only where the caller overrode it, so the server and
# this script cannot disagree about a project only the server can see.
form=(-F "name=$name" -F "files=@$archive;filename=project.zip")
[ -n "${BUILD+x}" ] && form+=(-F "build_command=$build")
[ -n "${TEST+x}" ]  && form+=(-F "test_command=$test")
response="$(curl -sS -w '\n%{http_code}' -X POST "$API/v1/projects/upload" "${form[@]}")" \
    || die "could not reach the control plane"
http_code="${response##*$'\n'}"
body="${response%$'\n'*}"
detail() { printf '%s' "$body" | $PY -c 'import json,sys
try: print(json.load(sys.stdin).get("detail") or "")
except Exception: pass'; }
case "$http_code" in
    201) pid=$(printf '%s' "$body" | $PY -c 'import json,sys; print(json.load(sys.stdin)["id"])') ;;
    409) die "a project named '$name' already exists, and an upload may carry different
       code, so it is not reused. Pick another name:
           NAME=$name-2 ./run-on.sh \"$project_path\" \"$objective\"" ;;
    *)   die "upload refused (HTTP $http_code): $(detail)" ;;
esac
echo "==> project $name uploaded as $pid"

# Creating a project that already exists returns the existing record, commands
# and all, so a second run with different commands would silently use the first
# run's — which is how a run was spent executing `pytest` after the caller had
# replaced it. Compare, then correct it through the API.
read_toolchain() {
    curl -fsS "$API/v1/projects/$1" | $PY -c '
import json, sys
t = json.load(sys.stdin).get("toolchain") or {}
print(t.get("build_command") or "")
print(t.get("test_command") or "")
'
}
stored=$(read_toolchain "$pid")
stored_build=$(printf '%s\n' "$stored" | sed -n 1p)
stored_test=$(printf '%s\n' "$stored" | sed -n 2p)

if [ "$stored_build" != "$build" ] || [ "$stored_test" != "$test" ]; then
    printf '\n==> the project %s already exists, with different commands:\n' "$name"
    printf '    stored : build=%s | test=%s\n' "${stored_build:-none}" "${stored_test:-none}"
    printf '    wanted : build=%s | test=%s\n' "${build:-none}" "${test:-none}"
    printf '==> correcting its toolchain\n\n'
    curl -fsS -X PUT "$API/v1/projects/$pid/toolchain" -H 'content-type: application/json' \
      -d "{\"language\":\"$lang\",
           \"build_command\":$(json_or_null "$build"),
           \"test_command\":$(json_or_null "$test")}" >/dev/null \
      || die "the control plane refused to change this project's toolchain.
       It refuses while a run of the project is still in flight: a run reads
       these commands every time it validates a candidate, and changing them
       under it would judge its candidates by two different rules. Wait for
       that run, cancel it, or use another project name:
           NAME=$name-2 ./run-on.sh \"$project_path\" \"$objective\""
fi

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
