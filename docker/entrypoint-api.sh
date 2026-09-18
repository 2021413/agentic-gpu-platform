#!/bin/sh
# ---------------------------------------------------------------------------
# Control plane entry point.
#
#   1. any argument given  -> run it verbatim (compose `command:`, one-shots
#                             such as `alembic upgrade head`, a shell, ...);
#   2. API_SERVER=console  -> `agentic-api`, the console script declared in
#      (default)              pyproject.toml as bootstrap.app:main;
#   3. API_SERVER=uvicorn  -> documented fallback, useful while bootstrap.main
#                             does not exist yet or to get --reload.
#
# The bootstrap layer is still being written; if it cannot be imported we fail
# loudly here instead of letting uvicorn print an obscure import traceback.
# ---------------------------------------------------------------------------
set -eu

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

if ! python -c "import bootstrap.app" >/dev/null 2>&1; then
    cat >&2 <<'MSG'
[entrypoint-api] FATAL: `bootstrap.app` cannot be imported.

The control plane composition root (src/bootstrap/app.py, exposing main() and
the FastAPI application) is not part of this image yet. Until it lands:

  - run a one-shot command instead:
        docker compose run --rm api python -c "import domain; print('ok')"
  - or override the service command in docker-compose.yml / docker compose run.

Once bootstrap exists, this container starts by itself with no change here.
MSG
    exit 1
fi

case "${API_SERVER:-console}" in
    console)
        exec agentic-api
        ;;
    uvicorn)
        # API_APP defaults to an application factory; set API_APP_FACTORY=0 if
        # bootstrap.app exposes a module-level `app` object instead.
        set -- uvicorn "${API_APP:-bootstrap.app:create_app}" \
            --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
        if [ "${API_APP_FACTORY:-1}" = "1" ]; then
            set -- "$@" --factory
        fi
        exec "$@"
        ;;
    *)
        echo "[entrypoint-api] unknown API_SERVER='${API_SERVER}' (console|uvicorn)" >&2
        exit 2
        ;;
esac
