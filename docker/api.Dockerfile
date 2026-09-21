# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Control plane API + orchestrator image.
#
# Build from the REPOSITORY ROOT:
#   docker build -f docker/api.Dockerfile -t agentic-api:dev .
#
# The same image also runs the CPU-only fake worker (`agentic-worker`): both
# entry points ship in the same distribution, and a non-GPU worker needs
# nothing else. Only the GPU worker needs its own CUDA image.
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12

# --- build stage -----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build toolchain is needed only here; the runtime stage never sees it.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# Third-party dependencies first, in their own layer: they change far less
# often than the source tree, so editing src/ does not reinstall FastAPI.
COPY pyproject.toml README.md ./
RUN python -c "\
import pathlib, tomllib;\
pyproject = tomllib.loads(pathlib.Path('pyproject.toml').read_text());\
pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(pyproject['project']['dependencies']) + '\n')\
" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Then the project itself (console scripts agentic-api / agentic-worker).
COPY src ./src
# Prompt templates are runtime data the wheel force-includes: without them the
# build fails, and an image that skipped them would ship agents that cannot speak.
COPY prompts ./prompts
RUN pip install --no-cache-dir --no-deps .

# --- runtime stage ---------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# The orchestrator's own tools. Not optional extras: it creates a git worktree
# per candidate, so without git every run fails on its first job with
# "git could not be executed" — which is exactly what happened the first time
# a run was driven through the HTTP API rather than through the tests.
#
# ripgrep is what the repository-context provider prefers; it falls back to
# grep, so it is a speed choice rather than a requirement.
RUN apt-get update \
 && apt-get install --no-install-recommends -y git ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git --version && rg --version | head -1

# Unprivileged account: the control plane never needs root, and agent-produced
# code must never be one misconfiguration away from it.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app \
 && mkdir -p /var/lib/agentic/workspaces /var/lib/agentic/artifacts \
 && chown -R app:app /var/lib/agentic

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Source, prompts and (once they exist) alembic.ini + migrations. The build
# context is filtered by .dockerignore, so tests/docs/caches stay out.
COPY --chown=app:app . /app

COPY --chmod=0755 docker/entrypoint-api.sh /usr/local/bin/entrypoint-api.sh

USER app

EXPOSE 8000

# Readiness is the API's own /health endpoint; no curl in the image, urllib is
# already there. start-period covers migrations and connection pool warm-up.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import os,urllib.request,sys;\
url='http://127.0.0.1:%s/health' % os.environ.get('API_PORT','8000');\
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)" || exit 1

ENTRYPOINT ["entrypoint-api.sh"]
CMD []
