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

# pytest is not a dependency of the control plane; it is a dependency of the
# *target projects* the control plane works on, and it has to live here because
# this is where the deterministic tools actually execute. With ENVIRONMENT=local
# the composition root picks SubprocessSandboxExecutor (container.py passes
# `prefer_docker=not settings.environment.is_local`), so a tool command is a
# child process of the API container and TOOL_SANDBOX_IMAGE is never consulted
# at all. Without this line a Python target's `test_command` cannot start, and
# every candidate is marked non-viable for a reason that is not about the code.
#
# Python only. Per-project sandbox images are a recorded not-done; adding gcc,
# node, cargo and go here would be implementing that by the back door, in the
# one image that is supposed to stay the control plane.
# `pytest-asyncio` is not optional company for pytest here: the target
# project sets `asyncio_mode = "auto"`, so without the plugin every async
# fixture errors at setup — 38 errors out of 173 on a baseline that is
# green on the host. A validation step that fails on unmodified code marks
# every candidate non-viable for a reason that has nothing to do with the
# code the agents wrote, which is the exact poison the repair loop must
# not be fed.
RUN pip install --no-cache-dir "pytest>=8.3" "pytest-asyncio>=0.24"

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

# The project under work is bind-mounted at /projects/current and belongs to the
# host user (uid 1000), while this image runs as `app` (uid 10001). git refuses
# a repository it does not own — "detected dubious ownership" — so the very
# first worktree the orchestrator creates fails, and the run dies before a
# single agent has spoken.
#
# --system rather than --global, and that is the whole point of putting it here:
# SubprocessSandboxExecutor builds each child's environment from scratch and
# repoints HOME at the workspace, so a ~/.gitconfig belonging to `app` is simply
# not read by the git that a tool invokes. /etc/gitconfig is read whatever HOME
# says, which makes it the only file that covers both the orchestrator's own git
# calls and git run from inside the sandbox. It is also why this cannot live in
# the entrypoint: that already runs as `app` and cannot write /etc.
#
# Scoped to the one path, not `*`: the mount point is fixed by the compose file,
# so there is no reason to disarm the check for every directory in the image.
RUN git config --system --add safe.directory /projects/current

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
