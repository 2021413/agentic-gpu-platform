# Development guide

> **What this document claims.**
> Commands touching the toolchain (`make`, Docker, Compose, pytest, pre-commit,
> CI) were run or dry-run against this repository. Steps that depend on layers
> still being written are marked **[target]** and say what happens today.
> The `domain` layer and the architecture tests exist; `application`,
> `infrastructure`, `interfaces`, `bootstrap` and `worker_agent` are being
> written in parallel, so the API container cannot serve requests yet.

---

## 1. Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.12+ | everything |
| Docker Engine + Compose v2 | recent | local stack, integration tests |
| GNU Make | any | the shortcuts below |
| NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/) | — | **only** for a real GPU worker |

No GPU is required to develop, to run the stack, or to run CI.

## 2. First install

```bash
git clone <this repository> && cd programGen
make install          # creates .venv, installs -e ".[dev]", installs the git hooks
cp .env.example .env  # then edit: at minimum SERVICE_TOKEN
```

`make` alone prints every target with its description.

Prefer `uv`? `uv venv && uv pip install -e ".[dev]"` produces the same
environment; the Makefile deliberately sticks to stdlib `venv` + `pip` so that
no extra tool is mandatory.

## 3. Daily commands

| Command | What it does |
|---|---|
| `make lint` | `ruff check` + `ruff format --check` |
| `make format` | applies formatting and safe autofixes |
| `make typecheck` | `mypy` in strict mode |
| `make test` | unit tests — no Docker, no GPU |
| `make test-integration` | tests marked `integration` (PostgreSQL + Redis) |
| `make test-e2e` | the end-to-end scenario against the fake worker |
| `make check` | lint + typecheck + unit tests, i.e. what CI runs without Docker |
| `make hooks-run` | every pre-commit hook on the whole tree |
| `make docker-up` / `make docker-down` | the local stack, without / with GPU teardown |
| `make clean` | caches and build artefacts |

## 4. The local stack

```bash
make docker-up          # api + postgres + redis + fake worker (no GPU)
make docker-logs        # follow everything
make docker-down        # stop, keep the volumes
make docker-reset       # stop and delete the volumes (Postgres data, model cache)
```

| Service | Image | Host port | Role |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | `${POSTGRES_PORT:-5432}` | durable run state |
| `redis` | `redis:7-alpine` | `${REDIS_PORT:-6379}` | registry, queue, events |
| `api` | built from `docker/api.Dockerfile` | `${API_PUBLISHED_PORT:-8000}` | control plane + orchestrator |
| `worker-fake` | same image, `agentic-worker` | — | CPU-only worker, deterministic fake LLM |
| `worker-gpu` | built from `docker/worker.Dockerfile` | — | profile `gpu` only, real vLLM worker |

Startup is ordered by health, not by hope: `api` waits for
`postgres`/`redis` to pass `pg_isready`/`PING` (`condition: service_healthy`),
and `worker-fake` waits for `GET /health` on the API.

> **Today's reality.** `api` runs the `agentic-api` console script
> (`bootstrap.app:main`). While `src/bootstrap/` does not exist,
> `docker/entrypoint-api.sh` stops immediately with an explicit message instead
> of an obscure traceback, and `worker-fake` never starts because its
> `depends_on` is never satisfied. Until then, work with the backing services
> alone:
> ```bash
> docker compose up -d postgres redis
> make test
> ```
> Nothing needs to change in the Docker files once the layer lands.
> `API_SERVER=uvicorn` switches the container to a plain
> `uvicorn bootstrap.app:create_app --factory` (set `API_APP_FACTORY=0` if the
> module exposes a ready-made `app` object instead) — useful if `main()` is not
> the entry point that ships first.

### Running the API on the host with autoreload

```bash
make dev    # starts postgres + redis in Docker, then uvicorn --reload on the host
```

Override the import path if it differs: `make dev API_APP=bootstrap.app:create_app`.

## 5. Tests

```bash
make test               # pytest -m "not integration and not e2e"
make test-integration   # pytest -m integration    (needs a Docker daemon)
make test-e2e           # pytest -m e2e
make test-all
```

* `tests/domain/` — pure rules: state machines, retries, eligibility, leasing,
  candidate selection. Fast, no I/O.
* `tests/architecture/` — the dependency rule, enforced by AST inspection.
  If you make `domain` import FastAPI, this is what fails.
* `tests/infrastructure/`, `tests/api/`, `tests/e2e/` — **[target]**, owned by
  the layers being written.

The markers `integration` and `e2e` are declared in `pyproject.toml`; the
default `make test` selection is exactly "everything that needs no service".

## 6. Migrations

```bash
make migrate                               # alembic upgrade head
make migration m="add runs and jobs"       # autogenerate a revision
```

**[target]**: `alembic.ini` and the migration tree belong to the infrastructure
layer and are not in the repository yet, so these targets fail until it lands.
CI already applies migrations conditionally (`if: hashFiles('alembic.ini')`),
so nothing has to change there either.

## 7. Workers

### Add a second CPU worker (proving multi-worker scheduling)

The clean way — a second service, with its own identity and endpoint, in a
`docker-compose.override.yml`:

```yaml
services:
  worker-fake-2:
    extends:
      file: docker-compose.yml
      service: worker-fake
    environment:
      WORKER_ENDPOINT: http://worker-fake-2:9000
```

`docker compose up -d --scale worker-fake=2` also works and gives each replica
its own generated `WORKER_ID`, but both replicas then advertise the same
`WORKER_ENDPOINT` hostname (Docker DNS round-robins between them). That is fine
while jobs are *pulled* from the queue, and wrong as soon as the control plane
dials a specific worker — prefer the explicit service.

### Remove a worker

Drain first, so in-flight work is finished rather than reclaimed by timeout:

```bash
curl -X POST -H "Authorization: Bearer $SERVICE_TOKEN" \
     http://localhost:8000/internal/workers/$WORKER_ID/drain     # [target endpoint]
docker compose stop worker-fake
```

Killing it without draining is also safe — it just costs a timeout: the jobs
are requeued when their leases expire. See
[worker-lifecycle.md](worker-lifecycle.md) section 6.

### Attach a real vLLM / Qwen GPU worker

Checklist before starting:

1. NVIDIA driver and Container Toolkit installed — verify with
   `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`;
2. enough VRAM for the model (Qwen3-Coder-30B-A3B in bf16 wants a lot; use
   several GPUs with `TENSOR_PARALLEL_SIZE`, or a quantised build);
3. enough disk in the model cache volume (tens of GB);
4. `HUGGING_FACE_HUB_TOKEN` if the repository is gated.

Then, in `.env`:

```dotenv
MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct   # never hard-coded anywhere
MODEL_CONTEXT_LENGTH=131072
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90
WORKER_CONCURRENCY=4
# Qwen tool calling, if the orchestrator uses native tool calls:
VLLM_EXTRA_ARGS=--enable-auto-tool-choice --tool-call-parser hermes
```

and start it:

```bash
make docker-up-gpu        # docker compose --profile gpu up -d --build
docker compose logs -f worker-gpu
```

What happens inside the container (`docker/entrypoint-worker.sh`):

1. vLLM starts on `127.0.0.1:${VLLM_PORT}` with the model named by `MODEL_ID`,
   reading weights from `HF_HOME=/models/huggingface` — the `hf-cache` named
   volume, **never the image**;
2. the entrypoint waits for `GET /health` (up to
   `VLLM_STARTUP_TIMEOUT_SECONDS`, 30 min by default: a cold cache downloads
   tens of GB);
3. `agentic-worker` starts, registers with `CONTROL_PLANE_URL`, and heartbeats;
4. if either process exits, the container exits, so a worker that lost its
   engine leaves the registry instead of advertising capacity it no longer has.

Pre-download the weights once, so the first run is not a 30-minute silence:

```bash
docker volume create agentic-gpu-platform_hf-cache
docker run --rm -v agentic-gpu-platform_hf-cache:/models/huggingface \
  -e HF_HOME=/models/huggingface \
  python:3.12-slim bash -c \
  "pip install -q huggingface_hub && \
   huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct"
```

Sanity-check the engine alone, without the platform:

```bash
docker compose --profile gpu exec worker-gpu \
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8001/health').status)"
```

### A worker on another machine (RunPod, a remote GPU box)

Nothing in the core knows about any cloud provider. Run the same image there
and point it at the control plane:

```bash
docker run -d --gpus all \
  -e CONTROL_PLANE_URL=https://control-plane.example.com \
  -e SERVICE_TOKEN=... \
  -e WORKER_ENDPOINT=https://gpu-a.example.com:9000 \
  -e MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct \
  -v /mnt/models:/models/huggingface \
  -p 9000:9000 \
  agentic-worker-gpu:dev
```

`WORKER_ENDPOINT` must be what the **control plane** can dial. Automatic
provisioning belongs to an optional `ComputeProvider` adapter
(`src/domain/ports/compute_provider.py`), never to orchestration code.

## 8. Configuration

Every key is documented in [`.env.example`](../.env.example). The keys from
section 22 of the specification are grouped first; the rest are additions the
stack needs in practice, and those marked `(proposed)` are names chosen by this
tooling while the settings module is being written — **confirm them against
`bootstrap`'s settings when it lands, and change them in one place if they
differ**:

`ENVIRONMENT`, `METRICS_ENABLED`, `REAPER_INTERVAL_SECONDS`,
`SCHEDULER_STRATEGY`, `WORKSPACE_ROOT`, `ARTIFACT_ROOT`, `LLM_PROVIDER`,
`INFERENCE_BASE_URL`, `INFERENCE_API_KEY`, `WORKER_ROLES`, `WORKER_PORT`,
`WORKER_DRAIN_TIMEOUT_SECONDS`.

Never commit a `.env`: a pre-commit hook refuses it, and `gitleaks` scans for
tokens.

## 9. Pre-commit and CI

`make install` installs the hooks; they run Ruff (lint + format), mypy from the
project virtualenv, file-hygiene checks, large-file and secret detection.

CI (`.github/workflows/ci.yml`) has four jobs:

| Job | Contains |
|---|---|
| `lint` | Ruff lint, Ruff format check, mypy strict |
| `test-unit` | unit + architecture tests, coverage, no Docker |
| `test-integration` | PostgreSQL 16 and Redis 7 service containers, migrations if `alembic.ini` exists, `integration` then `e2e` markers |
| `docker-build` | builds `docker/api.Dockerfile` and smoke-tests the image |

**No job requires a GPU** — an explicit acceptance criterion. The CUDA/vLLM
image is never built in CI: it is huge and nothing there could run it. While a
marker collects no test, the step reports a notice instead of failing.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `bootstrap.app cannot be imported` when `api` starts | Expected today: the composition root does not exist yet (section 4) |
| `worker-fake` never starts | Its `depends_on: api (service_healthy)` is unsatisfied — same cause |
| `port is already allocated` | Change `POSTGRES_PORT`, `REDIS_PORT` or `API_PUBLISHED_PORT` in `.env` |
| `docker: Error response ... nvidia` | NVIDIA Container Toolkit missing or not configured; the default profile still works |
| vLLM: `No available memory for the cache blocks` | Lower `GPU_MEMORY_UTILIZATION` or `MODEL_CONTEXT_LENGTH`, or raise `TENSOR_PARALLEL_SIZE` |
| The worker downloads the model on every start | The `hf-cache` volume is not mounted, or `HF_HOME` was overridden |
| Integration tests cannot reach PostgreSQL | Start the stack (`docker compose up -d postgres redis`) or check `DATABASE_URL` |
| `make migrate` fails | `alembic.ini` is not in the repository yet (section 6) |

## 11. Where to read next

* [architecture.md](architecture.md) — layers, the dependency rule, ports.
* [worker-lifecycle.md](worker-lifecycle.md) — states, heartbeats, leases.
