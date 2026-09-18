# Agentic GPU Platform

A horizontally scalable platform that orchestrates one or more GPU inference
workers to perform agentic software engineering on a shared project.

```text
              Planner
                 │
 Orchestrator ───┼── Coder ──► deterministic tools (compile · test · analyse)
                 │                        │
              Reviewer ◄──────────────────┘
                 │
            retry if needed
```

Two rules shape the whole design:

* **Only deterministic tool results prove anything.** A model claiming "tests
  passed" establishes nothing; a `ToolResult` with exit code 0 does.
* **Every loop is bounded.** Plan revisions, coder iterations, repairs and job
  attempts all have budgets, and an exhausted budget ends a run cleanly instead
  of spinning.

GPU workers register themselves at runtime and may join or disappear mid-run.
The orchestrator schedules at the job level and never binds a project to a fixed
pool, so adding a GPU is a `docker run`, not a redeploy.

---

## Architecture

```text
interfaces  ──►  application  ──►  domain
                                     ▲
infrastructure  ─────────────────────┘   (implements the domain's ports)
```

The arrow never reverses. `domain` imports no framework, no driver and no outer
layer — a rule enforced mechanically by `tests/architecture/`, which parses
every module's imports and fails the build on a misplaced dependency.

| Layer | Holds | Examples |
|---|---|---|
| `domain` | entities, value objects, policies, **ports** | `Run`, `Job`, `Worker`, `RunStateMachine`, `RetryPolicy`, `LLMProvider` |
| `application` | use cases and orchestration | `CreateRunUseCase`, `RunOrchestrator`, `JobExecutor` |
| `infrastructure` | adapters | PostgreSQL, Redis, vLLM-compatible HTTP, git worktrees, sandbox |
| `interfaces` | HTTP | public API, internal worker API, SSE |
| `bootstrap` | composition root | the one place that names concrete classes |
| `worker_agent` | the process next to a GPU | register, heartbeat, drain, leave |

Details in [`docs/architecture.md`](docs/architecture.md).

---

## Running it locally

No GPU required: the local stack serves the whole workflow with a fake
inference provider.

```bash
cp .env.example .env
make docker-up          # api · postgres · redis · fake worker
make migrate            # apply the schema
```

For development without containers:

```bash
make install            # creates .venv and installs the dev extras
make check              # ruff + ruff format + mypy --strict
make test               # unit tests: no Docker, no GPU
make test-integration   # PostgreSQL and Redis via testcontainers
make test-e2e           # the full workflow on the real stack
```

See [`docs/development.md`](docs/development.md).

---

## Using it

Create a project, then run an objective against it.

```bash
curl -X POST localhost:8080/v1/projects \
  -H 'content-type: application/json' \
  -d '{"name":"demo","repository_url":"https://example.com/demo.git",
       "toolchain":{"language":"python","build_command":"python -m compileall .",
                    "test_command":"pytest -q"}}'

curl -X POST localhost:8080/v1/projects/$PROJECT_ID/runs \
  -H 'content-type: application/json' \
  -H "idempotency-key: $(uuidgen)" \
  -d '{"objective":"Implement the packet parser and add tests","candidate_count":2}'
```

Follow it live — run state, candidate progress, tool results, review verdict:

```bash
curl -N localhost:8080/v1/runs/$RUN_ID/events
```

The stream carries structured progress only. Hidden chain-of-thought is never
forwarded, by construction: the inference adapter reads message content and
nothing else.

Inspect and stop:

```bash
curl localhost:8080/v1/runs/$RUN_ID
curl localhost:8080/v1/runs/$RUN_ID/candidates
curl -X POST localhost:8080/v1/runs/$RUN_ID/cancel
```

The full surface is in [`docs/api.md`](docs/api.md).

### Build and test commands are configuration, never guesses

A project declares its toolchain. The platform never infers a build command,
because reporting a guessed command's failure as a code defect would poison the
repair loop. A stage with no configured command is recorded as *skipped*, never
as passed.

---

## GPU workers

### How a worker joins

The worker agent runs beside an inference server inside the GPU container. It
waits for that server to answer before registering — advertising capacity that
does not exist wastes the first job scheduled onto it — then announces its
model, context length, concurrency and supported roles:

```text
POST /internal/workers/register
POST /internal/workers/{id}/heartbeat   every HEARTBEAT_INTERVAL_SECONDS
```

Registration is idempotent on the worker id, so a restart refreshes the entry
instead of creating a twin that doubles the apparent capacity.

### Adding another GPU

```bash
MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct \
docker compose --profile gpu up -d --scale worker-gpu=2
```

Nothing else changes: no orchestrator restart, no project restart, no code
change. The scheduler asks the registry on every job, so a worker that appears
mid-run receives the next one.

### Draining and removing one

```bash
curl -X POST localhost:8080/internal/workers/$WORKER_ID/drain \
     -H "authorization: Bearer $SERVICE_TOKEN"
```

A draining worker receives no new jobs but finishes what it holds, then
deregisters once idle. `SIGTERM` does the same thing: the container stop drains
rather than kills.

### Pointing it at a real Qwen/vLLM engine

The model is configuration, never code. `MODEL_ID`, `MODEL_CONTEXT_LENGTH`,
`TENSOR_PARALLEL_SIZE` and `GPU_MEMORY_UTILIZATION` drive the vLLM server; the
orchestrator only ever sees a registered endpoint and its declared capabilities.
Weights stay outside the image, on a mounted Hugging Face cache. Setup in
[`docs/development.md`](docs/development.md).

---

## Failure recovery

Failures are not interchangeable, and the retry policy branches on what actually
went wrong:

| What failed | What happens |
|---|---|
| Worker unreachable, transport error | retried **on another worker** |
| Inference timeout or server error | retried on another worker |
| Model answered, but violated its schema | re-asked once with the exact violation attached, then reported honestly |
| A tool could not run at all | retried on the same worker: the workspace is unchanged |
| Build or tests failed | **not a retry** — it is work for the coder, through the bounded repair loop |
| Reviewer returned FAIL | repair loop, until the repair budget is spent |

**A worker that disappears mid-job** stops renewing its lease. The lease lapses,
the job becomes claimable again, and its attempt counter increments — which is
what makes permanently stuck jobs impossible. The vanished worker is declared
unavailable once its heartbeat times out, and that is visible on the event
stream rather than silent.

**Restarting the orchestrator** loses nothing: run state, plans, candidates,
jobs, reviews, tool results and the event history all live in PostgreSQL. What a
restart loses is the intent to act, which `resume_active_runs` restores.

Lifecycle details in [`docs/worker-lifecycle.md`](docs/worker-lifecycle.md).

---

## Isolation and safety

Each candidate owns its own git worktree on its own branch; two coders can never
write to the same path. The accepted patch reaches the base repository through
one controlled merge and no other route.

Tool execution is treated as hostile. Commands are argument vectors — no shell,
so no quoting surface — and run under explicit limits with an environment built
from an allow-list, so no control-plane secret ever enters a project sandbox.

**Be precise about what that buys you.** The subprocess sandbox bounds time,
memory and output size and keeps secrets out, but it does **not** isolate the
filesystem or the network: it protects against accidents, not against hostile
code. Running model-written code in production means using the Docker sandbox,
where `--network none`, `--cap-drop ALL`, `--pids-limit` and an empty
environment are enforced by the kernel.

---

## Configuration

Environment driven, validated as a whole at startup. Combinations that cannot
work are rejected immediately rather than at 3am: a heartbeat timeout shorter
than the beat would reap healthy workers, and the fake inference provider is
refused outside local development. Every key is documented in
[`.env.example`](.env.example).

---

## Tests

```bash
make test               # ~450 tests, no Docker, no GPU
make test-integration   # PostgreSQL and Redis
make test-e2e           # a full run on the real stack with a fake model
```

The suite is structured around the behaviours that matter rather than around
files: a draining worker receives no new jobs, an expired lease makes a job
retryable until the budget is spent, two candidates never share a workspace, a
worker joining mid-run gets the next job, an abandoned transaction publishes
nothing, and a failing build skips the test stage.

CI never needs a GPU.
