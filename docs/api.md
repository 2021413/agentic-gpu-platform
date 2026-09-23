# HTTP API

> **What this document claims.**
> Everything below was read from `src/interfaces/` and is covered by
> `tests/api/`. The adapters behind the use cases (PostgreSQL, Redis, workers)
> are written separately; this document describes the *boundary*, not what is
> plugged into it. A deployment whose composition root is incomplete will start
> and then fail every request with a wiring error — see [Wiring](#wiring).

Two surfaces live in the same ASGI application:

| Surface | Prefix | Who calls it | Authentication |
|---|---|---|---|
| Public control plane | `/v1`, `/health`, `/ready` | users, CI, dashboards | none (deploy behind your own gateway) |
| Internal worker API | `/internal` | GPU workers | service token, mandatory |

`create_api(include_internal_api=False)` omits the internal surface for a
deployment that serves it on a separate port or ingress.

The interactive documentation is served at `/docs` and the OpenAPI document at
`/openapi.json`.

---

## Conventions

* **Identifiers are UUIDs**, as strings, everywhere.
* **Timestamps are ISO-8601 with a timezone**, always UTC.
* **Enumerations are uppercase strings** (`CREATED`, `PLANNING`, `READY`, ...);
  their values are the same as the domain's and are a compatibility contract.
* **Unknown fields are refused** (`extra: forbid`) on every request body: a typo
  in a field name is an error, never a silently ignored option.
* **Errors are RFC 9457 problem documents** with `Content-Type:
  application/problem+json` — see [Errors](#errors).
* **Every response carries `X-Request-ID`**, echoing the one you sent or a
  generated one, plus `X-Response-Time-Ms`. Quote the request id in a bug report:
  the access log line carries the same value.

---

## Public API

### `POST /v1/projects` — register a project

```json
{
  "name": "packet-parser",
  "repository_url": "https://github.com/acme/packet-parser.git",
  "local_path": null,
  "default_branch": "main",
  "toolchain": {
    "language": "python",
    "build_command": "python -m compileall .",
    "test_command": "pytest -q",
    "static_analysis_command": null,
    "install_command": null,
    "working_subdirectory": null,
    "environment": {}
  },
  "metadata": {}
}
```

At least one of `repository_url` / `local_path` is required (422 otherwise).
Build and test commands are configured, never guessed: a project that declares
none simply has no such validation step.

`201 Created`:

```json
{
  "id": "0f7e...",
  "name": "packet-parser",
  "repository_url": "https://github.com/acme/packet-parser.git",
  "default_branch": "main",
  "language": "python",
  "created_at": "2026-01-01T12:00:00+00:00"
}
```

Creation is **idempotent on the name**: posting the same name again returns the
existing project instead of a duplicate (still with `201`) — *unchanged*. It is
not a way to correct a project; see `PUT .../toolchain` below.

### `PUT /v1/projects/{project_id}/toolchain` — correct the commands

Body: the `toolchain` object above, on its own.

```json
{
  "language": "python",
  "build_command": "make",
  "test_command": "make test",
  "static_analysis_command": null,
  "install_command": null,
  "working_subdirectory": null,
  "environment": {}
}
```

A **replacement**, not a merge: what you send is what the project will run, and
a command you leave out is a command the project no longer has. That is the
only shape in which "this project should have no test command any more" can be
expressed at all.

Only the commands. A project's `name`, `repository_url`, `local_path` and
`default_branch` are not part of this resource and sending one is a `422` —
they say *which* code is worked on, and rewriting them in place would leave the
project's existing runs recorded against a tree they never touched. Changing
those still means a new project.

`200` with the same body as `GET`, and:

* `404` if the project is unknown;
* `409` `project_not_modifiable` while **any run of this project is still in
  flight** — a run reads these commands every time it validates a candidate, so
  a change landing mid-run would judge one run's candidates by two different
  definitions of passing. The problem document lists the runs in
  `details.active_run_ids`: wait for them, or cancel them.

### `GET /v1/projects/{project_id}`

`200` with the same body as above, `404` if unknown.

### `GET /v1/projects?limit=50&offset=0`

`200` with a JSON array of projects. `limit` is 1..200, `offset` >= 0.

### `POST /v1/projects/{project_id}/runs` — start a run

Header: `Idempotency-Key: <1..255 chars>` (optional but strongly recommended).

```json
{
  "objective": "Implement the packet parser and add tests",
  "candidate_count": 2,
  "metadata": {}
}
```

`candidate_count` (1..16) may be omitted, in which case the task-complexity
policy decides how many implementations are worth racing. The run's own limits
clamp the value further.

`202 Accepted` — the run is durable and scheduled; nothing waits behind
inference:

```json
{
  "id": "a1b2...",
  "project_id": "0f7e...",
  "status": "CREATED",
  "objective": "Implement the packet parser and add tests",
  "candidate_count": 2,
  "plan_revisions": 0,
  "repair_iterations": 0,
  "selected_candidate_id": null,
  "input_tokens": 0,
  "output_tokens": 0,
  "failure_kind": null,
  "failure_reason": null,
  "created_at": "2026-01-01T12:00:00+00:00",
  "updated_at": "2026-01-01T12:00:00+00:00",
  "completed_at": null
}
```

**Replaying the same `Idempotency-Key` returns the first run**, with the same
id and the same `202`. Delivery is assumed to be at-least-once: a client whose
request timed out must retry with the same key rather than create a twin run.

`status` is one of `CREATED`, `PLANNING`, `PLAN_READY`, `CODING`, `VALIDATING`,
`REVIEWING`, `REPAIRING`, `COMPLETED`, `FAILED`, `CANCELLING`, `CANCELLED`.

### `GET /v1/runs/{run_id}?detailed=false`

```json
{ "run": { "...": "as above" }, "plan": null, "candidates": [] }
```

With `detailed=true`, `plan` carries the latest plan revision (tasks,
dependencies, assumptions, constraints, risk areas) and `candidates` every
attempt with its validation and review outcome. Only structured results are
exposed — never an agent's reasoning.

### `POST /v1/runs/{run_id}/cancel`

Body optional: `{"reason": "superseded by #42"}`.

`200` with the run, whose status is `CANCELLED`. **Idempotent**: cancelling an
already cancelled or finished run returns its current state with the same `200`,
because the caller's intent is already satisfied. `404` if the run is unknown.

### `GET /v1/runs/{run_id}/candidates`

`200` with `{"candidates": [...]}`, `404` if the run is unknown (an empty list
would wrongly claim the run exists). Each candidate reports `status`, `viable`,
`build_passed`, `tests_passed`, `validation_summary`, `changed_files`,
`total_churn`, `review_verdict`, iteration counters, the `worker_id` that
produced it, a `summary` and its `uncertainties`.

### `GET /v1/runs/{run_id}/events` — SSE

See [Streaming](#streaming).

### `GET /v1/workers?only_available=false`

```json
{
  "workers": [
    {
      "id": "9c1d...",
      "model_id": "Qwen3-Coder-30B-A3B",
      "status": "READY",
      "endpoint": "http://worker-1:8000",
      "capacity": 4,
      "active_jobs": 1,
      "context_length": 262144,
      "gpu_type": "H200",
      "gpu_count": 1,
      "supported_roles": ["CODER", "PLANNER", "REVIEWER"],
      "registered_at": "2026-01-01T11:00:00+00:00",
      "last_heartbeat_at": "2026-01-01T12:00:00+00:00"
    }
  ]
}
```

The pool is *discovered*, never configured: this is whatever has registered and
is still heartbeating. `only_available=true` restricts it to workers that may
receive a new job right now; a `DRAINING` worker never qualifies.

### `GET /health` — liveness

`200 {"status": "alive"}`, always, from the process alone. It never touches
PostgreSQL, Redis or a worker: a probe that fails during a dependency outage
gets the container killed and turns a recoverable outage into a crash loop.

### `GET /ready` — readiness

```json
{
  "ready": false,
  "dependencies": [
    {"name": "database", "healthy": true, "detail": null},
    {"name": "redis", "healthy": false, "detail": "connection refused"}
  ]
}
```

`200` when every dependency answered, `503` otherwise — the instance then leaves
the load balancer without being restarted. Which dependencies are checked is
decided by the probe the composition root injects.

---

## Internal worker API

Every route below requires a service token and lives under `/internal/workers`.

### Authentication

Send **either** header:

```http
Authorization: Bearer $SERVICE_TOKEN
X-Service-Token: $SERVICE_TOKEN
```

(`X-Service-Token` exists because some proxies rewrite `Authorization`.)

* no credential → `401` with `WWW-Authenticate: Bearer realm="internal"` and
  `"code": "unauthenticated"`;
* wrong credential → `403`, `"code": "permission_denied"` — the token was read
  and rejected, so resending it is pointless.

The token is compared with `secrets.compare_digest`, in constant time: `==` on
bytes short-circuits on the first differing byte, which leaks the shared prefix
and makes a token guessable over enough requests. It is read from configuration
(`SERVICE_TOKEN`), never hard-coded, and the authenticator refuses to be built
with an empty one rather than starting with authentication silently disabled.

Authentication is an abstraction (`ServiceAuthenticator` in
`src/interfaces/worker_api/auth.py`); the shared token is the first
implementation, and replacing it with mTLS or signed JWTs touches no route.

### `POST /internal/workers/register`

```json
{
  "endpoint": "http://worker-1:8000",
  "model_id": "Qwen3-Coder-30B-A3B",
  "context_length": 262144,
  "max_concurrency": 4,
  "worker_id": "9c1d...",
  "supported_roles": ["PLANNER", "CODER", "REVIEWER"],
  "gpu": {"gpu_type": "H200", "gpu_count": 1, "memory_gb": 141.0, "tensor_parallel_size": 1},
  "supports_tools": true,
  "supports_json_schema": true,
  "metadata": {}
}
```

`endpoint` must be an `http(s)` URL. `worker_id` is optional and supplied by the
worker when it has a stable identity (a pod name, an instance id); **sending the
same id again refreshes that worker** instead of creating a twin that would
double the pool's apparent capacity.

`201 Created` with the worker representation shown under `GET /v1/workers`.

### `POST /internal/workers/{worker_id}/heartbeat`

```json
{"active_jobs": 2, "queued_jobs": 0, "draining": false}
```

`200` with the refreshed worker. A heartbeat from a worker previously declared
unavailable readmits it: a network partition must not permanently remove a
healthy GPU. `404` if the worker is not registered (it should re-register).

### `POST /internal/workers/{worker_id}/drain`

No body (or `{}`). `200` with the worker in `DRAINING`: it finishes what it
already holds and accepts nothing new.

### `DELETE /internal/workers/{worker_id}?graceful=true`

`204 No Content`, **idempotent even for a worker that is already gone** — a
retried deregistration during a rolling shutdown must not produce a 404 storm.
Jobs the worker still held are reclaimed through lease expiry.

### `GET /internal/workers/{worker_id}/health`

```json
{
  "id": "9c1d...",
  "status": "READY",
  "live": true,
  "accepts_new_jobs": true,
  "active_jobs": 1,
  "capacity": 4,
  "last_heartbeat_at": "2026-01-01T12:00:00+00:00"
}
```

This is the *control plane's opinion*, built from registration and heartbeats —
not a proxy to the worker's own health endpoint. It answers "would a job be sent
there right now?", which is the only question scheduling asks. `404` if unknown.

---

## Streaming

`GET /v1/runs/{run_id}/events` returns `text/event-stream`.

```bash
curl -N -H 'Accept: text/event-stream' \
     http://localhost:8000/v1/runs/$RUN_ID/events

# resume after the last frame you processed
curl -N -H 'Last-Event-ID: 42' \
     http://localhost:8000/v1/runs/$RUN_ID/events

# for a client that cannot set headers (e.g. the browser EventSource API)
curl -N "http://localhost:8000/v1/runs/$RUN_ID/events?after=42"
```

Each frame:

```text
event: run.state_changed
id: 42
data: {"sequence":42,"name":"run.state_changed","occurred_at":"2026-01-01T12:00:03+00:00","payload":{"run_id":"a1b2...","previous":"CREATED","current":"PLANNING"}}
```

* `event` is the domain event name: `run.created`, `run.state_changed`,
  `run.plan_requested`, `run.plan_completed`, `candidate.started`,
  `candidate.validation_started`, `candidate.validation_completed`,
  `candidate.completed`, `review.requested`, `review.completed`,
  `run.repair_requested`, `candidate.selected`, `run.completed`, `run.failed`,
  `run.cancelled`.
* `id` is the **resumption cursor** and is present only on events replayed from
  the durable store. Live events are streamed without an `id` because they have
  no durable sequence yet, so `Last-Event-ID` stays at the last durable event:
  on reconnect you may see a few live events twice, but never a gap. Duplicates
  are recoverable — every frame carries its event name and payload — a silent
  gap is not.
* A `404` for an unknown run is returned *before* the stream opens, as an
  ordinary problem document.
* The connection is kept alive with a comment frame every 15 seconds, because a
  planning stage can be silent for minutes and proxies cut idle connections.

**The stream ends by itself** when the run reaches a terminal state
(`run.completed`, `run.failed`, `run.cancelled`), and immediately after the
replay when the run was already finished before you connected. Disconnecting is
always safe: the subscription is released on the server.

**No hidden reasoning is ever streamed.** Frames carry structured progress only;
fields known to hold model prose (`reasoning`, `chain_of_thought`, `raw_output`,
`prompt`, `messages`, ...) are stripped, at any nesting depth, and long strings
are truncated before leaving the process.

---

## Errors

Every failure is an RFC 9457 document:

```json
{
  "type": "/problems/not_found",
  "title": "Resource not found",
  "status": 404,
  "detail": "Run not found",
  "instance": "/v1/runs/a1b2...",
  "code": "not_found",
  "details": {"entity": "Run", "id": "a1b2..."},
  "request_id": "5f3c9c1e..."
}
```

`type` is a relative URI reference (resolved against the request URL) so no
hostname is baked into error bodies; the base is configurable per deployment.
**Branch on `code`**, never on the prose in `title` or `detail`.

| `code` | Status | Meaning and why that status |
|---|---|---|
| `validation_error` | 422 | The payload did not satisfy the schema. `details.errors` lists the offending locations. |
| `not_found` | 404 | The referenced project, run or worker does not exist. |
| `invalid_state_transition` | 409 | The resource exists but cannot move that way; retrying unchanged keeps failing until it moves — that is a conflict, not a bad request. |
| `run_not_modifiable` | 409 | The run is already terminal. |
| `project_not_modifiable` | 409 | The project has runs in flight, which are reading the toolchain being replaced. |
| `run_cancelled` | 409 | The work was abandoned because the run is cancelling. |
| `idempotency_conflict` | 409 | The same idempotency key was reused with a different body: the caller contradicted itself. |
| `job_lease_expired` | 409 | A worker reported on a job whose lease it no longer holds; the result is discarded. |
| `job_not_retryable` | 409 | The job exhausted its attempts. |
| `worker_unavailable` | 409 | That specific worker cannot take work right now — a state conflict on a named resource. |
| `plan_invalid` | 422 | A syntactically valid plan that is unusable (cycles, dangling dependencies). |
| `structured_output_invalid` | 422 | The model answered, but the answer violates its schema. |
| `inference_failed` | 502 | The inference engine misbehaved or was unreachable: the control plane is healthy, its upstream is not. |
| `llm_timeout` | 504 | An inference request passed its deadline. Gateway timeout says "upstream, not you". |
| `no_compatible_worker` | 503 | No worker satisfies the requirements *right now*. The fleet is elastic, so this is temporary and carries `Retry-After: 5`. |
| `tool_execution_failed` | 500 | A deterministic tool could not run at all (missing binary, sandbox refusal). A tool that ran and failed is a normal result, not an error. |
| `workspace_error` | 500 | An isolated workspace could not be created, mutated or released. |
| `candidate_error` | 500 | A candidate reached an unusable state. |
| `unauthenticated` | 401 | No service token on an internal route. |
| `permission_denied` | 403 | The service token was rejected. |
| `not_ready` | 503 | Readiness check failed. |
| `internal_error` | 500 | Unexpected failure. The `detail` is deliberately opaque — an internal message can quote a prompt, a path or a connection string. Use `request_id` to find the log line. |

A code the API has never heard of is answered `500`, not `400`: an unmapped
error means the boundary was not updated for something the platform raises,
which is a defect on the server's side.

---

## Wiring

`create_api` assembles routers, middleware and error handlers; it never builds a
dependency. The composition root passes one container:

```python
from interfaces.api.app import create_api
from interfaces.api.dependencies.container import ApiDependencies

app = create_api(dependencies=ApiDependencies(...))
# or, when the adapters are only available later:
app = create_api()
app.state.dependencies = ApiDependencies(...)   # before the first request
```

`ApiDependencies` (see `src/interfaces/api/dependencies/container.py`) requires
the five project/run use cases, the five worker use cases, the `EventBus` port
(SSE needs the live stream), a `ReadinessProbe` and a `ServiceAuthenticator`.
A request served without it fails with one explicit `RuntimeError` naming
`app.state.dependencies`, rather than an `AttributeError` three frames deeper.

Individual providers can be replaced through
`app.dependency_overrides[...]` — that is how `tests/api/` substitutes doubles.
