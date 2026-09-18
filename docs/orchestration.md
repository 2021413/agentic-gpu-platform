# Orchestration

How a run gets from an objective to a deterministic `COMPLETED` or `FAILED`.

Everything described here is implemented in `src/application/orchestration/` and
exercised by `tests/application/` (in-memory) and `tests/e2e/` (real stack).

---

## 1. The shape of the thing

Orchestration is a **governed state machine**, not a recursive agent loop. The
set of reachable states is small, enumerable, and validated in one place
(`domain/services/run_state_machine.py`). An illegal transition raises, no
matter which use case attempted it.

```text
CREATED ──► PLANNING ──► PLAN_READY ──► CODING ──► VALIDATING ──► REVIEWING ──► COMPLETED
   │                          │            ▲                          │
   │                          │            └──────── REPAIRING ◄──────┘
   └──────────────────────────┴──► CODING           (bounded)
              (planner bypassed for trivial work)

any live state ──► CANCELLING ──► CANCELLED
any live state ──► FAILED
```

Two properties matter more than the diagram:

* **Terminal states absorb.** `COMPLETED`, `FAILED` and `CANCELLED` allow no
  outgoing transition, so a late callback cannot resurrect a finished run.
* **Cancellation is legal from every live state**, and is idempotent.

---

## 2. Why the run is persisted before it is scheduled

`CreateRunUseCase` writes the run, commits, and only then returns. Scheduling
happens afterwards. A crash in between leaves a `CREATED` run that
`resume_active_runs` picks up; the opposite order would acknowledge a run to a
client and then lose it.

The same ordering applies to jobs: they are persisted as `QUEUED` inside the
transaction and published to the queue after the commit. Durable first,
transport second. At-least-once delivery is assumed, so `enqueue` is idempotent.

---

## 3. Deciding how much machinery an objective deserves

`TaskComplexityPolicy` runs at creation and answers two questions: does this
need a plan, and how many candidates is it worth?

| Assessment | Plan | Candidates |
|---|---|---|
| `TRIVIAL` — short, mechanical (typo, rename, docstring) | skipped | 1 |
| `SIMPLE` — single-step | yes | 1 |
| `COMPLEX` — multi-step vocabulary, several files, long objective | yes | up to `MAX_PARALLEL_CANDIDATES` |

The default implementation is a deterministic heuristic. No routing LLM is
required: a heuristic is cheaper, reproducible, and wrong in ways a user can
see. Replacing it with a model later means implementing one interface.

---

## 4. The loop

```text
1  create run                     persisted, then scheduled
2  PLAN job        ──► planner    structured plan, dependency graph validated
3  N candidates                   one isolated git worktree each
4  CODE job × N    ──► coder      patch applied in its own workspace, then read back
5  BUILD job                      only if the project configured a build command
6  TEST job                       skipped when the build already failed
7  STATIC_ANALYSIS job            advisory unless configured as blocking
8  select                         deterministic evidence first
9  REVIEW job      ──► reviewer   patch + validation results, not the coder's transcript
10 PASS ──► merge ──► COMPLETED
   FAIL ──► REPAIRING ──► back to 4, while the budget allows
```

### Why each validation stage is its own job

Build, tests and static analysis are separate jobs rather than one opaque
"validate" step. Each stage is then individually observable, individually
retryable, and individually attributable when something fails. The sequence
stops at the first observed failure — running a test suite against code that did
not compile tells you nothing you do not already know.

A stage whose command the project never configured is recorded as **skipped**,
never as passed. `ValidationReport` distinguishes the three states (`pass`,
`fail`, `skipped`) precisely so that "we did not look" cannot masquerade as "it
works".

### Why the reviewer does not see the coder's conversation

It receives the patch, the deterministic results, the implementation summary and
the stated uncertainties. Reviewing the reasoning that produced a defect is how
a reviewer inherits it.

---

## 5. Candidate selection

`DeterministicCandidateSelectionPolicy` ranks on evidence, in this order:

1. a reviewer `PASS`;
2. tests observed passing;
3. build observed succeeding;
4. a non-empty patch at all.

Ties break towards the **smaller diff** — a deliberate bias towards reviewable
work. There is no invented numeric score: a candidate wins because a tool said
so, not because a model liked it.

A candidate with no patch is reported as empty rather than as "failed
validation", so the rationale names the real cause.

---

## 6. Scheduling across a pool that changes

The scheduler is asked on **every job**. Nothing is cached, nothing is pinned to
a project, and the pool is discovered from the registry each time. That is the
entire reason a worker can appear or vanish mid-run without a restart.

`LeastLoadedCompatibleScheduler` filters on eligibility — health, drain state,
free capacity, role support, model compatibility, context window — then orders
by occupancy, breaking ties deterministically so scheduling is reproducible in
tests.

When no worker is compatible, `select` returns `None` rather than raising. An
empty pool is an ordinary, transient condition in an elastic fleet: the job is
requeued with backoff, not failed.

Reserving a slot on the chosen worker matters because heartbeats are far too
coarse to reflect second-by-second occupancy.

---

## 7. Failure, and why kinds are not interchangeable

`RetryPolicy` branches on `FailureKind`:

| Kind | Action | Why |
|---|---|---|
| `INFRASTRUCTURE` | retry on another worker | the job was fine, the machine was not |
| `INFERENCE` | retry on another worker | another engine may well succeed |
| `INVALID_STRUCTURED_OUTPUT` | one repair prompt, bounded | show the model its own answer and the exact violation |
| `TOOL` | retry on the same worker | the tool could not run; the workspace is unchanged |
| `COMPILATION` / `TEST` / `REVIEW` | **not a retry** | the code is wrong; that is the coder's work |
| `CANCELLED` | stop | |

Treating a failing test as a retryable job failure would burn the retry budget
re-running a suite that will fail identically. Treating an unreachable worker as
a code defect would send a coder to fix a network partition.

---

## 8. Leases, and why jobs cannot get stuck

A consumer claims a job under a time-bounded lease and renews it while working.
If the consumer disappears, the lease lapses; the job's attempt counter
increments and it becomes claimable again. Reclaiming is a range scan on the
lease expiry, so detection costs nothing.

Every report carries the lease token. A late report from a previous holder is
rejected instead of corrupting the state of a job that has already been
reassigned.

When a job finally exhausts its attempts, `handle_dead_job` fails its run. That
path exists because silence is the worst outcome: a client would otherwise watch
a run that can no longer progress.

---

## 9. Concurrency

* Candidates run concurrently, each in its own git worktree, on its own branch.
  No two writable workspaces ever share a path.
* Decisions *about a run* are serialized by `RunCoordinator`. Candidates finish
  at unpredictable moments and all ask the same question — "are we done, and who
  won?". That must be answered once. Only the decision is serialized; the
  candidates never contend.
* Concurrency is bounded everywhere by semaphores. Unlimited task creation is
  how an async service turns a spike into an outage.
* Correctness never depends on parallelism: with one worker the jobs simply
  queue behind each other and the run completes the same way.

The default coordinator is in-process, which is correct for one orchestrator.
Running several replicas against one project requires the distributed-lock
implementation — a substitution, not a rewrite, because it is a port.

---

## 10. Cancellation

Cancelling marks the run, drops its queued jobs, cancels its candidates and
releases its workspaces. In-flight work stops being renewable, so a worker
learns of the cancellation at its next lease renewal.

It is idempotent, and a terminal run refuses it rather than pretending.

---

## 11. Restarting the orchestrator

Nothing durable lives in the orchestrator. Runs, plans, candidates, jobs,
reviews, tool results and events are all in PostgreSQL; the queue and registry
are in Redis. A restart loses only the *intent to act*, which
`resume_active_runs` rebuilds by re-examining every non-terminal run.

This is covered end to end in `tests/e2e/test_full_workflow.py`.
