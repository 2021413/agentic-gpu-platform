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

GPU workers register themselves at runtime and may join or disappear during a
run; the orchestrator schedules at the job level and never binds a project to a
fixed pool. Documentation lives in [`docs/`](docs/).
