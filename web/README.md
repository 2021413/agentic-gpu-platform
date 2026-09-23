# Viewer

A live view of the control plane: what a run is doing, what each agent was
actually shown, what the candidates produced, and — when the deployment asks
for it — the button that lets a patch land.

## Running it

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/v1` and `/health` to `http://localhost:8000`. Point it
somewhere else with `API_URL=http://host:8000 npm run dev`.

For a built viewer served from its own origin, set `VITE_API_URL` at build time
and add that origin to `CORS_ALLOW_ORIGINS` in the control plane's `.env` — the
API is closed to browsers by default, on purpose.

```bash
VITE_API_URL=https://control-plane.example npm run build
```

## What it shows

**What the agent was shown.** The repository tree with the files that actually
reached the prompt highlighted and weighted by token cost; everything else is
dimmed. The dimmed part is the point — a selection containing no code is the
defect that had agents inventing from a filename list, and here it is one
glance rather than an afternoon of logs.

**The timeline.** Every domain event as it arrives, over SSE. `EventSource`
resumes from `Last-Event-ID`, which the API honours, so a dropped connection
costs duplicates rather than a gap.

**Candidates.** Best-of-N side by side with the evidence in the order the
platform trusts it: build, tests, then the reviewer. "Did not run" renders as
`—`, never as a passing zero.

**The fleet.** Workers, their load, and the context window they advertise —
the number that decides whether a prompt is schedulable at all.

## A note on the types

`src/api.ts` declares the response shapes by hand rather than generating them.
They are checked against the API's own OpenAPI document by
`tests/api/test_viewer_contract.py`, because TypeScript only ever verified the
client against its own declarations — and the declarations were the thing that
was wrong the first time.
