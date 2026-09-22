/**
 * The control plane, watched live.
 *
 * One rule shapes this file: nothing on screen is inferred. Every number comes
 * from the API, and where the API says "did not run" the interface says so too
 * rather than showing a comforting zero — the whole reason this viewer exists
 * is that a green dashboard over a broken pipeline is worse than no dashboard.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type Candidate,
  type ContextManifest,
  type Project,
  type Review,
  type Run,
  type Worker,
} from "./api";
import { useRunStream } from "./useRunStream";
import { Candidates } from "./components/Candidates";
import { ContextTree } from "./components/ContextTree";
import { Fleet } from "./components/Fleet";
import { Timeline } from "./components/Timeline";

const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const { events, state } = useRunStream(runId);

  const fail = (exc: unknown) => setError(exc instanceof Error ? exc.message : String(exc));

  useEffect(() => {
    api.projects().then(setProjects).catch(fail);
  }, []);

  // The fleet is polled rather than streamed: worker events exist, but a fleet
  // that is only correct while a run is open would be wrong exactly when you
  // most want to look at it — before starting one.
  useEffect(() => {
    const read = () => api.workers().then(setWorkers).catch(() => undefined);
    read();
    const timer = setInterval(read, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!projectId) return;
    api.runs(projectId).then(setRuns).catch(fail);
  }, [projectId]);

  const refreshRun = useCallback(async () => {
    if (!runId) return;
    try {
      const [detail, cands, revs] = await Promise.all([
        api.run(runId),
        api.candidates(runId),
        api.reviews(runId),
      ]);
      setRun(detail);
      setCandidates(cands);
      setReviews(revs);
      if (projectId) setRuns(await api.runs(projectId));
    } catch (exc) {
      fail(exc);
    }
  }, [runId, projectId]);

  useEffect(() => {
    void refreshRun();
  }, [refreshRun]);

  // Every frame is a hint that something changed. Re-reading is cheap and it
  // keeps one source of truth: the stream says *when*, the API says *what*.
  useEffect(() => {
    if (events.length === 0) return;
    void refreshRun();
  }, [events.length, refreshRun]);

  const manifests = useMemo(
    () =>
      events
        .filter((e) => e.name === "context.selected")
        .map((e) => e.payload as unknown as ContextManifest),
    [events],
  );
  const latestManifest = manifests.length > 0 ? manifests[manifests.length - 1] : null;

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      await refreshRun();
    } catch (exc) {
      fail(exc);
    } finally {
      setBusy(false);
    }
  };

  const startRun = async (objective: string, count: number) => {
    if (!projectId) return;
    await act(async () => {
      const created = await api.createRun(projectId, {
        objective,
        candidate_count: count,
      });
      setRunId(created.id);
    });
  };

  const tokens = run ? run.input_tokens + run.output_tokens : 0;

  return (
    <div className="app">
      <header className="top">
        <h1>Agentic control plane</h1>
        <div className="top-right">
          <span className={`badge ${workers.length ? "ok" : "bad"}`}>
            {workers.length} worker{workers.length === 1 ? "" : "s"}
          </span>
        </div>
      </header>

      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error}
        </div>
      )}

      <div className="layout">
        <aside className="side">
          <section className="panel">
            <header className="panel-head">
              <h3>Projects</h3>
            </header>
            <ul className="list">
              {projects.map((project) => (
                <li
                  key={project.id}
                  className={project.id === projectId ? "active" : ""}
                  onClick={() => {
                    setProjectId(project.id);
                    setRunId(null);
                    setRun(null);
                  }}
                >
                  <strong>{project.name}</strong>
                  <small>
                    {project.toolchain.language} · build{" "}
                    {project.toolchain.build_command ?? "none"} · test{" "}
                    {project.toolchain.test_command ?? "none"}
                  </small>
                </li>
              ))}
              {projects.length === 0 && <li className="muted">No project yet.</li>}
            </ul>
          </section>

          {projectId && (
            <section className="panel">
              <header className="panel-head">
                <h3>Runs</h3>
              </header>
              <NewRun onStart={startRun} busy={busy} />
              <ul className="list">
                {runs.map((r) => (
                  <li
                    key={r.id}
                    className={r.id === runId ? "active" : ""}
                    onClick={() => setRunId(r.id)}
                  >
                    <span className={`pill ${r.status.toLowerCase()}`}>{r.status}</span>
                    <small>{r.objective}</small>
                  </li>
                ))}
                {runs.length === 0 && <li className="muted">No run yet.</li>}
              </ul>
            </section>
          )}

          <Fleet workers={workers} />
        </aside>

        <main className="main">
          {!run && <p className="muted pad">Pick a run, or start one.</p>}

          {run && (
            <>
              <section className="panel run-head">
                <div className="run-title">
                  <span className={`pill big ${run.status.toLowerCase()}`}>{run.status}</span>
                  <h2>{run.objective}</h2>
                </div>
                <div className="run-meta">
                  <span>{tokens.toLocaleString()} tokens</span>
                  <span>{run.candidate_count} candidate(s)</span>
                  <span>{run.repair_iterations} repair(s)</span>
                </div>
                {run.failure_reason && <p className="warn">{run.failure_reason}</p>}

                <div className="actions">
                  {!TERMINAL.has(run.status) && run.status !== "AWAITING_APPROVAL" && (
                    <button
                      disabled={busy}
                      onClick={() =>
                        void act(() => api.cancelRun(run.id, "cancelled from the viewer"))
                      }
                    >
                      Cancel run
                    </button>
                  )}
                  {run.status === "AWAITING_APPROVAL" && (
                    <Approval
                      busy={busy}
                      onApprove={() => act(() => api.approveRun(run.id))}
                      onReject={(reason) => act(() => api.rejectRun(run.id, reason))}
                    />
                  )}
                </div>
              </section>

              {latestManifest && <ContextTree manifest={latestManifest} />}
              <Candidates runId={run.id} candidates={candidates} reviews={reviews} />
              <Timeline events={events} state={state} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function NewRun({
  onStart,
  busy,
}: {
  onStart: (objective: string, count: number) => Promise<void>;
  busy: boolean;
}) {
  const [objective, setObjective] = useState("");
  const [count, setCount] = useState(1);

  return (
    <form
      className="new-run"
      onSubmit={(e) => {
        e.preventDefault();
        if (objective.trim()) void onStart(objective.trim(), count);
      }}
    >
      <input
        value={objective}
        onChange={(e) => setObjective(e.target.value)}
        placeholder="What should the agents do?"
      />
      <div className="row">
        <label>
          candidates
          <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
            {[1, 2, 3].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={busy || !objective.trim()}>
          Start
        </button>
      </div>
      {count > 1 && (
        <small className="warn-inline">
          {count}× the GPU cost: every candidate is a full generation.
        </small>
      )}
    </form>
  );
}

function Approval({
  busy,
  onApprove,
  onReject,
}: {
  busy: boolean;
  onApprove: () => Promise<void>;
  onReject: (reason: string) => Promise<void>;
}) {
  const [reason, setReason] = useState("");
  return (
    <div className="approval">
      <p>
        Reviewed and waiting. Approving merges this patch into the project
        repository.
      </p>
      <button className="primary" disabled={busy} onClick={() => void onApprove()}>
        Approve and merge
      </button>
      <div className="row">
        <input
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="Why not? The coder gets this."
        />
        <button disabled={busy || !reason.trim()} onClick={() => void onReject(reason.trim())}>
          Reject
        </button>
      </div>
    </div>
  );
}
