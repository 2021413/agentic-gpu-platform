/**
 * The control plane, watched live.
 *
 * One rule shapes this file: nothing on screen is inferred. Every number comes
 * from the API, and where the API says "did not run" the interface says so too
 * rather than showing a comforting zero — the whole reason this viewer exists
 * is that a green dashboard over a broken pipeline is worse than no dashboard.
 */

import { useCallback, useEffect, useMemo, useState, type KeyboardEvent } from "react";
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
import { Fleet, freeSlots, isLive } from "./components/Fleet";
import { StartHere } from "./components/StartHere";
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

  // The click handler and the keyboard handler must do the same thing, so the
  // thing they do lives in one place rather than twice in the markup.
  const selectProject = (id: string) => {
    setProjectId(id);
    setRunId(null);
    setRun(null);
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
  const selected = useMemo(
    () => projects.find((p) => p.id === projectId) ?? null,
    [projects, projectId],
  );

  /*
   * The top bar reports what the fleet can do, not how many rows the workers
   * endpoint returned. "2 workers" over one READY and one OFFLINE machine is
   * the precise shape of the lie this viewer exists to refuse, so the badge
   * counts the live ones and turns amber the moment a registered worker is not
   * among them.
   */
  const live = workers.filter(isLive).length;
  const fleetTone =
    workers.length === 0 || live === 0 ? "bad" : live < workers.length ? "warn" : "ok";

  return (
    <div className="app">
      <header className="top">
        <h1>Agentic control plane</h1>
        <div className="top-right">
          <span
            className={`badge ${fleetTone}`}
            title={`${freeSlots(workers)} free slot(s) across ${live} live worker(s)`}
          >
            {live}/{workers.length} live
          </span>
        </div>
      </header>

      {error && (
        <div className="error" aria-live="polite" onClick={() => setError(null)}>
          {error}
        </div>
      )}

      <div className="layout">
        <aside className="side">
          <section className="panel">
            <header className="panel-head">
              <h3>Projects</h3>
              {/* Not a badge: a badge means a state worth reacting to, and this
                  is only how many rows the scroller holds. It is here so that a
                  list cut off at five reads as five of twelve. */}
              <span className="count">{projects.length}</span>
            </header>
            <ul className="list">
              {projects.map((project) => (
                <li
                  key={project.id}
                  className={project.id === projectId ? "active" : ""}
                  role="button"
                  tabIndex={0}
                  aria-current={project.id === projectId ? "true" : undefined}
                  onClick={() => selectProject(project.id)}
                  onKeyDown={activates(() => selectProject(project.id))}
                >
                  <strong>{project.name}</strong>
                  <small>
                    {project.toolchain.language} · build{" "}
                    {project.toolchain.build_command ?? "none"} · test{" "}
                    {project.toolchain.test_command ?? "none"}
                  </small>
                </li>
              ))}
              {projects.length === 0 && (
                <li className="muted">No project yet. Register one, then it is selectable here.</li>
              )}
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
                    role="button"
                    tabIndex={0}
                    aria-current={r.id === runId ? "true" : undefined}
                    onClick={() => setRunId(r.id)}
                    onKeyDown={activates(() => setRunId(r.id))}
                  >
                    <span className={`pill ${r.status.toLowerCase()}`}>{r.status}</span>
                    <small>{r.objective}</small>
                  </li>
                ))}
                {runs.length === 0 && (
                  <li className="muted">No run yet. Type an objective above to start the first.</li>
                )}
              </ul>
            </section>
          )}

          <Fleet workers={workers} />
        </aside>

        <main className="main">
          {!run && <StartHere projects={projects} workers={workers} project={selected} />}

          {run && (
            <>
              <section className="panel run-head">
                <div className="run-title">
                  <span className={`pill big ${run.status.toLowerCase()}`}>{run.status}</span>
                  <h2>{run.objective}</h2>
                </div>
                <div className="run-meta">
                  {/* The total is the sum of two API fields, so the split stays one
                      hover away instead of disappearing into the addition. */}
                  <div className="stat">
                    <span
                      className="stat-value"
                      title={`${run.input_tokens.toLocaleString()} in / ${run.output_tokens.toLocaleString()} out`}
                    >
                      {tokens.toLocaleString()}
                    </span>
                    <span className="stat-label">tokens</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{run.candidate_count}</span>
                    <span className="stat-label">candidates</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{run.repair_iterations}</span>
                    <span className="stat-label">repairs</span>
                  </div>
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

/**
 * A row in `.list` is a button wearing an `<li>`, so it has to answer the
 * keyboard like one. Space's default action is to scroll the panel, which would
 * move the row out from under the choice being made, so it is suppressed.
 */
function activates(onActivate: () => void) {
  return (event: KeyboardEvent<HTMLElement>) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onActivate();
  };
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
