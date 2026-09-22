/**
 * The GPU fleet, as the scheduler sees it.
 *
 * `context_length` is shown because it is the number that decides whether a
 * prompt is schedulable at all, and because it spent this project's life
 * disagreeing with what the engine actually served.
 */

import type { Worker } from "../api";

export function Fleet({ workers }: { workers: Worker[] }) {
  return (
    <section className="panel fleet">
      <header className="panel-head">
        <h3>Fleet</h3>
        <span className={`badge ${workers.length === 0 ? "bad" : "ok"}`}>
          {workers.length} worker{workers.length === 1 ? "" : "s"}
        </span>
      </header>
      {workers.length === 0 && (
        <p className="warn">
          No worker is registered. A run created now fails with “no compatible
          worker is available”.
        </p>
      )}
      <ul className="workers">
        {workers.map((worker) => {
          const load = worker.capacity
            ? Math.round((worker.active_jobs / worker.capacity) * 100)
            : 0;
          return (
            <li key={worker.id}>
              <div className="worker-head">
                <span className={`pill ${worker.status.toLowerCase()}`}>{worker.status}</span>
                <span className="model">{worker.model_id}</span>
              </div>
              <div className="worker-meta">
                <span>{worker.context_length.toLocaleString()} tk window</span>
                <span>
                  {worker.active_jobs}/{worker.capacity} busy
                </span>
                <span className="roles">{worker.supported_roles.join(" · ")}</span>
              </div>
              <div className="load">
                <span style={{ width: `${load}%` }} />
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
