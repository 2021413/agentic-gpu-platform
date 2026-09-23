/**
 * The GPU fleet, as the scheduler sees it.
 *
 * `context_length` is shown because it is the number that decides whether a
 * prompt is schedulable at all, and because it spent this project's life
 * disagreeing with what the engine actually served.
 *
 * A worker that is gone is drawn as gone. Presenting an OFFLINE worker with the
 * same weight, the same detail rows and the same load meter as a READY one is
 * the green-dashboard failure this viewer exists to prevent: the panel would be
 * saying "two workers, both idle, eight slots free" about a fleet that can take
 * four jobs.
 */

import type { Worker } from "../api";
import { ago, useNow } from "../time";

/** The statuses the scheduler will actually hand a job to. */
const LIVE = new Set(["READY", "BUSY"]);
/** The statuses that mean the worker is gone rather than merely busy or new. */
const GONE = new Set(["OFFLINE", "UNHEALTHY"]);

export const isLive = (worker: Worker) => LIVE.has(worker.status);
export const isGone = (worker: Worker) => GONE.has(worker.status);

/** Slots the scheduler could fill right now. Only live workers have any. */
export function freeSlots(workers: Worker[]): number {
  return workers
    .filter(isLive)
    .reduce((total, w) => total + Math.max(0, w.capacity - w.active_jobs), 0);
}

/** The roles at least one live worker advertises. A run needs all three. */
export function liveRoles(workers: Worker[]): Set<string> {
  return new Set(workers.filter(isLive).flatMap((w) => w.supported_roles));
}

/** Past this many slots the cells are thinner than the gaps between them. */
const MAX_SEGMENTS = 24;

/**
 * A meter, not a progress bar: it reports a level, it does not promise an end.
 *
 * It is drawn as one cell per slot because the question is "how many of the
 * four are taken", which a segmented row answers without reading the digits
 * beside it — and because a continuous track sitting at 0% is a full-width
 * hairline, indistinguishable from the rules used as separators everywhere else
 * in this stylesheet. A gauge that reads as a divider when idle is worse than
 * no gauge.
 */
function Load({ active, capacity }: { active: number; capacity: number }) {
  // Past MAX_SEGMENTS the cells stop being countable, and a meter you cannot
  // read is worth less than the "n/m busy" line already above it, so the meter
  // stands down rather than degrading back into a hairline.
  if (capacity <= 0 || capacity > MAX_SEGMENTS) return null;
  return (
    <div className="load" role="img" aria-label={`${active} of ${capacity} slots busy`}>
      {Array.from({ length: capacity }, (_, slot) => (
        <span key={slot} className={slot < active ? "slot on" : "slot"} />
      ))}
    </div>
  );
}

export function Fleet({ workers }: { workers: Worker[] }) {
  const now = useNow();

  return (
    <section className="panel fleet">
      <header className="panel-head">
        <h3>Fleet</h3>
      </header>
      {workers.length === 0 && (
        <p className="warn">
          No worker is registered. A run created now fails with “no compatible
          worker is available”.
        </p>
      )}
      <ul className="workers">
        {workers.map((worker) => {
          const gone = isGone(worker);
          return (
            <li key={worker.id} className={gone ? "gone" : ""}>
              <div className="worker-head">
                <span className={`pill ${worker.status.toLowerCase()}`}>{worker.status}</span>
                <span className="model">{worker.model_id}</span>
              </div>
              <div className="worker-meta">
                {gone ? (
                  /* Its window and its capacity are not facts about the fleet any
                     more, they are facts about a machine that stopped answering,
                     so the one thing worth reading is when it stopped. */
                  <span>
                    no capacity ·{" "}
                    {worker.last_heartbeat_at
                      ? `last heartbeat ${ago(worker.last_heartbeat_at, now)}`
                      : "never sent a heartbeat"}
                  </span>
                ) : (
                  <>
                    <span>{worker.context_length.toLocaleString()} tk window</span>
                    <span>
                      {worker.active_jobs}/{worker.capacity} busy
                    </span>
                    <span className="roles">{worker.supported_roles.join(" · ")}</span>
                  </>
                )}
              </div>
              {!gone && <Load active={worker.active_jobs} capacity={worker.capacity} />}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
