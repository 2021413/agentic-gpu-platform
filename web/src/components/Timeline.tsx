/**
 * The run as it happens.
 *
 * Events arrive in order and are grouped into swimlanes: the run's own
 * transitions on the left, and one lane per candidate, because best-of-N is
 * the thing that is hardest to follow from a flat log.
 */

import type { RunEvent } from "../api";
import { ago, useNow } from "../time";

const TONE: Record<string, string> = {
  "run.failed": "bad",
  "job.failed": "bad",
  "run.cancelled": "warn",
  "job.requeued": "warn",
  "job.lease_expired": "warn",
  "run.repair_requested": "warn",
  "run.approval_rejected": "warn",
  "worker.unavailable": "bad",
  "run.completed": "good",
  "candidate.selected": "good",
  "run.awaiting_approval": "hold",
  "context.selected": "info",
};

function clock(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, { hour12: false });
}

/** One line of detail, chosen per event type rather than dumping the payload. */
function describe(event: RunEvent): string {
  const p = event.payload as Record<string, never>;
  switch (event.name) {
    case "run.state_changed":
      return `${p.previous} → ${p.current}`;
    case "run.plan_completed":
      return `${(p.task_keys as unknown as string[] | undefined)?.length ?? "?"} tasks`;
    case "candidate.validation_completed":
      return String(p.summary ?? "");
    case "context.selected": {
      const files = Object.keys((p.files ?? {}) as Record<string, number>).length;
      return `${files} file(s), ${Number(p.estimated_tokens ?? 0).toLocaleString()} tokens`;
    }
    case "review.completed":
      return String(p.verdict ?? "");
    case "job.leased":
      return `${p.job_type ?? ""} → worker ${String(p.worker_id ?? "").slice(0, 8)}`;
    case "run.failed":
      return String(p.reason ?? "");
    case "run.approval_rejected":
      return String(p.reason ?? "");
    default:
      return "";
  }
}

export function Timeline({ events, state }: { events: RunEvent[]; state: string }) {
  const now = useNow();

  return (
    <section className="panel timeline">
      <header className="panel-head">
        <h3>Timeline</h3>
        <span className={`stream ${state}`}>{state}</span>
      </header>
      {events.length === 0 && (
        <p className="muted">No event yet. Each one appears here as the run emits it.</p>
      )}
      <ol className="events">
        {events.map((event) => {
          const candidate = event.payload.candidate_id as string | undefined;
          return (
            <li key={`${event.sequence}-${event.name}`} className={TONE[event.name] ?? ""}>
              <span className="time" title={clock(event.occurred_at)}>
                {ago(event.occurred_at, now)}
              </span>
              <span className="dot" />
              <span className="name">{event.name}</span>
              {candidate && <span className="lane">#{candidate.slice(0, 6)}</span>}
              <span className="detail">{describe(event)}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
