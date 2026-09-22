/**
 * The run as it happens.
 *
 * Events arrive in order and are grouped into swimlanes: the run's own
 * transitions on the left, and one lane per candidate, because best-of-N is
 * the thing that is hardest to follow from a flat log.
 */

import { useEffect, useState } from "react";
import type { RunEvent } from "../api";

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

/**
 * How long ago, in the coarsest unit that is still true.
 *
 * A clock time answers "when did that happen"; watching a run you are asking
 * "how long has it been stuck", and that is a subtraction you should not have
 * to do in your head. The unit is never smaller than a second because nothing
 * here is worth re-reading faster than that, and a future timestamp — clock
 * skew between the API host and this browser — is clamped to "now" rather than
 * rendered as a negative age.
 */
function ago(iso: string, now: number): string {
  const seconds = Math.floor(Math.max(0, now - new Date(iso).getTime()) / 1000);
  if (seconds < 1) return "now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/**
 * One clock for the whole list rather than a timer per row: the ages all move
 * together anyway, and a hundred intervals in a panel left open for an hour is
 * a cost paid for nothing.
 */
function useNow(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  return now;
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
