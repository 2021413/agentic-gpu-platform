/**
 * The live run stream.
 *
 * `EventSource` reconnects on its own and replays from `Last-Event-ID`, which
 * the API honours — so a dropped connection costs a gap of nothing rather than
 * a gap in the timeline. Events are kept in arrival order and deduplicated by
 * sequence, because the server's own contract is "a handful of duplicates,
 * never a gap".
 */

import { useEffect, useRef, useState } from "react";
import { runEventsUrl, type RunEvent } from "./api";

export type StreamState = "connecting" | "open" | "closed";

export function useRunStream(runId: string | null) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<StreamState>("closed");
  const seen = useRef<Set<number>>(new Set());

  useEffect(() => {
    setEvents([]);
    seen.current = new Set();
    if (!runId) {
      setState("closed");
      return;
    }

    setState("connecting");
    const source = new EventSource(runEventsUrl(runId));
    source.onopen = () => setState("open");
    source.onerror = () => setState("connecting");

    const handle = (raw: MessageEvent<string>) => {
      let frame: RunEvent;
      try {
        frame = JSON.parse(raw.data) as RunEvent;
      } catch {
        return; // a keepalive comment or a frame we do not understand
      }
      if (typeof frame.sequence === "number") {
        if (seen.current.has(frame.sequence)) return;
        seen.current.add(frame.sequence);
      }
      setEvents((previous) => [...previous, frame]);
    };

    // The server names each frame after its domain event, so there is no
    // single "message" type to listen for. `onmessage` catches unnamed frames;
    // the named ones need explicit listeners.
    source.onmessage = handle;
    for (const name of EVENT_NAMES) source.addEventListener(name, handle as EventListener);

    return () => {
      for (const name of EVENT_NAMES) source.removeEventListener(name, handle as EventListener);
      source.close();
      setState("closed");
    };
  }, [runId]);

  return { events, state };
}

/**
 * Every event the control plane emits.
 *
 * Listed rather than discovered because `EventSource` dispatches by name: an
 * event missing from this list arrives at no listener and silently disappears
 * from the timeline. Keep it in step with `domain/events`.
 */
export const EVENT_NAMES = [
  "run.created",
  "run.state_changed",
  "run.plan_requested",
  "run.plan_completed",
  "run.repair_requested",
  "run.awaiting_approval",
  "run.approval_rejected",
  "run.completed",
  "run.failed",
  "run.cancelled",
  "candidate.started",
  "candidate.validation_started",
  "candidate.validation_completed",
  "candidate.completed",
  "candidate.selected",
  "review.requested",
  "review.completed",
  "job.enqueued",
  "job.leased",
  "job.completed",
  "job.failed",
  "job.requeued",
  "job.lease_expired",
  "worker.registered",
  "worker.heartbeat",
  "worker.status_changed",
  "worker.draining",
  "worker.unavailable",
  "worker.deregistered",
  "context.selected",
] as const;
