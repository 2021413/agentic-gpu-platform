/**
 * Ages, shared.
 *
 * Both the timeline and the fleet answer "how long ago", and they must answer
 * it the same way: two panels on one screen disagreeing about what "5m ago"
 * rounds to would be worse than either rounding on its own.
 */

import { useEffect, useState } from "react";

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
export function ago(iso: string, now: number): string {
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
export function useNow(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  return now;
}
