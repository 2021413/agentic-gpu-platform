/**
 * Best-of-N, side by side, with the evidence that decided it.
 *
 * Deterministic results first and the reviewer's opinion last, in that order,
 * because that is the order the platform itself trusts them in.
 */

import { useState } from "react";
import { api, type Candidate, type CandidatePatch, type Review } from "../api";

function Mark({ value }: { value: boolean | null }) {
  if (value === null) return <span className="mark skip" title="did not run">—</span>;
  return value ? <span className="mark ok">✓</span> : <span className="mark bad">✗</span>;
}

export function Candidates({
  runId,
  candidates,
  reviews,
}: {
  runId: string;
  candidates: Candidate[];
  reviews: Review[];
}) {
  const [patch, setPatch] = useState<CandidatePatch | null>(null);
  const [error, setError] = useState<string | null>(null);

  const showDiff = async (candidateId: string) => {
    setError(null);
    try {
      setPatch(await api.diff(runId, candidateId));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  return (
    <section className="panel candidates">
      <header className="panel-head">
        <h3>Candidates</h3>
        <span className="badge">{candidates.length}</span>
      </header>

      {candidates.length === 0 && (
        <p className="muted">No candidate yet. One appears per coder job, as it reports.</p>
      )}

      {candidates.map((candidate) => {
        const theirs = reviews.filter((r) => r.candidate_id === candidate.id);
        return (
          <article key={candidate.id} className={`candidate ${candidate.status.toLowerCase()}`}>
            <div className="candidate-head">
              <span className="index">#{candidate.index}</span>
              <span className={`pill ${candidate.status.toLowerCase()}`}>{candidate.status}</span>
              <span className="churn">{candidate.total_churn} lines</span>
              <button onClick={() => void showDiff(candidate.id)}>diff</button>
            </div>

            <div className="evidence">
              <span>build <Mark value={candidate.build_passed} /></span>
              <span>tests <Mark value={candidate.tests_passed} /></span>
              <span>
                review{" "}
                {candidate.review_verdict ? (
                  <span className={`mark ${candidate.review_verdict === "PASS" ? "ok" : "bad"}`}>
                    {candidate.review_verdict}
                  </span>
                ) : (
                  <span className="mark skip">—</span>
                )}
              </span>
            </div>

            {candidate.summary && <p className="summary">{candidate.summary}</p>}

            {candidate.changed_files.length > 0 && (
              <ul className="files">
                {candidate.changed_files.map((file) => (
                  <li key={file}>{file}</li>
                ))}
              </ul>
            )}

            {candidate.uncertainties.length > 0 && (
              <ul className="notes">
                {candidate.uncertainties.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            )}

            {theirs.map((review) => (
              <div key={review.id} className="review">
                <span className={`pill ${review.verdict.toLowerCase()}`}>
                  round {review.iteration}: {review.verdict}
                </span>
                {review.summary && <p>{review.summary}</p>}
                {review.findings.map((finding, i) => (
                  <div key={i} className={`finding ${finding.severity.toLowerCase()}`}>
                    <span className="sev">{finding.severity}</span>
                    {finding.file && (
                      <span className="where">
                        {finding.file}
                        {finding.line ? `:${finding.line}` : ""}
                      </span>
                    )}
                    <span>{finding.summary}</span>
                    {finding.repair_instruction && (
                      <em className="fix">{finding.repair_instruction}</em>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </article>
        );
      })}

      {error && <p className="warn">{error}</p>}

      {patch && (
        <div className="overlay" onClick={() => setPatch(null)}>
          <div className="diff" onClick={(e) => e.stopPropagation()}>
            <header>
              <strong>Candidate #{patch.index}</strong>
              <span>
                {patch.changed_files.length} file(s), {patch.total_churn} lines
              </span>
              <button onClick={() => setPatch(null)}>close</button>
            </header>
            <pre>
              {patch.diff
                ? patch.diff.split("\n").map((line, i) => (
                    <span key={i} className={
                      line.startsWith("+++") || line.startsWith("---") ? "meta"
                      : line.startsWith("+") ? "add"
                      : line.startsWith("-") ? "del"
                      : line.startsWith("@@") ? "hunk"
                      : ""
                    }>{line}{"\n"}</span>
                  ))
                : "This candidate produced no patch."}
            </pre>
          </div>
        </div>
      )}
    </section>
  );
}
