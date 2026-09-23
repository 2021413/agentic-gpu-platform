/**
 * What the main column shows when no run is selected.
 *
 * The obvious thing to put here is "recent activity across projects", and it is
 * the thing the project's own status document names as missing. It stays
 * missing: the control plane has no endpoint that lists runs across projects —
 * `GET /v1/projects/{id}/runs` is the only listing there is — and the two ways
 * to fake one are both worse than the gap. Inventing an endpoint would put a
 * 404 on the first screen; fanning out one request per project on mount would
 * make the landing page cost O(projects) round trips and still be wrong the
 * moment a project is added.
 *
 * So this column shows what is already in hand and is worth knowing before you
 * start something expensive: whether the fleet can take the job at all, what
 * the selected project will actually run to validate a patch, and where the run
 * will stop and wait for you. It also says plainly that there is no global run
 * list, because a column that is quiet for a reason should say the reason
 * rather than let you wonder whether it is broken.
 */

import type { Project, Worker } from "../api";
import { freeSlots, isLive, liveRoles } from "./Fleet";

/** The three roles a run has to place before it can finish. */
const REQUIRED_ROLES = ["PLANNER", "CODER", "REVIEWER"];

export function StartHere({
  projects,
  workers,
  project,
}: {
  projects: Project[];
  workers: Worker[];
  project: Project | null;
}) {
  const live = workers.filter(isLive);
  const slots = freeSlots(workers);
  const covered = liveRoles(workers);
  const missing = REQUIRED_ROLES.filter((role) => !covered.has(role));

  return (
    <>
      <section className="panel">
        <header className="panel-head">
          <h3>No run selected</h3>
        </header>

        <p className="muted">
          {project
            ? `Runs for ${project.name} are listed on the left. Pick one to watch it, or start a new one.`
            : "Pick a project on the left, then one of its runs."}{" "}
          There is no cross-project list of recent runs here because the API has
          none: runs are listed per project. Nothing is being hidden.
        </p>

        {/* The same readouts as a run header, because they are read the same way:
            value first, label under it, tabular so they do not dance. */}
        <div className="run-meta">
          <div className="stat">
            <span className="stat-value">{slots}</span>
            <span className="stat-label">free slots</span>
          </div>
          <div className="stat">
            <span className="stat-value">
              {live.length}/{workers.length}
            </span>
            <span className="stat-label">live workers</span>
          </div>
          <div className="stat">
            <span className="stat-value">{projects.length}</span>
            <span className="stat-label">projects</span>
          </div>
        </div>

        {/* Only the bad cases are coloured. A green "fleet is fine" line on an
            idle screen would train the eye to ignore the one colour that has to
            mean something when it appears. */}
        {workers.length === 0 ? (
          <p className="warn">
            No worker is registered. A run created now fails with “no compatible
            worker is available”.
          </p>
        ) : missing.length > 0 ? (
          <p className="warn">
            No live worker advertises {missing.join(", ")}. A run started now
            cannot be scheduled past that stage.
          </p>
        ) : slots === 0 ? (
          <p className="warn">
            Every slot on the fleet is taken. A run started now queues until one
            frees.
          </p>
        ) : (
          <p className="muted small">
            {REQUIRED_ROLES.join(", ")} are all covered by a live worker.
          </p>
        )}

        <h4 className="sub">What happens when you start one</h4>
        <ol className="steps">
          <li>
            The objective is planned, then coded once per candidate. Each
            candidate is a full generation on the fleet, so two cost twice.
          </li>
          <li>
            Every candidate is built and tested with the project's own commands.
            A command the project does not define is reported as “did not run”,
            never as a pass.
          </li>
          <li>
            The reviewer reads the survivors. A rejected one goes back for
            repair rather than being dropped.
          </li>
          <li>
            The run then stops and waits. Nothing reaches the repository until
            you approve it here.
          </li>
        </ol>
      </section>

      {project && <Toolchain project={project} />}
    </>
  );
}

/**
 * The selected project's toolchain, in full.
 *
 * The sidebar row truncates these commands to one ellipsised line, and they are
 * the commands that decide what "validated" is worth — a project with no test
 * command produces candidates whose tests "passed" by not existing.
 */
function Toolchain({ project }: { project: Project }) {
  const t = project.toolchain;
  const rows: [string, string | null][] = [
    ["language", t.language],
    ["branch", project.default_branch],
    ["install", t.install_command],
    ["build", t.build_command],
    ["test", t.test_command],
    ["analysis", t.static_analysis_command],
    ["subdir", t.working_subdirectory],
    ["repository", project.repository_url],
  ];

  return (
    <section className="panel">
      <header className="panel-head">
        <h3>{project.name}</h3>
      </header>
      <dl className="kv">
        {rows.map(([label, value]) => (
          <div key={label} className="kv-row">
            <dt>{label}</dt>
            <dd className={value ? "" : "unset"}>{value ?? "not set"}</dd>
          </div>
        ))}
      </dl>
      {!t.test_command && (
        <p className="warn">
          No test command. Candidates from this project can only ever report
          “tests did not run”, so a review is the only evidence there is.
        </p>
      )}
    </section>
  );
}
