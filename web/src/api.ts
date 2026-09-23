/**
 * Typed client for the control plane.
 *
 * Every shape here mirrors a response model in `src/interfaces/api/schemas`.
 * They are written out rather than inferred because a field that quietly
 * changes name is exactly the class of defect this project has spent its life
 * chasing: the two sides must be readable side by side.
 */

const BASE = import.meta.env.VITE_API_URL ?? "";

export type RunStatus =
  | "CREATED" | "PLANNING" | "PLAN_READY" | "CODING" | "VALIDATING"
  | "REVIEWING" | "REPAIRING" | "AWAITING_APPROVAL" | "COMPLETED"
  | "FAILED" | "CANCELLING" | "CANCELLED";

export interface Toolchain {
  language: string;
  build_command: string | null;
  test_command: string | null;
  static_analysis_command: string | null;
  install_command: string | null;
  working_subdirectory: string | null;
}

export interface Project {
  id: string;
  name: string;
  repository_url: string | null;
  default_branch: string;
  language: string;
  created_at: string;
  toolchain: Toolchain;
}

export interface Run {
  id: string;
  project_id: string;
  status: RunStatus;
  objective: string;
  candidate_count: number;
  repair_iterations: number;
  selected_candidate_id: string | null;
  input_tokens: number;
  output_tokens: number;
  failure_kind: string | null;
  failure_reason: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface Candidate {
  id: string;
  run_id: string;
  index: number;
  status: string;
  viable: boolean;
  build_passed: boolean | null;
  tests_passed: boolean | null;
  validation_summary: string;
  changed_files: string[];
  total_churn: number;
  review_verdict: string | null;
  coder_iterations: number;
  repair_iterations: number;
  worker_id: string | null;
  summary: string;
  uncertainties: string[];
}

export interface CandidatePatch {
  candidate_id: string;
  run_id: string;
  index: number;
  diff: string;
  changed_files: string[];
  total_churn: number;
  base_revision: string | null;
}

export interface ReviewFinding {
  summary: string;
  severity: string;
  file: string | null;
  line: number | null;
  repair_instruction: string | null;
}

export interface Review {
  id: string;
  run_id: string;
  candidate_id: string;
  verdict: string;
  iteration: number;
  summary: string;
  findings: ReviewFinding[];
  created_at: string;
}

export interface Worker {
  id: string;
  status: string;
  model_id: string;
  endpoint: string;
  context_length: number;
  active_jobs: number;
  /** The API calls this `capacity`, not `max_concurrency`. */
  capacity: number;
  supported_roles: string[];
  gpu_type: string | null;
  gpu_count: number;
  registered_at: string;
  last_heartbeat_at: string | null;
}

/** One frame of the run stream. `name` is the domain event name. */
export interface RunEvent {
  sequence: number;
  name: string;
  occurred_at: string;
  payload: Record<string, unknown>;
}

/** The `context.selected` payload: what one inference was actually shown. */
export interface ContextManifest {
  role: string;
  candidate_id: string | null;
  files: Record<string, number>;
  tree: string[];
  notes: string[];
  estimated_tokens: number;
  budget_tokens: number;
}

export class ApiError extends Error {
  constructor(readonly status: number, readonly detail: string, readonly code?: string) {
    super(detail);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    // The API answers RFC 9457 problem documents; surfacing `detail` is what
    // turns "500" into something a human can act on.
    let detail = `${response.status} ${response.statusText}`;
    let code: string | undefined;
    try {
      const problem = await response.json();
      detail = problem.detail ?? problem.title ?? detail;
      code = problem.code;
    } catch {
      /* not a problem document; the status line is all we have */
    }
    throw new ApiError(response.status, detail, code);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  projects: () => request<Project[]>("/v1/projects"),
  project: (id: string) => request<Project>(`/v1/projects/${id}`),
  createProject: (body: {
    name: string;
    local_path: string;
    default_branch: string;
    toolchain: { language: string; build_command: string | null; test_command: string | null };
  }) => request<Project>("/v1/projects", { method: "POST", body: JSON.stringify(body) }),

  /**
   * One page of a project's runs, newest first.
   *
   * The endpoint takes `limit` (1..200) and `offset`, and that is all: its
   * `RunListResponse` carries the rows and no total. So the caller can walk the
   * history but can never say "50 of 137" — only "50, and the page came back
   * full, so there are more". The viewer says exactly that rather than invent
   * the total it was not given.
   */
  runs: (projectId: string, page: { limit: number; offset: number }) =>
    request<{ runs: Run[] }>(
      `/v1/projects/${projectId}/runs?limit=${page.limit}&offset=${page.offset}`,
    ).then((r) => r.runs),
  run: (id: string) =>
    request<{ run?: Run } & Run>(`/v1/runs/${id}`).then((body) => body.run ?? body),
  createRun: (projectId: string, body: { objective: string; candidate_count: number }) =>
    request<Run>(`/v1/projects/${projectId}/runs`, { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (id: string, reason: string) =>
    request<Run>(`/v1/runs/${id}/cancel`, { method: "POST", body: JSON.stringify({ reason }) }),
  approveRun: (id: string) => request<Run>(`/v1/runs/${id}/approve`, { method: "POST" }),
  rejectRun: (id: string, reason: string) =>
    request<Run>(`/v1/runs/${id}/reject`, { method: "POST", body: JSON.stringify({ reason }) }),

  candidates: (runId: string) =>
    request<{ candidates: Candidate[] }>(`/v1/runs/${runId}/candidates`).then((r) => r.candidates),
  diff: (runId: string, candidateId: string) =>
    request<CandidatePatch>(`/v1/runs/${runId}/candidates/${candidateId}/diff`),
  reviews: (runId: string) =>
    request<{ reviews: Review[] }>(`/v1/runs/${runId}/reviews`).then((r) => r.reviews),

  workers: () => request<{ workers: Worker[] }>("/v1/workers").then((r) => r.workers),
};

/** Where the run stream lives. EventSource needs a URL, not a fetch. */
export const runEventsUrl = (runId: string) => `${BASE}/v1/runs/${runId}/events`;
