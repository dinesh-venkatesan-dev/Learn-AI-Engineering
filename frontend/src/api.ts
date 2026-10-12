import type { Catalog, LabResult, ModulePage, Report, Scorecard, StepResult, SyllabusItem } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}
const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });

export const getCatalog = () => request<Catalog>("/api/catalog");
export const getSyllabus = (track: string) => request<SyllabusItem[]>(`/api/syllabus/${track}`);
export const getModule = (track: string, id: string) => request<ModulePage>(`/api/modules/${track}/${id}`);
export const runLab = (id: string, inputs: Record<string, unknown>) => post<LabResult>(`/api/labs/${id}/run`, inputs);
export const getLastLab = (id: string) => request<LabResult | null>(`/api/labs/${id}/last`);
export const evaluateIndex = (k = 5) => post<Scorecard>(`/api/rag/evaluate?k=${k}`);
export const getScorecards = () => request<Scorecard[]>("/api/rag/scorecards");
export const getLastRun = () => request<{ steps: StepResult[]; report: Report;
  settings: { chunk_size: number; chunk_overlap: number } } | null>("/api/rag/ingest/last");

export interface IngestOptions { chunkSize: number; chunkOverlap: number; force: boolean }

/** Run ingestion; each finished step arrives as it happens. Returns a function that stops listening. */
export function runIngestion(options: IngestOptions, handlers: {
  onStep: (result: StepResult) => void; onDone: (report: Report) => void; onError: (message: string) => void;
}): () => void {
  const params = new URLSearchParams({ chunk_size: String(options.chunkSize), chunk_overlap: String(options.chunkOverlap),
                                       force: String(options.force) });
  const source = new EventSource(`/api/rag/ingest?${params}`);
  let finished = false;
  source.addEventListener("step", (e) => handlers.onStep(JSON.parse((e as MessageEvent).data)));
  source.addEventListener("done", (e) => { finished = true; source.close(); handlers.onDone(JSON.parse((e as MessageEvent).data)); });
  source.addEventListener("error", (e) => {
    const data = (e as MessageEvent).data;
    source.close();
    if (!finished) handlers.onError(data ? JSON.parse(data).message : "The connection to the server was lost. Is it running?");
  });
  return () => source.close();
}
