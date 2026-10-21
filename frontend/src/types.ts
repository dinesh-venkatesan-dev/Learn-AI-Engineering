// Shapes of what the API returns.

export interface CodeRef {
  module: string; symbol: string; why: string; file: string; abs_path: string; start: number; source: string;
  notes: { line: number; text: string }[];
}

export interface Step {
  id: StepId; title: string; icon: string; tagline: string; explain: string; code: CodeRef[]; production: string[];
  lab?: string; lab_entry?: LabEntry;
}
export type StepId = string;

export interface Lesson {
  id: string; title: string; subtitle: string; summary: string; learn: string[];
  layout: { path: string; role: string }[]; commands: { label: string; cmd: string }[]; steps: Step[];
}

export interface Doc { source: string; page: number | null; chars: number; text?: string }

export interface StepResult<D = unknown> { step: StepId; ms: number; stats: Record<string, number | string>; data: D }

export interface LoadData { documents: Doc[]; failed: { source: string; error: string }[]; unsupported: string[] }
export interface CleanData { documents: (Doc & { before: string; after: string })[] }
export interface ChunkData { documents: Doc[]; chunks: (Doc & { start: number; index: number })[] }
export interface EmbedData { vector: number[]; text: string }
export interface StoreData {
  table: string; rows: number; size: string; records: number;
  sample: { langchain_id: string; content: string; source: string; page: number; file_type: string;
            langchain_metadata: Record<string, unknown>; dims: number; first_numbers: number[] }[];
}

export interface Report { files: number; unsupported: string[]; failed: unknown[]; documents: number; chunks: number; added: number;
  updated: number; skipped: number; deleted: number; seconds: number }

export interface SearchHit { text: string; metadata: Record<string, unknown>; distance: number }

// ── The platform: catalogue, modules, labs ──────────────────────────────────────
export interface CatalogModule { id: string; title: string; status: "built" | "preview" | "planned"; summary: string; from: string[];
  except?: string[]; capstones?: string[]; notes?: string; items: number; covered: number; stars: number; why: string }
export interface CatalogTrack { id: string; title: string; icon: string; summary: string; modules: CatalogModule[]; items: number; covered: number }
export interface Catalog { tracks: CatalogTrack[]; items: number; covered: number; scale: Record<string, string> }

export interface SyllabusItem { id: string; lesson: string; part: string; course: string; lesson_title: string; text: string;
  module?: string; module_title?: string; covered?: boolean; stars?: number }

export interface LabInput { name: string; label: string; kind: "text" | "textarea" | "number" | "select" | "bool"; default: unknown;
  options: string[] | null; min: number | null; max: number | null; help: string }

export interface LabEntry {
  lab: string; id: string; title: string; icon: string; explain: string; covers: string[]; uses_model: boolean; calls: string;
  inputs: LabInput[]; code: CodeRef[]; covered_items: SyllabusItem[];
}

export interface ModulePage {
  track: string; id: string; title: string; summary: string; learn: string[]; flow: string; labs: LabEntry[]; stars: number; why: string;
  path: { title: string; summary: string; steps: Step[]; covered_items: SyllabusItem[]; runner?: "rag-ingestion" };
  layout: { path: string; role: string }[]; commands: { label: string; cmd: string }[];
}

export type View =
  | { type: "markdown"; text: string }
  | { type: "callout"; tone: "info" | "success" | "warning" | "decision"; title: string; text: string }
  | { type: "table"; title: string; columns: string[]; rows: unknown[][] }
  | { type: "scorecard"; title: string; label_key: string; rows: Record<string, unknown>[]; metrics: MetricSpec[] }
  | { type: "results"; title: string; score_label: string; hits: { source: string; text: string; score: number; relevant?: boolean | null; matched?: string[] }[] }
  | { type: "bars"; title: string; unit: string; max: number | null; items: { label: string; values: Record<string, number> }[] }
  | { type: "answer"; title: string; text: string; meta: Record<string, unknown> }
  | { type: "columns"; title: string; children: View[] }
  | { type: "images"; title: string; items: { src: string; caption: string }[] }
  | { type: "code"; title: string; language: string; text: string }
  | { type: "highlight"; title: string; text: string; spans: [number, number][] }
  | { type: "radar"; title: string; rings: number[]; points: { x: number; y: number; label: string; text: string; rank?: number | null }[] }
  | { type: "example"; point: string; title: string; explain: string; code: string; outputs: View[] };

export interface MetricSpec { key: string; label: string; better: "higher" | "lower" | "none"; format: "ratio" | "percent" | "ms" | "int" | "text" | "usd"; help: string }

export type Example = Extract<View, { type: "example" }>;
export interface ModelMessage { role: string; text: string; name?: string; tool_calls?: { name: string; args: unknown }[] }
export interface ModelCall { model: string; messages: ModelMessage[]; reply?: ModelMessage; tokens_in?: number | null; tokens_out?: number | null; seconds?: number; error?: string }
export interface LabResult { lab: string; inputs: Record<string, unknown>; ms: number; views: View[]; examples?: Example[]; model_calls?: { calls: ModelCall[]; total: number } }

export interface Scorecard {
  at: string; config: Record<string, string | number>; k: number; questions: number; hit_reachable: number; answerable: number;
  mrr: number; coverage: number; latency_p50_ms: number; latency_p95_ms: number; [key: string]: unknown;
  by_kind: Record<string, Rates>;
  by_source: Record<string, Rates & { reachable: boolean }>;
  results?: { id: string; question: string; kind: string; source: string; rank: number | null; reachable: boolean;
              relevant_in_index: number; precision: number; recall: number;
              top: { source: string; text: string; relevant: boolean }[] }[];
}

export interface Rates { questions: number; found: number; hit: number; precision: number; recall: number; ndcg: number; mrr: number }
