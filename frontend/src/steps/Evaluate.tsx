// The Evaluate step: score the live index on the evaluation questions, and keep every scorecard to compare changes.
import { useEffect, useState } from "react";
import { evaluateIndex, getScorecards } from "../api";
import { Empty, Pills, Stat } from "../components/ui";
import { ViewBlock } from "../components/Views";
import { formatMetric } from "../format";
import type { MetricSpec, Scorecard } from "../types";

const METRICS: MetricSpec[] = [
  { key: "hit@1", label: "hit@1", better: "higher", format: "ratio", help: "Share of questions whose first chunk contains the answer" },
  { key: "hit@5", label: "hit@5", better: "higher", format: "ratio", help: "Share of questions with a relevant chunk in the top 5: can the model answer at all?" },
  { key: "precision@5", label: "P@5", better: "higher", format: "ratio", help: "Share of the 5 returned chunks that are relevant (the rest is noise)" },
  { key: "recall@5", label: "R@5", better: "higher", format: "ratio", help: "Share of ALL relevant chunks in the index that made the top 5" },
  { key: "ndcg@5", label: "nDCG@5", better: "higher", format: "ratio", help: "Ranking quality: relevant chunks count more near the top; 1 = perfect order" },
  { key: "map@5", label: "MAP@5", better: "higher", format: "ratio", help: "Mean average precision: precision at each relevant rank, averaged" },
  { key: "mrr", label: "MRR", better: "higher", format: "ratio", help: "Mean of 1/rank of the first relevant chunk (0 if none): rewards ranking it high" },
  { key: "coverage", label: "coverage", better: "higher", format: "ratio", help: "Share of questions whose source document is in the index at all" },
  { key: "answerable", label: "answerable", better: "higher", format: "ratio", help: "Share of questions where some indexed chunk contains the whole answer" },
  { key: "latency_p50_ms", label: "p50", better: "lower", format: "ms", help: "Median search latency" },
  { key: "latency_p95_ms", label: "p95", better: "lower", format: "ms", help: "95th percentile search latency: the slow tail users notice" },
];

let latest: Scorecard | null = null; // the last full result (with per-question detail), kept while the app is open

export function EvaluateView() {
  const [history, setHistory] = useState<Scorecard[]>([]);
  const [current, setCurrent] = useState<Scorecard | null>(latest);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [k, setK] = useState(5);

  useEffect(() => { getScorecards().then(setHistory, () => setHistory([])); }, []);

  async function run() {
    setBusy(true); setError(null);
    try {
      latest = await evaluateIndex(k);
      setCurrent(latest);
      setHistory(await getScorecards());
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  }

  const shown = current ?? history[history.length - 1] ?? null;
  const previous = history.length >= 2 ? history[history.length - 2] : null;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4 rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <Pills label="top k" items={[3, 5, 10].map((n) => ({ id: n, label: String(n) }))} value={k} onChange={setK} />
        <span className="text-sm text-zinc-500">Scores whatever is in the index now: run ingestion first after changing settings.</span>
        <button onClick={run} disabled={busy} className="ml-auto rounded-xl bg-sky-600 px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-sky-700 disabled:opacity-50">
          {busy ? "Scoring…" : "▶ Run evaluation"}</button>
      </div>
      {error && <div className="rounded-lg bg-rose-50 px-4 py-2 text-sm text-rose-800 dark:bg-rose-500/10 dark:text-rose-200">{error}</div>}
      {!shown && <Empty>Press <b>▶ Run evaluation</b> to score retrieval on the evaluation questions. This makes no model calls.</Empty>}

      {shown && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label={`hit@${shown.k}`} value={formatMetric(shown[`hit@${shown.k}`], "ratio")} tone="emerald" />
            <Stat label={`precision@${shown.k}`} value={formatMetric(shown[`precision@${shown.k}`], "ratio")} tone="sky" />
            <Stat label={`recall@${shown.k}`} value={formatMetric(shown[`recall@${shown.k}`], "ratio")} tone="sky" />
            <Stat label={`nDCG@${shown.k}`} value={formatMetric(shown[`ndcg@${shown.k}`], "ratio")} tone="sky" />
            <Stat label="MRR" value={formatMetric(shown.mrr, "ratio")} tone="sky" />
            <Stat label="coverage" value={formatMetric(shown.coverage, "ratio")} tone={shown.coverage < 1 ? "amber" : "zinc"} />
            <Stat label="answerable" value={formatMetric(shown.answerable, "ratio")} tone={shown.answerable < shown.coverage ? "amber" : "zinc"} />
            <Stat label="latency p95" value={formatMetric(shown.latency_p95_ms, "ms")} />
          </div>
          {previous && current && <Delta now={history[history.length - 1]} before={previous} />}
          {shown.coverage < 1 && (
            <ViewBlock view={{ type: "callout", tone: "warning", title: "Some answers can't be found at all", text:
              `Coverage is ${formatMetric(shown.coverage, "ratio")}: questions whose source document isn't in the index (scanned PDFs, images, Office files) can never be answered, whatever the search method. Module 2 (loading every format) and Module 3 (OCR) raise this ceiling.` }} />
          )}
          {shown.answerable < shown.coverage && (
            <ViewBlock view={{ type: "callout", tone: "info", title: "Some answers were split by chunking", text:
              `${formatMetric(shown.coverage, "percent")} of the questions have their file in the index, but only ${formatMetric(shown.answerable, "percent")} have a chunk that holds the whole answer. The rest were cut apart by the splitter: no search method can return them whole. Module 4 (chunking) measures how to fix that.` }} />
          )}
          <ViewBlock view={{ type: "bars", title: `By question kind, top ${shown.k}`, unit: "", max: 1,
            items: Object.entries(shown.by_kind).map(([kind, v]) => ({ label: `${kind} (${v.questions})`, values: { hit: v.hit, precision: v.precision, recall: v.recall, nDCG: v.ndcg } })) }} />
          <ViewBlock view={{ type: "table", title: "By source document", columns: ["source", "questions", "found", "hit", "precision", "recall", "in index?"],
            rows: Object.entries(shown.by_source).sort((a, b) => a[1].hit - b[1].hit)
              .map(([s, v]) => [s, v.questions, v.found, v.hit.toFixed(2), v.precision.toFixed(2), v.recall.toFixed(2), v.reachable ? "yes" : "✗ no"]) }} />
        </>
      )}

      {current?.results && <Questions card={current} />}

      {history.length > 0 && (
        <ViewBlock view={{ type: "scorecard", title: "Scorecard history: every evaluation of this index", label_key: "run", metrics: METRICS,
          rows: history.map((h, i) => ({ ...h, run: `#${i + 1} · chunk ${h.config.chunk_size}/${h.config.chunk_overlap}` })) }} />
      )}
    </div>
  );
}

function Delta({ now, before }: { now: Scorecard; before: Scorecard }) {
  const changed = METRICS.filter((m) => m.better !== "none" && typeof now[m.key] === "number" && now[m.key] !== before[m.key]);
  if (!changed.length) return <ViewBlock view={{ type: "callout", tone: "info", title: "Same as the previous run", text: "No metric changed. Change a setting, re-run ingestion, then evaluate again." }} />;
  const lines = changed.map((m) => {
    const a = before[m.key] as number, b = now[m.key] as number;
    const good = m.better === "higher" ? b > a : b < a;
    return `- **${m.label}**: ${formatMetric(a, m.format)} → ${formatMetric(b, m.format)} ${good ? "✅" : "🔻"}`;
  });
  const cfg = (s: Scorecard) => `chunk ${s.config.chunk_size}/${s.config.chunk_overlap}, k=${s.k}`;
  return <ViewBlock view={{ type: "callout", tone: "decision", title: `vs the previous run (${cfg(before)} → ${cfg(now)})`, text: lines.join("\n") }} />;
}

function Questions({ card }: { card: Scorecard }) {
  const [filter, setFilter] = useState<"missed" | "found" | "all">("missed");
  const results = card.results ?? [];
  const rows = results.filter((r) => filter === "all" || (filter === "found") === (r.rank !== null));
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-base font-semibold">Every question</h3>
        <Pills value={filter} onChange={setFilter} items={[
          { id: "missed", label: `Missed (${results.filter((r) => r.rank === null).length})` },
          { id: "found", label: `Found (${results.filter((r) => r.rank !== null).length})` }, { id: "all", label: "All" }]} />
      </div>
      <div className="space-y-2">{rows.map((r) => (
        <details key={r.id} className="rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
          <summary className="flex cursor-pointer items-center gap-3 px-4 py-2.5 text-sm">
            <span className={`w-14 shrink-0 rounded px-1.5 text-center font-mono text-xs font-semibold ${r.rank === null ? "bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-200" : "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-200"}`}>
              {r.rank === null ? "miss" : `#${r.rank}`}</span>
            <span className="flex-1">{r.question}</span>
            <span className="rounded bg-zinc-100 px-1.5 text-xs text-zinc-500 dark:bg-zinc-800">{r.kind}</span>
            {!r.reachable && <span className="rounded bg-amber-100 px-1.5 text-xs text-amber-800 dark:bg-amber-400/15 dark:text-amber-200">source not indexed</span>}
          </summary>
          <div className="space-y-2 border-t border-zinc-100 px-4 py-3 dark:border-zinc-800">
            <div className="text-xs text-zinc-500">Answer lives in <span className="font-mono">{r.source}</span> · {r.relevant_in_index} relevant chunk{r.relevant_in_index === 1 ? "" : "s"} in the index · precision {r.precision.toFixed(2)} · recall {r.recall.toFixed(2)}. What search returned:</div>
            {r.top.map((t, i) => (
              <div key={i} className={`rounded-lg p-2 text-sm ${t.relevant ? "bg-emerald-50 dark:bg-emerald-500/10" : "bg-zinc-50 dark:bg-zinc-950"}`}>
                <div className="font-mono text-xs text-zinc-500">#{i + 1} {t.source} {t.relevant && "✓"}</div>
                <div className="line-clamp-2 text-zinc-600 dark:text-zinc-400">{t.text}</div>
              </div>))}
          </div>
        </details>))}</div>
    </div>
  );
}
