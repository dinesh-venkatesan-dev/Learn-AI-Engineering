// Generic renderers for what a lab returns. Every lab in every module reuses these.
import type { MetricSpec, View } from "../types";
import { formatMetric } from "../format";
import { Snippet } from "./CodeBlock";
import { Markdown } from "./Markdown";

const SERIES = ["bg-sky-500", "bg-violet-500", "bg-amber-500", "bg-emerald-500"];
const SERIES_TEXT = ["text-sky-600 dark:text-sky-400", "text-violet-600 dark:text-violet-400", "text-amber-600 dark:text-amber-400", "text-emerald-600 dark:text-emerald-400"];

export function Views({ views }: { views: View[] }) {
  let step = 0;
  return <div className="space-y-6">{views.map((v, i) => v.type === "example"
    ? <ExampleView key={i} view={v} step={++step} />
    : <ViewBlock key={i} view={v} />)}</div>;
}

/** One learning point: what it is, the code that does it, and what that code produced in this run. */
export function ExampleView({ view, step }: { view: Extract<View, { type: "example" }>; step: number }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <header className="flex items-start gap-3 border-b border-zinc-100 px-5 py-4 dark:border-zinc-800">
        <span className="grid size-7 shrink-0 place-items-center rounded-full bg-sky-600 text-sm font-bold text-white">{step}</span>
        <h3 className="min-w-0 text-lg font-semibold leading-snug">{view.title}</h3>
      </header>
      <div className="space-y-4 px-5 py-4">
        <Markdown text={view.explain} />
        <div>
          <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-500">The code</div>
          <Snippet source={view.code} />
        </div>
        <div>
          <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-emerald-600">What it produced in this run</div>
          <div className="space-y-4">{view.outputs.map((o, i) => <ViewBlock key={i} view={o} />)}</div>
        </div>
      </div>
    </section>
  );
}

function Title({ children }: { children: string }) {
  return children ? <h3 className="mb-3 text-base font-semibold">{children}</h3> : null;
}

export function ViewBlock({ view }: { view: View }) {
  switch (view.type) {
    case "markdown": return <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900"><Markdown text={view.text} /></div>;
    case "callout": return <Callout view={view} />;
    case "table": return <TableView view={view} />;
    case "scorecard": return <ScorecardView view={view} />;
    case "results": return <ResultsView view={view} />;
    case "bars": return <BarsView view={view} />;
    case "answer": return (
      <div className="h-full rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-2 text-sm font-semibold text-zinc-500">{view.title}</div>
        <Markdown text={view.text} />
        <div className="mt-3 flex flex-wrap gap-2">{Object.entries(view.meta).map(([k, v]) => <span key={k} className="rounded-md bg-zinc-100 px-2 py-0.5 font-mono text-xs dark:bg-zinc-800">{k}: {String(v)}</span>)}</div>
      </div>
    );
    case "images": return (
      <div><Title>{view.title}</Title>
        <div className="flex flex-wrap gap-4">{view.items.map((im) => (
          <figure key={im.caption} className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
            <img src={im.src} alt={im.caption} className="max-h-80 w-auto" />
            <figcaption className="border-t border-zinc-100 px-3 py-1.5 font-mono text-xs text-zinc-500 dark:border-zinc-800">{im.caption}</figcaption>
          </figure>))}</div>
      </div>
    );
    case "code": return (
      <div><Title>{view.title}</Title>
        <pre className="max-h-[28rem] overflow-auto rounded-xl bg-zinc-900 p-4 font-mono text-[13px] leading-relaxed text-zinc-100 dark:bg-black">{view.text}</pre>
      </div>
    );
    case "example": return <ExampleView view={view} step={0} />;
    case "columns": return (
      <div><Title>{view.title}</Title>
        <div className={`grid gap-4 ${view.children.length >= 3 ? "lg:grid-cols-3" : "lg:grid-cols-2"}`}>{view.children.map((c, i) => <ViewBlock key={i} view={c} />)}</div>
      </div>
    );
  }
}

const TONES = {
  info: "border-sky-300 bg-sky-50 dark:border-sky-500/40 dark:bg-sky-500/10",
  success: "border-emerald-300 bg-emerald-50 dark:border-emerald-500/40 dark:bg-emerald-500/10",
  warning: "border-amber-300 bg-amber-50 dark:border-amber-500/40 dark:bg-amber-500/10",
  decision: "border-violet-300 bg-violet-50 dark:border-violet-500/40 dark:bg-violet-500/10",
};
const ICONS = { info: "ℹ️", success: "✅", warning: "⚠️", decision: "🧭" };

function Callout({ view }: { view: Extract<View, { type: "callout" }> }) {
  return (
    <div className={`rounded-2xl border-l-4 p-5 ${TONES[view.tone]}`}>
      {view.title && <div className="mb-1 font-semibold">{ICONS[view.tone]} {view.title}</div>}
      <Markdown text={view.text} />
    </div>
  );
}

function TableView({ view }: { view: Extract<View, { type: "table" }> }) {
  return (
    <div><Title>{view.title}</Title>
      {view.rows.length === 0 ? <div className="text-sm text-zinc-500">(none)</div> : (
        <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500 dark:bg-zinc-950"><tr>{view.columns.map((c) => <th key={c} className="px-4 py-2 font-medium">{c}</th>)}</tr></thead>
            <tbody>{view.rows.map((row, i) => <tr key={i} className="border-t border-zinc-100 dark:border-zinc-800">{row.map((cell, j) => <td key={j} className="px-4 py-2 align-top">{String(cell)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ScorecardView({ view }: { view: Extract<View, { type: "scorecard" }> }) {
  const best = (m: MetricSpec) => {
    if (m.better === "none") return null;
    const values = view.rows.map((r) => r[m.key]).filter((v): v is number => typeof v === "number");
    if (values.length < 2 || values.every((v) => v === values[0])) return null; // a tie everywhere isn't a winner
    return m.better === "higher" ? Math.max(...values) : Math.min(...values);
  };
  return (
    <div><Title>{view.title}</Title>
      <div className="overflow-x-auto rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-zinc-200 text-xs text-zinc-500 dark:border-zinc-800">
            <tr><th className="px-4 py-3 font-medium">Variant</th>{view.metrics.map((m) => (
              <th key={m.key} className="px-4 py-3 text-right font-medium" title={m.help}>
                {m.label}{m.better !== "none" && <span className="ml-1 text-zinc-400">{m.better === "higher" ? "↑" : "↓"}</span>}
              </th>))}</tr>
          </thead>
          <tbody>{view.rows.map((row, i) => (
            <tr key={i} className="border-t border-zinc-100 dark:border-zinc-800">
              <td className="whitespace-nowrap px-4 py-3 font-medium">{String(row[view.label_key])}</td>
              {view.metrics.map((m) => {
                const isBest = best(m) !== null && row[m.key] === best(m);
                return <td key={m.key} className={`whitespace-nowrap px-4 py-3 text-right font-mono tabular-nums ${isBest ? "font-semibold text-emerald-700 dark:text-emerald-400" : ""}`}>
                  {formatMetric(row[m.key], m.format)}{isBest && " ★"}</td>;
              })}
            </tr>))}</tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-zinc-500">★ best in column · ↑ higher is better · ↓ lower is better · hover a header for its definition</p>
    </div>
  );
}

function ResultsView({ view }: { view: Extract<View, { type: "results" }> }) {
  const max = Math.max(...view.hits.map((h) => Math.abs(h.score)), 1e-9);
  return (
    <div><Title>{view.title}</Title>
      {view.hits.length === 0 && <div className="rounded-xl border border-dashed border-zinc-300 p-4 text-sm text-zinc-500 dark:border-zinc-700">No matches.</div>}
      <ol className="space-y-2">{view.hits.map((h, i) => (
        <li key={i} className={`rounded-xl border p-3 ${h.relevant === true ? "border-emerald-300 bg-emerald-50/60 dark:border-emerald-500/40 dark:bg-emerald-500/10" : "border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900"}`}>
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="font-semibold">#{i + 1} <span className="font-mono text-xs font-normal text-zinc-500">{h.source}</span></span>
            <span className="flex items-center gap-2">
              {h.relevant === true && <span className="rounded bg-emerald-600 px-1.5 text-xs font-semibold text-white">✓ answers it</span>}
              {h.relevant === false && <span className="rounded bg-zinc-200 px-1.5 text-xs text-zinc-600 dark:bg-zinc-700 dark:text-zinc-300">✗ not it</span>}
              <span className="font-mono text-xs">{view.score_label} {h.score}</span>
            </span>
          </div>
          <div className="mt-2 h-1 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800"><div className="h-full rounded-full bg-sky-500" style={{ width: `${(Math.abs(h.score) / max) * 100}%` }} /></div>
          <p className="mt-2 line-clamp-3 whitespace-pre-wrap text-sm text-zinc-600 dark:text-zinc-400">{h.text}</p>
          {h.matched && h.matched.length > 0 && <div className="mt-1 flex flex-wrap gap-1">{h.matched.map((w) => <span key={w} className="rounded bg-amber-100 px-1.5 font-mono text-xs text-amber-900 dark:bg-amber-400/20 dark:text-amber-200">{w}</span>)}</div>}
        </li>))}</ol>
    </div>
  );
}

function BarsView({ view }: { view: Extract<View, { type: "bars" }> }) {
  const series = [...new Set(view.items.flatMap((i) => Object.keys(i.values)))];
  const max = view.max ?? Math.max(...view.items.flatMap((i) => Object.values(i.values)), 1e-9);
  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900"><Title>{view.title}</Title>
      <div className="mb-3 flex flex-wrap gap-4 text-xs">{series.map((s, i) => <span key={s} className="flex items-center gap-1.5"><span className={`size-2.5 rounded-sm ${SERIES[i % 4]}`} />{s}</span>)}</div>
      <div className="space-y-3">{view.items.map((item) => (
        <div key={item.label} className="grid grid-cols-[7rem_1fr] items-center gap-3">
          <div className="text-sm font-medium">{item.label}</div>
          <div className="space-y-1">{series.map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div className="h-3 flex-1 overflow-hidden rounded bg-zinc-100 dark:bg-zinc-800"><div className={`h-full rounded ${SERIES[i % 4]}`} style={{ width: `${((item.values[s] ?? 0) / max) * 100}%` }} /></div>
              <span className={`w-12 text-right font-mono text-xs ${SERIES_TEXT[i % 4]}`}>{(item.values[s] ?? 0).toFixed(2)}</span>
            </div>))}</div>
        </div>))}</div>
    </div>
  );
}
