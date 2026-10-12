// The production path: the run bar, an overview of the steps, and one page per step (output · code · production).
import { useState } from "react";
import { CodeBlock } from "../components/CodeBlock";
import { Markdown } from "../components/Markdown";
import { Card, Empty, Eyebrow, Tabs } from "../components/ui";
import type { Ingestion, Run } from "../ingestion";
import { href } from "../router";
import { EvaluateView } from "../steps/Evaluate";
import { ChunkView, CleanView, EmbedView, LoadView, StoreView } from "../steps/views";
import type { ChunkData, CleanData, EmbedData, LoadData, ModulePage, Step, StepId, StepResult, StoreData } from "../types";
import { CoveredItems, LabBody } from "./Lab";

export function PathPage({ page, step, ingestion }: { page: ModulePage; step: string | null; ingestion: Ingestion }) {
  const current = page.path.steps.find((s) => s.id === step);
  const ingests = page.path.runner === "rag-ingestion";
  return (
    <div className="space-y-6">
      {ingests && <RunBar ingestion={ingestion} />}
      {current ? <StepPage key={current.id} page={page} step={current} run={ingestion.run} ingests={ingests} />
               : step ? <Empty>No step “{step}”.</Empty> : <PathOverview page={page} ingestion={ingestion} />}
    </div>
  );
}

function RunBar({ ingestion }: { ingestion: Ingestion }) {
  const { run, options, setOptions, start } = ingestion;
  const r = run.report;
  const box = "rounded-lg border border-zinc-300 bg-white px-2 py-1 font-mono dark:border-zinc-700 dark:bg-zinc-800";
  return (
    <div className="sticky top-0 z-10 -mx-6 flex flex-wrap items-center gap-x-5 gap-y-3 border-b border-zinc-200 bg-white/85 px-6 py-3 backdrop-blur lg:-mx-10 lg:px-10 dark:border-zinc-800 dark:bg-zinc-950/85">
      <label className="flex items-center gap-2 text-sm"><span className="text-zinc-500">Chunk size</span>
        <input type="number" min={100} max={4000} step={50} value={options.chunkSize}
               onChange={(e) => setOptions({ ...options, chunkSize: Number(e.target.value) })} className={`${box} w-20`} /></label>
      <label className="flex items-center gap-2 text-sm"><span className="text-zinc-500">Overlap</span>
        <input type="number" min={0} max={options.chunkSize - 1} step={10} value={options.chunkOverlap}
               onChange={(e) => setOptions({ ...options, chunkOverlap: Number(e.target.value) })} className={`${box} w-16`} /></label>
      <label className="flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400">
        <input type="checkbox" checked={options.force} onChange={(e) => setOptions({ ...options, force: e.target.checked })} />Re-embed everything</label>
      <div className="ml-auto flex items-center gap-4">
        <span className="text-sm text-zinc-500">
          {run.status === "running" && "Running…"}
          {run.status === "error" && <span className="text-rose-600">{run.error}</span>}
          {r && run.status === "done" && <><b className="text-emerald-600">{r.added} added</b> · {r.skipped} skipped · {r.deleted} deleted · {r.seconds}s</>}
        </span>
        <button onClick={start} disabled={run.status === "running"}
                className="rounded-xl bg-sky-600 px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-sky-700 disabled:opacity-50">
          {run.status === "running" ? "Running…" : "▶ Run ingestion"}</button>
      </div>
    </div>
  );
}

function PathOverview({ page, ingestion }: { page: ModulePage; ingestion: Ingestion }) {
  const { start } = ingestion;
  const ingests = page.path.runner === "rag-ingestion";
  const run: Run = ingests ? ingestion.run : { status: "idle", results: {} };
  return (
    <div className="space-y-8">
      <header>
        <Eyebrow><a href={href.module(page.track, page.id)} className="hover:underline">Module {page.id}</a> · 🏭 Production path</Eyebrow>
        <h1 className="mt-1 text-4xl font-bold tracking-tight">{page.path.title}</h1>
      </header>
      <Card><Markdown text={page.path.summary} /></Card>
      <section>
        <h2 className="mb-3 text-xl font-semibold">The steps</h2>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          {page.path.steps.map((s, i) => {
            const result = run.results[s.id];
            return (
              <a key={s.id} href={href.path(page.track, page.id, s.id)}
                 className={`rounded-2xl border p-4 transition hover:-translate-y-0.5 hover:shadow-md ${result
                   ? "border-emerald-400/60 bg-emerald-50 dark:bg-emerald-500/10" : run.status === "running" && s.id !== "evaluate"
                   ? "animate-pulse border-sky-300 bg-white dark:border-sky-700 dark:bg-zinc-900" : "border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900"}`}>
                <div className="flex items-center justify-between text-xs text-zinc-500"><span>Step {i + 1}</span>{result && <span className="font-mono">{result.ms} ms</span>}</div>
                <div className="mt-1 text-2xl">{s.icon}</div>
                <div className="font-semibold">{s.title}</div>
                <div className="mt-1 text-xs leading-snug text-zinc-500">{result ? headline(s.id, result) : s.tagline}</div>
                {s.lab && <div className="mt-2 text-[11px] font-medium text-sky-600">▶ runnable</div>}
              </a>
            );
          })}
        </div>
        {ingests && run.status === "idle" && <p className="mt-3 text-sm text-zinc-500">Press <button onClick={start} className="font-semibold text-sky-600 underline">Run ingestion</button>, then click any step to see what it produced and the code that did it. Finish with <b>Evaluate</b> to score the index.</p>}
      </section>
      <section>
        <h2 className="mb-3 text-xl font-semibold">What this path covers</h2>
        <CoveredItems items={page.path.covered_items} />
      </section>
    </div>
  );
}

function headline(id: StepId, r: StepResult): string {
  const s = r.stats;
  switch (id) {
    case "load": return `${s.files} files → ${s.documents} documents`;
    case "clean": return `${(Number(s.chars_before) - Number(s.chars_after)).toLocaleString()} chars of noise removed`;
    case "chunk": return `${s.chunks} chunks`;
    case "embed": return `${s.embedded} texts embedded`;
    case "store": return `${s.added} added · ${s.skipped} skipped`;
    default: return "";
  }
}

type Tab = "output" | "code" | "production";

function StepPage({ page, step, run, ingests }: { page: ModulePage; step: Step; run: Run; ingests: boolean }) {
  const hasOutput = ingests || !!step.lab_entry;
  const [tab, setTab] = useState<Tab>(hasOutput ? "output" : "code");
  const steps = page.path.steps;
  const index = steps.indexOf(step);
  const prev = steps[index - 1], next = steps[index + 1];
  const result = run.results[step.id];
  return (
    <article className="space-y-6">
      <header>
        <Eyebrow><a href={href.path(page.track, page.id)} className="hover:underline">🏭 Production path</a> · Step {index + 1} of {steps.length}</Eyebrow>
        <h1 className="mt-1 flex items-center gap-3 text-4xl font-bold tracking-tight"><span>{step.icon}</span>{step.title}</h1>
        <p className="mt-2 text-lg text-zinc-600 dark:text-zinc-400">{step.tagline}</p>
      </header>
      <Card><Markdown text={step.explain} /></Card>
      <Tabs value={tab} onChange={setTab} tabs={[
        ...(hasOutput ? [{ id: "output" as Tab, label: step.id === "evaluate" ? "The scorecard" : step.lab_entry ? "▶ Run it" : "What it produced" }] : []),
        { id: "code", label: `The code (${step.code.length})` }, { id: "production", label: "In production" }]} />

      {tab === "output" && step.lab_entry && <LabBody lab={step.lab_entry} showTabs={false} />}
      {tab === "output" && !step.lab_entry && ingests && (step.id === "evaluate" ? <EvaluateView />
        : result ? <StepOutput result={result} />
        : <Empty>{run.status === "running" ? "Running… this step's output appears here when it finishes." : <>Press <b>▶ Run ingestion</b> (top right) to see what this step produces.</>}</Empty>)}
      {tab === "code" && (
        <div className="space-y-6">
          <p className="text-zinc-600 dark:text-zinc-400">The actual code from the package, read from the file each time this page loads. <span className="rounded bg-amber-100 px-1.5 dark:bg-amber-400/15">Highlighted lines</span> have a note underneath.</p>
          {step.code.map((c) => <CodeBlock key={`${c.module}.${c.symbol}`} code={c} />)}
        </div>
      )}
      {tab === "production" && (
        <Card><ul className="space-y-4">{step.production.map((p, i) => (
          <li key={i} className="flex gap-3"><span className="mt-1 text-sky-500">▸</span><Markdown text={p} /></li>))}</ul></Card>
      )}

      <nav className="flex justify-between border-t border-zinc-200 pt-6 dark:border-zinc-800">
        {prev ? <a href={href.path(page.track, page.id, prev.id)}><div className="text-xs text-zinc-500">← Previous</div><div className="font-semibold">{prev.icon} {prev.title}</div></a>
              : <a href={href.path(page.track, page.id)}><div className="text-xs text-zinc-500">← Back to</div><div className="font-semibold">Path overview</div></a>}
        {next ? <a href={href.path(page.track, page.id, next.id)} className="text-right"><div className="text-xs text-zinc-500">Next →</div><div className="font-semibold">{next.icon} {next.title}</div></a>
              : <a href={href.module(page.track, page.id)} className="text-right"><div className="text-xs text-zinc-500">Done →</div><div className="font-semibold">Module overview</div></a>}
      </nav>
    </article>
  );
}

function StepOutput({ result }: { result: StepResult }) {
  const stats = result.stats;
  switch (result.step) {
    case "load": return <LoadView data={result.data as LoadData} stats={stats} />;
    case "clean": return <CleanView data={result.data as CleanData} stats={stats} />;
    case "chunk": return <ChunkView data={result.data as ChunkData} stats={stats} />;
    case "embed": return <EmbedView data={result.data as EmbedData} stats={stats} />;
    case "store": return <StoreView data={result.data as StoreData} stats={stats} />;
    default: return null;
  }
}
