// A lab: its inputs, a Run button, and whatever views it returns; plus the code behind it and what it covers.
import { useEffect, useState } from "react";
import { getLastLab, runLab } from "../api";
import { CodeBlock } from "../components/CodeBlock";
import { Markdown } from "../components/Markdown";
import { Card, Empty, Eyebrow, Tabs } from "../components/ui";
import { ExampleView, Views } from "../components/Views";
import { href } from "../router";
import type { Example, LabEntry, LabInput, LabResult, ModulePage, SyllabusItem } from "../types";

type Tab = "run" | "code" | "covers";

export function LabPage({ page, labId }: { page: ModulePage; labId: string }) {
  const index = page.labs.findIndex((l) => l.id === labId);
  const lab = page.labs[index];
  if (!lab) return <Empty>No lab “{labId}” in this module.</Empty>;
  const prev = page.labs[index - 1], next = page.labs[index + 1];
  return (
    <article className="space-y-6">
      <header>
        <Eyebrow><a href={href.module(page.track, page.id)} className="hover:underline">Module {page.id}</a> · Lab {index + 1} of {page.labs.length}</Eyebrow>
        <h1 className="mt-1 text-4xl font-bold tracking-tight">{lab.icon} {lab.title}</h1>
      </header>
      <Card><Markdown text={lab.explain} /></Card>
      <LabBody key={lab.id} lab={lab} />
      <nav className="flex justify-between border-t border-zinc-200 pt-6 dark:border-zinc-800">
        {prev ? <a href={href.lab(page.track, page.id, prev.id)}><div className="text-xs text-zinc-500">← Previous lab</div><div className="font-semibold">{prev.icon} {prev.title}</div></a>
              : <a href={href.module(page.track, page.id)}><div className="text-xs text-zinc-500">← Back to</div><div className="font-semibold">Module overview</div></a>}
        {next ? <a href={href.lab(page.track, page.id, next.id)} className="text-right"><div className="text-xs text-zinc-500">Next lab →</div><div className="font-semibold">{next.icon} {next.title}</div></a>
              : page.path.steps.length > 0 && <a href={href.path(page.track, page.id)} className="text-right"><div className="text-xs text-zinc-500">Next →</div><div className="font-semibold">🏭 {page.path.title}</div></a>}
      </nav>
    </article>
  );
}

export function LabBody({ lab, showTabs = true }: { lab: LabEntry; showTabs?: boolean }) {
  const [tab, setTab] = useState<Tab>("run");
  const [values, setValues] = useState<Record<string, unknown>>(() => Object.fromEntries(lab.inputs.map((i) => [i.name, i.default])));
  const [result, setResult] = useState<LabResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getLastLab(lab.id).then((last) => { if (last) { setResult(last); setValues(last.inputs); } }, () => { /* never run */ });
  }, [lab.id]);

  async function run() {
    setBusy(true); setError(null);
    try { setResult(await runLab(lab.id, values)); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  }

  return (
    <div className="space-y-6">
      {showTabs && <Tabs value={tab} onChange={setTab} tabs={[
        { id: "run", label: "▶ Run & learn" }, { id: "code", label: `The code (${lab.code.length})` },
        { id: "covers", label: `What it covers (${lab.covered_items.length})` }]} />}

      {tab === "run" && (
        <div className="space-y-6">
          <form onSubmit={(e) => { e.preventDefault(); run(); }}
                className="flex flex-wrap items-end gap-4 rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
            {lab.inputs.map((input) => <Field key={input.name} input={input} value={values[input.name]} onChange={(v) => setValues({ ...values, [input.name]: v })} />)}
            <div className="ml-auto flex items-center gap-3">
              {lab.uses_model && <span className="max-w-72 text-xs text-amber-700 dark:text-amber-300">Calls a model: {lab.calls || "a few calls"} per run (free plans allow about 30 a minute)</span>}
              <button disabled={busy} className="rounded-xl bg-sky-600 px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-sky-700 disabled:opacity-50">
                {busy ? "Running…" : "▶ Run lab"}</button>
            </div>
          </form>
          {error && <div className="rounded-lg bg-rose-50 px-4 py-2 text-sm text-rose-800 dark:bg-rose-500/10 dark:text-rose-200">{error}</div>}
          {result ? (
            <div className={busy ? "opacity-50" : ""}>
              <div className="mb-3 text-xs text-zinc-500">Ran in {result.ms.toLocaleString()} ms with {Object.entries(result.inputs).map(([k, v]) => `${k}=${String(v)}`).join(", ") || "no inputs"}</div>
              <Views views={result.views} />
            </div>
          ) : <Empty>{busy ? "Running… results appear here when the lab finishes." : <>Press <b>▶ Run lab</b> to run it.</>}</Empty>}
        </div>
      )}
      {tab === "code" && (
        <div className="space-y-6">
          <p className="text-zinc-600 dark:text-zinc-400">The real code, read from the package when this page loads. <span className="rounded bg-amber-100 px-1.5 dark:bg-amber-400/15">Highlighted lines</span> have a note underneath.</p>
          {lab.code.map((c) => <CodeBlock key={`${c.module}.${c.symbol}`} code={c} />)}
        </div>
      )}
      {tab === "covers" && <CoveredItems items={lab.covered_items} examples={result?.examples} notRunYet={!result} />}
    </div>
  );
}

function Field({ input, value, onChange }: { input: LabInput; value: unknown; onChange: (v: unknown) => void }) {
  const box = "rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-800";
  return (
    <label className={`flex flex-col gap-1 text-sm ${input.kind === "textarea" ? "w-full" : input.kind === "text" || input.kind === "select" ? "min-w-72 flex-1" : ""}`} title={input.help}>
      <span className="text-xs font-medium text-zinc-500">{input.label}</span>
      {input.kind === "select" && input.options ? (
        <select value={String(value)} onChange={(e) => onChange(e.target.value)} className={box}>
          {input.options.map((o) => <option key={o} value={o}>{o}</option>)}</select>
      ) : input.kind === "bool" ? (
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} className="size-5" />
      ) : input.kind === "textarea" ? (
        <textarea value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} rows={3} className={`${box} w-full`} />
      ) : input.kind === "number" ? (
        <input type="number" value={Number(value)} min={input.min ?? undefined} max={input.max ?? undefined}
               onChange={(e) => onChange(Number(e.target.value))} className={`${box} w-28 font-mono`} />
      ) : (
        <input value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} className={box} />
      )}
    </label>
  );
}

/** The learning points this covers; when a lab shows code per point, each point gets its code and output. */
export function CoveredItems({ items, examples, notRunYet = false }: { items: SyllabusItem[]; examples?: Example[]; notRunYet?: boolean }) {
  if (!examples?.length) {
    return (
      <Card>
        <p className="mb-3 text-sm text-zinc-500">The learning points from the old courses that this covers:</p>
        <ul className="space-y-2">{items.map((s) => (
          <li key={s.id} className="flex gap-3 text-sm"><span className="text-emerald-600">✓</span>
            <div><Markdown text={s.text} /><div className="text-xs text-zinc-400">{s.lesson_title} · <span className="font-mono">{s.id}</span></div></div></li>))}</ul>
        {notRunYet && <p className="mt-4 text-sm text-zinc-500">Press <b>▶ Run lab</b> to see the code for each point and what it produced.</p>}
      </Card>
    );
  }
  let step = 0;
  return (
    <div className="space-y-8">
      <p className="text-zinc-600 dark:text-zinc-400">Each learning point, with the few lines of code that do it and what those lines produced in your last run.</p>
      {items.map((s) => {
        const mine = examples.filter((e) => e.point === s.id);
        return (
          <section key={s.id} className="space-y-3">
            <div className="flex gap-3"><span className="mt-0.5 text-emerald-600">✓</span>
              <div><div className="font-semibold"><Markdown text={s.text} /></div><div className="font-mono text-xs text-zinc-400">{s.id}</div></div></div>
            {mine.map((e) => <ExampleView key={e.title} view={e} step={++step} />)}
          </section>
        );
      })}
    </div>
  );
}
