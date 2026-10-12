// A module's overview: what you'll learn, its labs, and its production path.
import { Markdown } from "../components/Markdown";
import { Card, Command, Coverage, Eyebrow, Stars } from "../components/ui";
import type { Run } from "../ingestion";
import { href } from "../router";
import type { Catalog, ModulePage } from "../types";

export function ModuleOverview({ catalog, page, run }: { catalog: Catalog; page: ModulePage; run: Run }) {
  const track = catalog.tracks.find((t) => t.id === page.track);
  const meta = track?.modules.find((m) => m.id === page.id);
  return (
    <div className="space-y-10">
      <header>
        <Eyebrow><a href={href.track(page.track)} className="hover:underline">{track?.title}</a> · Module {page.id}</Eyebrow>
        <h1 className="mt-1 text-4xl font-bold tracking-tight">{page.title}</h1>
        <div className="mt-2 flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400"><Stars n={page.stars} size="lg" /><span>{page.why}</span></div>
        {meta && <Coverage className="mt-4" covered={meta.covered} items={meta.items} />}
      </header>
      <Card><Markdown text={page.summary} /></Card>

      <section>
        <h2 className="mb-3 text-xl font-semibold">What you'll learn</h2>
        <Card><ol className="space-y-3">{page.learn.map((item, i) => (
          <li key={i} className="flex gap-3"><span className="grid size-6 shrink-0 place-items-center rounded-full bg-sky-100 text-xs font-bold text-sky-700 dark:bg-sky-500/20 dark:text-sky-300">{i + 1}</span><Markdown text={item} /></li>))}</ol></Card>
      </section>

      <section>
        <h2 className="mb-3 text-xl font-semibold">The big picture</h2>
        <pre className="overflow-x-auto rounded-2xl bg-zinc-900 p-5 font-mono text-[13px] leading-relaxed text-zinc-100 dark:bg-black">{page.flow}</pre>
      </section>

      <section>
        <h2 className="mb-1 text-xl font-semibold">🧪 Labs</h2>
        <p className="mb-3 text-zinc-600 dark:text-zinc-400">Run each alternative yourself and compare them on real numbers.</p>
        <div className="grid gap-4 md:grid-cols-2">
          {page.labs.map((l, i) => (
            <a key={l.id} href={href.lab(page.track, page.id, l.id)}
               className="flex flex-col rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center justify-between text-xs text-zinc-500"><span>Lab {i + 1}</span>
                {l.uses_model && <span title={l.calls} className="rounded bg-amber-100 px-1.5 text-amber-800 dark:bg-amber-400/15 dark:text-amber-200">calls a model</span>}</div>
              <div className="mt-1 text-lg font-semibold">{l.icon} {l.title}</div>
              <div className="mt-1 line-clamp-3 text-sm text-zinc-600 dark:text-zinc-400">{firstParagraph(l.explain)}</div>
              <div className="mt-auto pt-3 text-xs text-zinc-400">covers {l.covered_items.length} learning points</div>
            </a>
          ))}
        </div>
      </section>

      {meta?.status === "preview" && <Card><b>Preview.</b> This module's first lab is ready for review; the rest of its labs and its production path are being built.</Card>}

      {page.path.steps.length > 0 && <section>
        <h2 className="mb-1 text-xl font-semibold">🏭 {page.path.title}</h2>
        <p className="mb-3 text-zinc-600 dark:text-zinc-400">{firstParagraph(page.path.summary)}</p>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          {page.path.steps.map((s, i) => (
            <a key={s.id} href={href.path(page.track, page.id, s.id)}
               className={`rounded-2xl border p-4 transition hover:-translate-y-0.5 hover:shadow-md ${run.results[s.id]
                 ? "border-emerald-400/60 bg-emerald-50 dark:bg-emerald-500/10" : "border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900"}`}>
              <div className="text-xs text-zinc-500">Step {i + 1}</div>
              <div className="mt-1 text-2xl">{s.icon}</div>
              <div className="font-semibold">{s.title}</div>
              <div className="mt-1 text-xs leading-snug text-zinc-500">{s.tagline}</div>
            </a>
          ))}
        </div>
      </section>}

      <section>
        <h2 className="mb-3 text-xl font-semibold">Where the code lives</h2>
        <div className="overflow-hidden rounded-2xl border border-zinc-200 dark:border-zinc-800">
          {page.layout.map((f) => (
            <div key={f.path} className="flex flex-wrap gap-x-6 gap-y-1 border-b border-zinc-100 bg-white px-5 py-3 last:border-0 dark:border-zinc-800 dark:bg-zinc-900">
              <code className="w-80 shrink-0 font-mono text-sm text-sky-700 dark:text-sky-400">{f.path}</code><span className="text-sm text-zinc-600 dark:text-zinc-400">{f.role}</span>
            </div>))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-xl font-semibold">Run it yourself</h2>
        <div className="grid gap-3">{page.commands.map((c) => <div key={c.cmd}><div className="mb-1 text-sm text-zinc-500">{c.label} <span className="text-zinc-400">(in ai-course/)</span></div><Command cmd={c.cmd} /></div>)}</div>
      </section>
    </div>
  );
}

const firstParagraph = (text: string) => text.split("\n\n")[0].replace(/[*`]/g, "");
