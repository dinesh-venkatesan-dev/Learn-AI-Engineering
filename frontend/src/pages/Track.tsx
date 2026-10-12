// A track: its modules, and the syllabus: every learning point, its home module, and whether it's covered yet.
import { useEffect, useState } from "react";
import { getSyllabus } from "../api";
import { Markdown } from "../components/Markdown";
import { Coverage, Empty, Eyebrow, Pills, Stars } from "../components/ui";
import { href } from "../router";
import type { Catalog, SyllabusItem } from "../types";

const COURSES: Record<string, string> = {
  lc: "LangChain", rag: "RAG", lg: "LangGraph", mcp: "MCP", smith: "LangSmith", deep: "Deep Agents",
  api: "FastAPI", prod: "Production", data: "Data", deploy: "Deployment", new: "✨ New in this course",
};

export function TrackPage({ catalog, track }: { catalog: Catalog; track: string }) {
  const t = catalog.tracks.find((x) => x.id === track);
  const [loaded, setLoaded] = useState<{ track: string; items: SyllabusItem[] } | null>(null);
  const items = loaded?.track === track ? loaded.items : null;
  const [filter, setFilter] = useState<"all" | "covered" | "open">("all");
  const [minStars, setMinStars] = useState(1);
  useEffect(() => {
    getSyllabus(track).then((list) => setLoaded({ track, items: list }), () => setLoaded({ track, items: [] }));
  }, [track]);
  if (!t) return <Empty>No track “{track}”.</Empty>;

  return (
    <div className="space-y-8">
      <header>
        <Eyebrow>Track</Eyebrow>
        <h1 className="mt-1 text-4xl font-bold tracking-tight">{t.icon} {t.title}</h1>
        <p className="mt-2 text-lg text-zinc-600 dark:text-zinc-400">{t.summary}</p>
        <Coverage className="mt-4" covered={t.covered} items={t.items} />
      </header>

      <section className="space-y-3">
        {t.modules.map((m) => {
          const body = (
            <>
              <div className="flex items-center gap-3">
                <span className="font-mono text-sm text-zinc-400">{m.id}</span>
                <span className="flex-1 font-semibold">{m.title}</span>
                <Stars n={m.stars} />
                <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${m.status === "built"
                  ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-200" : m.status === "preview"
                  ? "bg-amber-100 text-amber-800 dark:bg-amber-400/15 dark:text-amber-200" : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800"}`}>
                  {m.status === "built" ? "✓ built" : m.status === "preview" ? "preview: first labs" : "planned"}</span>
              </div>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">{m.summary}</p>
              <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">Why {"★".repeat(m.stars)}: {m.why}</p>
              <Coverage className="mt-3" covered={m.covered} items={m.items} />
              <div className="mt-2 text-xs text-zinc-400">From the old courses: {m.from.join(", ")}{m.except?.length ? ` (except ${m.except.join(", ")})` : ""}</div>
              {m.notes && <div className="mt-1 text-xs text-sky-700 dark:text-sky-400">↪ {m.notes}</div>}
              {m.capstones?.length ? <div className="mt-1 text-xs text-violet-700 dark:text-violet-400">🏁 Capstone {m.capstones.join(", ")} is built by this module's production path</div> : null}
            </>
          );
          const box = "block rounded-2xl border p-4";
          return m.status !== "planned"
            ? <a key={m.id} href={href.module(t.id, m.id)} className={`${box} border-zinc-200 bg-white shadow-sm transition hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900`}>{body}</a>
            : <div key={m.id} className={`${box} border-dashed border-zinc-200 dark:border-zinc-800`}>{body}</div>;
        })}
      </section>

      <section>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-xl font-semibold">Syllabus: every learning point in this track</h2>
          <div className="flex flex-wrap gap-3">
            <Pills label="Importance" value={minStars} onChange={setMinStars}
                   items={[{ id: 1, label: "All" }, { id: 2, label: "★★ and up" }, { id: 3, label: "★★★ essential" }]} />
            <Pills label="Status" value={filter} onChange={setFilter}
                   items={[{ id: "all", label: "All" }, { id: "covered", label: "✓ Covered" }, { id: "open", label: "Not yet" }]} />
          </div>
        </div>
        {!items ? <div className="text-zinc-500">Loading…</div> : t.modules.map((m) => {
          const mine = items.filter((s) => s.module === m.id && (s.stars ?? 2) >= minStars
            && (filter === "all" || (filter === "covered") === !!s.covered));
          if (!mine.length) return null;
          const lessons = [...new Set(mine.map((s) => s.lesson))];
          return (
            <details key={m.id} className="mb-3 rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900" open={m.status !== "planned"}>
              <summary className="cursor-pointer px-4 py-3 font-medium">{m.id} · {m.title} <span className="text-sm text-zinc-500">({mine.length})</span></summary>
              <div className="border-t border-zinc-100 dark:border-zinc-800">
                {lessons.map((lesson) => {
                  const points = mine.filter((s) => s.lesson === lesson);
                  return (
                    <div key={lesson} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
                      <div className="flex items-center gap-3 bg-zinc-50 px-4 py-2 text-sm dark:bg-zinc-950/50">
                        <Stars n={points[0].stars ?? 2} />
                        <span className="flex-1 font-medium">{points[0].lesson_title}</span>
                        <span className="text-xs text-zinc-400">{COURSES[points[0].course] ?? points[0].course} · <span className="font-mono">{lesson}</span></span>
                      </div>
                      <ul>{points.map((s) => (
                        <li key={s.id} className="flex gap-3 px-4 py-1.5 pl-6 text-sm">
                          <span className={s.covered ? "text-emerald-600" : "text-zinc-300 dark:text-zinc-600"}>{s.covered ? "✓" : "○"}</span>
                          <div className="min-w-0 flex-1"><Markdown text={s.text} /></div>
                        </li>))}</ul>
                    </div>
                  );
                })}
              </div>
            </details>
          );
        })}
      </section>
    </div>
  );
}
