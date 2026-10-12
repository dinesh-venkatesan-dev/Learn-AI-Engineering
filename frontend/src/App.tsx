// The academy shell: sidebar (tracks → modules → module contents) and the page for the current route.
import { useEffect, useState } from "react";
import { getCatalog, getModule } from "./api";
import { Empty, Stars } from "./components/ui";
import { useIngestion } from "./ingestion";
import { HomePage } from "./pages/Home";
import { LabPage } from "./pages/Lab";
import { ModuleOverview } from "./pages/Module";
import { PathPage } from "./pages/Path";
import { TrackPage } from "./pages/Track";
import { href, useRoute, type Route } from "./router";
import { useTheme } from "./theme";
import type { Catalog, ModulePage } from "./types";

export function App() {
  const route = useRoute();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ingestion = useIngestion();
  const page = useModule(route);

  useEffect(() => { getCatalog().then(setCatalog, (e) => setError(String(e.message ?? e))); }, []);

  if (error) return <div className="p-10"><Empty>Couldn't load the course: {error}. Is the server running (<code>uv run academy</code>)?</Empty></div>;
  if (!catalog) return <div className="p-10 text-zinc-500">Loading…</div>;

  return (
    <div className="flex min-h-dvh">
      <Sidebar catalog={catalog} route={route} page={page.data} />
      <div className="min-w-0 flex-1">
        <main className="mx-auto max-w-5xl px-6 py-8 lg:px-10">
          {route.page === "home" && <HomePage catalog={catalog} />}
          {route.page === "track" && <TrackPage catalog={catalog} track={route.track} />}
          {"module" in route && page.error && <Empty>{page.error}</Empty>}
          {"module" in route && !page.error && !page.data && <div className="text-zinc-500">Loading…</div>}
          {page.data && route.page === "module" && <ModuleOverview catalog={catalog} page={page.data} run={ingestion.run} />}
          {page.data && route.page === "lab" && <LabPage page={page.data} labId={route.lab} />}
          {page.data && route.page === "path" && <PathPage page={page.data} step={route.step} ingestion={ingestion} />}
        </main>
      </div>
    </div>
  );
}

/** The module page for the current route, fetched once per module. */
function useModule(route: Route) {
  const key = "module" in route ? `${route.track}/${route.module}` : null;
  const [cache, setCache] = useState<Record<string, ModulePage | string>>({});
  useEffect(() => {
    if (!key || cache[key]) return;
    const [track, id] = key.split("/");
    getModule(track, id).then((p) => setCache((c) => ({ ...c, [key]: p })), (e) => setCache((c) => ({ ...c, [key]: String(e.message ?? e) })));
  }, [key, cache]);
  const value = key ? cache[key] : undefined;
  return { data: typeof value === "object" ? value : null, error: typeof value === "string" ? value : null };
}

function Sidebar({ catalog, route, page }: { catalog: Catalog; route: Route; page: ModulePage | null }) {
  const { dark, toggle } = useTheme();
  const track = "track" in route ? route.track : null;
  const module = "module" in route ? route.module : null;
  const link = (active: boolean) => `flex w-full items-center gap-2.5 rounded-lg px-3 py-1.5 text-left text-sm transition ${active
    ? "bg-sky-50 font-semibold text-sky-900 dark:bg-sky-500/15 dark:text-sky-100" : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800/70"}`;
  return (
    <aside className="sticky top-0 hidden h-dvh w-80 shrink-0 flex-col border-r border-zinc-200 bg-white lg:flex dark:border-zinc-800 dark:bg-zinc-900">
      <a href={href.home()} className="block px-5 pt-6 pb-4">
        <div className="text-lg font-bold tracking-tight">☁️ Nimbus academy</div>
        <div className="text-xs text-zinc-500">Build one product, learn every alternative</div>
      </a>
      <nav className="flex-1 overflow-y-auto px-3 pb-4">
        <a href={href.home()} className={link(route.page === "home")}>🏠 All tracks</a>
        {catalog.tracks.map((t) => (
          <div key={t.id} className="mt-1">
            <a href={href.track(t.id)} className={link(route.page === "track" && track === t.id)}>
              <span>{t.icon}</span><span className="flex-1">{t.title}</span>
              <span className="font-mono text-[11px] text-zinc-400">{t.covered}/{t.items}</span>
            </a>
            {track === t.id && (
              <div className="ml-4 border-l border-zinc-200 pl-2 dark:border-zinc-800">
                {t.modules.map((m) => (
                  <div key={m.id}>
                    {m.status !== "planned" ? (
                      <a href={href.module(t.id, m.id)} className={link(route.page === "module" && module === m.id)}>
                        <span className="font-mono text-xs text-zinc-400">{m.id}</span><span className="flex-1">{m.title}</span><Stars n={m.stars} size="xs" />
                      </a>
                    ) : (
                      <div className="flex gap-2.5 px-3 py-1.5 text-sm text-zinc-400"><span className="font-mono text-xs">{m.id}</span><span className="flex-1">{m.title}</span><Stars n={m.stars} size="xs" /></div>
                    )}
                    {module === m.id && page && <ModuleContents page={page} route={route} link={link} />}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>
      <button onClick={toggle} className="m-3 rounded-lg border border-zinc-200 px-3 py-2 text-sm text-zinc-600 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800">
        {dark ? "☀️ Light mode" : "🌙 Dark mode"}
      </button>
    </aside>
  );
}

function ModuleContents({ page, route, link }: { page: ModulePage; route: Route; link: (active: boolean) => string }) {
  const lab = route.page === "lab" ? route.lab : null;
  const step = route.page === "path" ? route.step ?? "" : null;
  return (
    <div className="mb-2 ml-4 border-l border-zinc-200 pl-2 dark:border-zinc-800">
      <div className="px-3 pt-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-400">Labs</div>
      {page.labs.map((l) => (
        <a key={l.id} href={href.lab(page.track, page.id, l.id)} className={link(lab === l.id)}><span>{l.icon}</span>{l.title}</a>
      ))}
      {page.path.steps.length > 0 && <>
        <div className="px-3 pt-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-400">🏭 Production path</div>
        <a href={href.path(page.track, page.id)} className={link(step === "")}>◎ Overview</a>
        {page.path.steps.map((s) => (
          <a key={s.id} href={href.path(page.track, page.id, s.id)} className={link(step === s.id)}><span>{s.icon}</span>{s.title}</a>
        ))}
      </>}
    </div>
  );
}
