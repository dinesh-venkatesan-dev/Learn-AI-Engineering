// All tracks, with how much of the old courses' learning each one covers so far.
import { Card, Coverage, Eyebrow, Stars } from "../components/ui";
import { href } from "../router";
import type { Catalog } from "../types";

export function HomePage({ catalog }: { catalog: Catalog }) {
  const modules = catalog.tracks.flatMap((t) => t.modules);
  const built = modules.filter((m) => m.status === "built").length;
  return (
    <div className="space-y-8">
      <header>
        <Eyebrow>Nimbus academy</Eyebrow>
        <h1 className="mt-1 text-4xl font-bold tracking-tight">Build one product. Learn every alternative.</h1>
        <p className="mt-3 text-lg text-zinc-600 dark:text-zinc-400">
          Every module has <b>labs</b> that run and compare each alternative (every loader, every splitter, every search
          method), a <b>⚖️ scorecard</b> that measures them on the same questions, and a <b>🏭 production path</b> that
          upgrades the Nimbus product only when the scorecard proves it.
        </p>
      </header>

      <Card>
        <div className="mb-2 flex items-baseline justify-between">
          <div className="font-semibold">Coverage of the old courses</div>
          <div className="text-sm text-zinc-500">{built} of {modules.length} modules built</div>
        </div>
        <Coverage covered={catalog.covered} items={catalog.items} />
        <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">
          Every learning point from the earlier CLI courses ({catalog.items.toLocaleString()} of them) has exactly one home
          module here. A module counts as built only when its labs and production path cover <i>all</i> of its points, and
          a test fails if one is missed.
        </p>
      </Card>

      <Card>
        <div className="mb-2 font-semibold">How important is each topic for a production app?</div>
        <div className="grid gap-2 sm:grid-cols-3">
          {[3, 2, 1].map((n) => <div key={n} className="flex items-start gap-2 text-sm"><Stars n={n} /><span className="text-zinc-600 dark:text-zinc-400">{catalog.scale[String(n)]}</span></div>)}
        </div>
        <p className="mt-2 text-xs text-zinc-500">Every module and lesson is rated. Short on time? Do the ★★★ topics first; they are what every production LLM app needs.</p>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {catalog.tracks.map((t) => {
          const done = t.modules.filter((m) => m.status === "built").length;
          return (
            <a key={t.id} href={href.track(t.id)}
               className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-3"><span className="text-3xl">{t.icon}</span>
                <div><div className="text-lg font-semibold">{t.title}</div><div className="text-xs text-zinc-500">{done}/{t.modules.length} modules built · {t.modules.filter((m) => m.stars === 3).length} essential (★★★)</div></div></div>
              <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-400">{t.summary}</p>
              <Coverage className="mt-4" covered={t.covered} items={t.items} />
            </a>
          );
        })}
      </div>
    </div>
  );
}
