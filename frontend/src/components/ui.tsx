// Small building blocks shared by the pages.
import { useState, type ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900 ${className}`}>{children}</section>;
}

export function Stat({ label, value, tone = "zinc" }: { label: string; value: ReactNode; tone?: "zinc" | "emerald" | "sky" | "amber" | "rose" }) {
  const tones = {
    zinc: "text-zinc-900 dark:text-zinc-100", emerald: "text-emerald-600 dark:text-emerald-400", sky: "text-sky-600 dark:text-sky-400",
    amber: "text-amber-600 dark:text-amber-400", rose: "text-rose-600 dark:text-rose-400",
  };
  return (
    <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className={`text-2xl font-semibold tabular-nums ${tones[tone]}`}>{value}</div>
      <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">{label}</div>
    </div>
  );
}

export function Pills<T extends string | number>({ items, value, onChange, label }: {
  items: { id: T; label: string }[]; value: T; onChange: (id: T) => void; label?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {label && <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">{label}</span>}
      {items.map((item) => (
        <button key={item.id} onClick={() => onChange(item.id)}
                className={`rounded-full border px-3 py-1 text-sm transition ${item.id === value
                  ? "border-sky-500 bg-sky-50 text-sky-800 dark:bg-sky-500/15 dark:text-sky-200"
                  : "border-zinc-200 text-zinc-600 hover:border-zinc-300 dark:border-zinc-700 dark:text-zinc-400"}`}>
          {item.label}
        </button>
      ))}
    </div>
  );
}

export function Text({ children, maxHeight = "max-h-96" }: { children: string; maxHeight?: string }) {
  return <pre className={`${maxHeight} overflow-auto whitespace-pre-wrap rounded-xl bg-zinc-50 p-4 font-mono text-[13px] leading-relaxed text-zinc-800 dark:bg-zinc-950 dark:text-zinc-300`}>{children}</pre>;
}

export function Command({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="group flex items-center justify-between gap-3 rounded-lg bg-zinc-900 px-4 py-2.5 font-mono text-[13px] text-zinc-100 dark:bg-black">
      <span className="truncate"><span className="text-zinc-500">$ </span>{cmd}</span>
      <button onClick={() => { navigator.clipboard?.writeText(cmd); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
              className="shrink-0 text-xs text-zinc-400 hover:text-white">{copied ? "copied" : "copy"}</button>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-2xl border-2 border-dashed border-zinc-200 px-6 py-14 text-center text-zinc-500 dark:border-zinc-800">{children}</div>;
}

export function Coverage({ covered, items, className = "" }: { covered: number; items: number; className?: string }) {
  const share = items ? covered / items : 0;
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
        <div className="h-full rounded-full bg-emerald-500" style={{ width: `${share * 100}%` }} />
      </div>
      <span className="w-24 text-right font-mono text-xs tabular-nums text-zinc-500">{covered}/{items} · {Math.round(share * 100)}%</span>
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string }[]; value: T; onChange: (id: T) => void }) {
  return (
    <div className="flex gap-1 rounded-xl bg-zinc-200/60 p-1 dark:bg-zinc-800/60">
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onChange(t.id)}
                className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition ${value === t.id ? "bg-white shadow-sm dark:bg-zinc-900" : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"}`}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return <div className="text-sm font-semibold uppercase tracking-wider text-sky-600">{children}</div>;
}

const STAR_LABEL = { 3: "Essential for production", 2: "Important: often needed, or depends on the use case", 1: "Good to know: specialised" };

/** How much a topic matters for a production LLM app: 3 essential, 2 important, 1 good to know. */
export function Stars({ n, size = "sm" }: { n: number; size?: "xs" | "sm" | "lg" }) {
  const text = { xs: "text-[10px]", sm: "text-sm", lg: "text-xl" }[size];
  return (
    <span title={STAR_LABEL[n as 1 | 2 | 3]} className={`shrink-0 whitespace-nowrap tracking-tight ${text}`}>
      {[1, 2, 3].map((i) => <span key={i} className={i <= n ? "text-amber-500" : "text-zinc-300 dark:text-zinc-700"}>★</span>)}
    </span>
  );
}
