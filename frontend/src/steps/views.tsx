// What each step produced, drawn so it's easy to inspect.
import { useState } from "react";
import type { ChunkData, CleanData, Doc, EmbedData, LoadData, StoreData } from "../types";
import { Command, Empty, Pills, Stat, Text } from "../components/ui";

const docLabel = (d: Doc) => (d.page && d.source.endsWith(".pdf") ? `${d.source} · p${d.page}` : d.source);
const docItems = (docs: Doc[]) => docs.map((d, i) => ({ id: i, label: docLabel(d) }));

export function LoadView({ data, stats }: { data: LoadData; stats: Record<string, number | string> }) {
  const [pick, setPick] = useState(0);
  const doc = data.documents[Math.min(pick, data.documents.length - 1)];
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3"><Stat label="files" value={stats.files} /><Stat label="documents" value={stats.documents} />
        <Stat label="failed" value={stats.failed} tone={Number(stats.failed) ? "rose" : "zinc"} /></div>
      <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500 dark:bg-zinc-950"><tr><th className="px-4 py-2">Document</th><th className="px-4 py-2">Characters</th><th className="px-4 py-2">metadata</th></tr></thead>
          <tbody>{data.documents.map((d, i) => (
            <tr key={i} onClick={() => setPick(i)} className={`cursor-pointer border-t border-zinc-100 dark:border-zinc-800 ${i === pick ? "bg-sky-50 dark:bg-sky-500/10" : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50"}`}>
              <td className="px-4 py-2 font-medium">{docLabel(d)}</td><td className="px-4 py-2 font-mono tabular-nums">{d.chars.toLocaleString()}</td>
              <td className="px-4 py-2 font-mono text-xs text-zinc-500">{JSON.stringify({ source: d.source, page: d.page })}</td></tr>))}</tbody>
        </table>
      </div>
      {data.unsupported.length > 0 && (
        <div className="rounded-lg bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:bg-amber-400/10 dark:text-amber-200">
          ⚠️ <b>{data.unsupported.length} files have no loader yet</b> and were skipped (Module 2 adds loaders for them):{" "}
          <span className="font-mono text-xs">{data.unsupported.join(", ")}</span>
        </div>
      )}
      {data.failed.map((f) => <div key={f.source} className="rounded-lg bg-rose-50 px-4 py-2 text-sm text-rose-800 dark:bg-rose-500/10 dark:text-rose-200">✗ {f.source}: {f.error}</div>)}
      <div><div className="mb-2 text-sm font-medium">page_content of <b>{docLabel(doc)}</b> (click a row to switch)</div><Text>{doc.text ?? ""}</Text></div>
    </div>
  );
}

export function CleanView({ data, stats }: { data: CleanData; stats: Record<string, number | string> }) {
  const [pick, setPick] = useState(Math.max(0, data.documents.findIndex((d) => d.source.endsWith(".html"))));
  const doc = data.documents[Math.min(pick, data.documents.length - 1)];
  const removed = Number(stats.chars_before) - Number(stats.chars_after);
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3"><Stat label="characters before" value={Number(stats.chars_before).toLocaleString()} />
        <Stat label="characters after" value={Number(stats.chars_after).toLocaleString()} tone="emerald" />
        <Stat label="noise removed" value={`${Math.round((removed / Number(stats.chars_before)) * 100)}%`} tone="amber" /></div>
      <Pills items={docItems(data.documents)} value={pick} onChange={setPick} />
      <div className="grid gap-4 lg:grid-cols-2">
        <div><div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Before · {doc.before.length.toLocaleString()} chars</div><Text>{doc.before}</Text></div>
        <div><div className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-600">After · {doc.after.length.toLocaleString()} chars</div><Text>{doc.after}</Text></div>
      </div>
    </div>
  );
}

const CHUNK_COLORS = ["bg-sky-100 dark:bg-sky-500/20", "bg-violet-100 dark:bg-violet-500/20"];

export function ChunkView({ data, stats }: { data: ChunkData; stats: Record<string, number | string> }) {
  const [pick, setPick] = useState(Math.max(0, data.documents.findIndex((d) => d.source.endsWith(".md"))));
  const doc = data.documents[Math.min(pick, data.documents.length - 1)];
  const mine = data.chunks.filter((c) => c.source === doc.source && c.page === doc.page);
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3"><Stat label="chunks" value={stats.chunks} tone="sky" /><Stat label="chunk size" value={stats.chunk_size} />
        <Stat label="overlap" value={stats.chunk_overlap} /></div>
      <Pills items={docItems(data.documents)} value={pick} onChange={setPick} />
      <p className="text-sm text-zinc-600 dark:text-zinc-400">Each colour is one chunk. <span className="rounded bg-amber-300/80 px-1.5 text-zinc-900">Yellow</span> text is overlap: it appears in two chunks. Grey text was dropped (a fragment shorter than the minimum).</p>
      <pre className="max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-xl border border-zinc-200 p-4 font-mono text-[13px] leading-relaxed dark:border-zinc-800">{highlight(doc.text ?? "", mine)}</pre>
      <div className="grid gap-3 md:grid-cols-2">
        {mine.map((c, i) => (
          <div key={c.start} className={`rounded-xl p-3 text-sm ${CHUNK_COLORS[i % 2]}`}>
            <div className="mb-1 flex justify-between font-mono text-xs"><b>chunk #{c.index}</b><span>{c.chars} chars · starts at {c.start}</span></div>
            <div className="line-clamp-3 text-zinc-700 dark:text-zinc-300">{c.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function highlight(text: string, chunks: { start: number; text?: string }[]) {
  const cover: number[][] = Array.from({ length: text.length }, () => []);
  chunks.forEach((c, i) => { for (let p = c.start; p < Math.min(c.start + (c.text?.length ?? 0), text.length); p++) cover[p].push(i); });
  const parts: { from: number; to: number; ids: number[] }[] = [];
  for (let p = 0; p < text.length; p++) {
    const last = parts[parts.length - 1];
    if (last && last.ids.join() === cover[p].join()) last.to = p + 1;
    else parts.push({ from: p, to: p + 1, ids: cover[p] });
  }
  return parts.map((part) => (
    <span key={part.from} className={!part.ids.length ? "text-zinc-400" : part.ids.length > 1 ? "rounded bg-amber-300/80 text-zinc-900" : CHUNK_COLORS[part.ids[0] % 2]}>
      {text.slice(part.from, part.to)}
    </span>
  ));
}

export function EmbedView({ data, stats }: { data: EmbedData; stats: Record<string, number | string> }) {
  if (!data.vector.length) return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3"><Stat label="texts embedded" value={0} tone="emerald" /><Stat label="dimensions" value={stats.dims} /><Stat label="model" value="bge-small" /></div>
      <Empty><b>Nothing was embedded in this run.</b><br />Every chunk was unchanged since the last run, so it was skipped: no model calls, no cost.<br />Tick <b>Re-embed everything</b> and run again to see a vector.</Empty>
    </div>
  );
  const max = Math.max(...data.vector.map(Math.abs));
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3"><Stat label="texts embedded" value={stats.embedded} tone="sky" /><Stat label="dimensions" value={stats.dims} /><Stat label="model" value="bge-small" /></div>
      <div><div className="mb-2 text-sm font-medium">The first chunk embedded in this run:</div><Text maxHeight="max-h-32">{data.text}</Text></div>
      <div>
        <div className="mb-2 text-sm font-medium">Its vector: {data.vector.length} numbers <span className="text-zinc-500">(<span className="text-rose-500">red</span> positive, <span className="text-sky-500">blue</span> negative; hover for values)</span></div>
        <div className="grid gap-[2px] rounded-xl bg-zinc-100 p-2 dark:bg-zinc-950" style={{ gridTemplateColumns: "repeat(32, minmax(0, 1fr))" }}>
          {data.vector.map((v, i) => {
            const a = Math.min(Math.abs(v) / max, 1);
            const c = v >= 0 ? `rgba(225,29,72,${a})` : `rgba(14,165,233,${a})`;
            return <div key={i} title={`[${i}] ${v}`} className="aspect-square rounded-[3px] bg-white dark:bg-zinc-900" style={{ backgroundImage: `linear-gradient(${c},${c})` }} />;
          })}
        </div>
        <p className="mt-2 font-mono text-xs text-zinc-500">[{data.vector.slice(0, 6).map((x) => x.toFixed(4)).join(", ")}, … {data.vector.length - 6} more]</p>
      </div>
    </div>
  );
}

export function StoreView({ data, stats }: { data: StoreData; stats: Record<string, number | string> }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Stat label="added" value={stats.added} tone="emerald" /><Stat label="updated" value={stats.updated} tone="sky" />
        <Stat label="skipped" value={stats.skipped} tone="zinc" /><Stat label="deleted" value={stats.deleted} tone="rose" />
        <Stat label="rows in table" value={stats.rows} />
      </div>
      <p className="text-sm text-zinc-600 dark:text-zinc-400">Table <code className="font-mono">{data.table}</code> ({data.size}) · {data.records} hashes in the record manager. The first rows, exactly as stored:</p>
      <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
        <table className="w-full text-left text-[13px]">
          <thead className="bg-zinc-50 font-mono text-xs text-zinc-500 dark:bg-zinc-950"><tr>{["langchain_id", "content", "source", "page", "embedding", "langchain_metadata"].map((h) => <th key={h} className="whitespace-nowrap px-3 py-2">{h}</th>)}</tr></thead>
          <tbody>{data.sample.map((r) => (
            <tr key={r.langchain_id} className="border-t border-zinc-100 align-top dark:border-zinc-800">
              <td className="px-3 py-2 font-mono text-xs" title={r.langchain_id}>{r.langchain_id.slice(0, 10)}…</td>
              <td className="min-w-64 px-3 py-2">{r.content}…</td>
              <td className="whitespace-nowrap px-3 py-2 font-mono text-xs">{r.source}</td><td className="px-3 py-2 font-mono">{r.page}</td>
              <td className="whitespace-nowrap px-3 py-2 font-mono text-xs">vector({r.dims})<br />[{r.first_numbers.map((x) => x.toFixed(3)).join(", ")}, …]</td>
              <td className="px-3 py-2 font-mono text-xs">{JSON.stringify(r.langchain_metadata)}</td>
            </tr>))}</tbody>
        </table>
      </div>
      <div className="space-y-2"><div className="text-sm font-medium">Look for yourself:</div>
        <Command cmd={`docker exec -it learn-ai-course-postgres psql -U nimbus -c "select left(content, 60), source, page from ${data.table};"`} />
        <Command cmd={`docker exec -it learn-ai-course-postgres psql -U nimbus -c "\\d ${data.table}"`} /></div>
    </div>
  );
}
