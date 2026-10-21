// "What the model saw": every chat-model call a lab made, as the model received it, and what it answered.
import { useState } from "react";
import type { ModelCall, ModelMessage } from "../types";
import { Empty } from "./ui";

const ROLE = {
  system: { label: "system", style: "bg-violet-100 text-violet-800 dark:bg-violet-500/20 dark:text-violet-200" },
  human: { label: "user", style: "bg-sky-100 text-sky-800 dark:bg-sky-500/20 dark:text-sky-200" },
  ai: { label: "assistant", style: "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-200" },
  tool: { label: "tool result", style: "bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200" },
} as Record<string, { label: string; style: string }>;
const FOLD = 700;

function Message({ m, reply = false }: { m: ModelMessage; reply?: boolean }) {
  const [open, setOpen] = useState(false);
  const role = ROLE[m.role] ?? { label: m.role, style: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300" };
  const long = m.text.length > FOLD;
  return (
    <div className={`rounded-xl border p-3 ${reply ? "border-emerald-300 dark:border-emerald-500/40" : "border-zinc-200 dark:border-zinc-800"}`}>
      <div className="mb-1.5 flex items-center gap-2 text-xs">
        <span className={`rounded px-1.5 py-0.5 font-semibold ${role.style}`}>{reply ? "reply" : role.label}</span>
        {m.name ? <span className="font-mono text-zinc-500">{m.name}</span> : null}
        <span className="text-zinc-400">{m.text.length.toLocaleString()} characters</span>
      </div>
      {m.text ? <pre className="whitespace-pre-wrap break-words font-mono text-[12.5px] leading-relaxed text-zinc-700 dark:text-zinc-300">{long && !open ? m.text.slice(0, FOLD) + " …" : m.text}</pre> : null}
      {long ? <button onClick={() => setOpen(!open)} className="mt-1 text-xs font-medium text-sky-600 hover:underline">{open ? "show less" : `show all ${m.text.length.toLocaleString()} characters`}</button> : null}
      {m.tool_calls?.map((c, i) => (
        <div key={i} className="mt-2 rounded-lg bg-zinc-50 px-2.5 py-1.5 font-mono text-xs dark:bg-zinc-800/60">
          🔧 <b>{c.name}</b>({JSON.stringify(c.args)})
        </div>))}
    </div>
  );
}

export function ModelCalls({ calls, total, notRunYet }: { calls?: ModelCall[]; total?: number; notRunYet: boolean }) {
  if (notRunYet) return <Empty>Press <b>▶ Run lab</b>: every prompt the lab sends to the model, and every reply, appears here.</Empty>;
  if (!calls?.length) return <Empty>The last run made no model calls: its answers came from the lab's cache.</Empty>;
  return (
    <div className="space-y-8">
      <p className="text-zinc-600 dark:text-zinc-400">
        Exactly what the model received in your last run, message by message, and what it answered.
        {total && total > calls.length ? ` Showing the first ${calls.length} of ${total} calls.` : ""}
      </p>
      {calls.map((c, i) => (
        <section key={i} className="space-y-2">
          <h3 className="flex flex-wrap items-baseline gap-x-3 font-semibold">
            Call {i + 1}
            <span className="font-mono text-xs font-normal text-zinc-500">
              {c.model}{c.tokens_in != null ? ` · ${c.tokens_in.toLocaleString()} tokens in → ${c.tokens_out ?? 0} out` : ""}{c.seconds != null ? ` · ${c.seconds} s` : ""}
            </span>
          </h3>
          {c.messages.map((m, j) => <Message key={j} m={m} />)}
          {c.reply ? <Message m={c.reply} reply /> : null}
          {c.error ? <div className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800 dark:bg-rose-500/10 dark:text-rose-200">{c.error}</div> : null}
        </section>))}
    </div>
  );
}
