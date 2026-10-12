// Real source code from the package: highlighted, with its true line numbers and notes pinned under key lines.
import { Fragment, useState } from "react";
import { Highlight, themes } from "prism-react-renderer";
import type { CodeRef } from "../types";
import { useTheme } from "../theme";

export function CodeBlock({ code }: { code: CodeRef }) {
  const { dark } = useTheme();
  const [copied, setCopied] = useState(false);
  const notes = new Map(code.notes.map((n) => [n.line, n.text]));
  const lastLine = code.start + code.source.trimEnd().split("\n").length - 1;

  return (
    <figure className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <figcaption className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 px-5 py-3 dark:border-zinc-800">
        <div className="min-w-0">
          <div className="font-mono text-[15px] font-semibold text-zinc-900 dark:text-zinc-100">{code.symbol}</div>
          <div className="text-sm text-zinc-500">{code.why}</div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <a href={`vscode://file${code.abs_path}:${code.start}`} title="Open in VS Code"
             className="rounded-md border border-zinc-200 px-2 py-1 font-mono text-zinc-600 hover:border-sky-400 hover:text-sky-700 dark:border-zinc-700 dark:text-zinc-400 dark:hover:text-sky-300">
            {code.file}:{code.start}–{lastLine} ↗
          </a>
          <button onClick={() => { navigator.clipboard?.writeText(code.source); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
                  className="rounded-md border border-zinc-200 px-2 py-1 text-zinc-600 hover:border-sky-400 dark:border-zinc-700 dark:text-zinc-400">
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </figcaption>
      <Highlight theme={dark ? themes.vsDark : themes.github} code={code.source.trimEnd()} language="python">
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre className="overflow-x-auto py-3 font-mono text-[13.5px] leading-[1.7]" style={{ background: "transparent" }}>
            {tokens.map((line, i) => {
              const number = code.start + i;
              const note = notes.get(number);
              return (
                <Fragment key={i}>
                  <div {...getLineProps({ line })} className={`flex ${note ? "bg-amber-100/70 dark:bg-amber-400/10" : ""}`}>
                    <span className={`w-14 shrink-0 select-none pr-4 text-right ${note ? "font-semibold text-amber-600 dark:text-amber-400" : "text-zinc-400 dark:text-zinc-600"}`}>{number}</span>
                    <span className="whitespace-pre pr-6">{line.map((token, j) => <span key={j} {...getTokenProps({ token })} />)}</span>
                  </div>
                  {note && (
                    <div className="flex bg-amber-50 dark:bg-amber-400/5">
                      <span className="w-14 shrink-0" />
                      <div className="my-1 mr-6 flex gap-2 rounded-lg border-l-4 border-amber-400 bg-white px-3 py-1.5 font-sans text-[13.5px] leading-snug text-zinc-700 shadow-sm dark:bg-zinc-900 dark:text-zinc-300">
                        <span>💡</span><span>{note}</span>
                      </div>
                    </div>
                  )}
                </Fragment>
              );
            })}
          </pre>
        )}
      </Highlight>
    </figure>
  );
}

/** A short snippet to learn from: highlighted, copyable, no file chrome. */
export function Snippet({ source }: { source: string }) {
  const { dark } = useTheme();
  const [copied, setCopied] = useState(false);
  return (
    <div className="group relative overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950">
      <button onClick={() => { navigator.clipboard?.writeText(source); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
              className="absolute top-2 right-2 rounded-md border border-zinc-200 bg-white px-2 py-0.5 text-xs text-zinc-500 opacity-0 transition group-hover:opacity-100 dark:border-zinc-700 dark:bg-zinc-900">
        {copied ? "Copied" : "Copy"}
      </button>
      <Highlight theme={dark ? themes.vsDark : themes.github} code={source} language="python">
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre className="overflow-x-auto p-4 font-mono text-[14px] leading-[1.7]" style={{ background: "transparent" }}>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>{line.map((token, j) => <span key={j} {...getTokenProps({ token })} />)}</div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
}
