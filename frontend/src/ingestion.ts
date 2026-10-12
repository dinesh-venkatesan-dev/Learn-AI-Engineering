// The Nimbus ingestion run, kept above the pages so its results survive navigation.
import { useEffect, useRef, useState } from "react";
import { getLastRun, runIngestion, type IngestOptions } from "./api";
import type { Report, StepId, StepResult } from "./types";

export type Run = {
  status: "idle" | "running" | "done" | "error"; results: Partial<Record<StepId, StepResult>>; report?: Report; error?: string;
};

export function useIngestion() {
  const [run, setRun] = useState<Run>({ status: "idle", results: {} });
  const [options, setOptions] = useState<IngestOptions>({ chunkSize: 300, chunkOverlap: 50, force: false });
  const stop = useRef<(() => void) | null>(null);

  useEffect(() => {
    // Show the last run again after a page refresh (the server keeps it in memory).
    getLastRun().then((last) => {
      if (!last) return;
      setRun((current) => current.status !== "idle" ? current
        : { status: "done", report: last.report, results: Object.fromEntries(last.steps.map((s) => [s.step, s])) });
      setOptions((current) => ({ ...current, chunkSize: last.settings.chunk_size, chunkOverlap: last.settings.chunk_overlap }));
    }, () => { /* no previous run */ });
    return () => stop.current?.();
  }, []);

  function start() {
    stop.current?.();
    setRun({ status: "running", results: {} });
    stop.current = runIngestion(options, {
      onStep: (r) => setRun((prev) => ({ ...prev, results: { ...prev.results, [r.step]: r } })),
      onDone: (report) => setRun((prev) => ({ ...prev, status: "done", report })),
      onError: (error) => setRun((prev) => ({ ...prev, status: "error", error })),
    });
  }

  return { run, options, setOptions, start };
}
export type Ingestion = ReturnType<typeof useIngestion>;
