// How metric values are shown everywhere.
import type { MetricSpec } from "./types";

export function formatMetric(value: unknown, format: MetricSpec["format"]): string {
  if (typeof value !== "number") return String(value ?? "—");
  switch (format) {
    case "ratio": return value.toFixed(2);
    case "percent": return `${Math.round(value * 100)}%`;
    case "ms": return value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value).toLocaleString()} ms`;
    case "int": return Math.round(value).toLocaleString();
    case "usd": return `$${value.toFixed(3)}`;
    default: return String(value);
  }
}
