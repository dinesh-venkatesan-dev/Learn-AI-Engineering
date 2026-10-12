"""Triage evaluation: run labelled tickets through the triager and score every field that routing depends on.

  valid        share of replies that matched the schema (a broken reply routes nowhere)
  accuracy     per field: category, urgency, order_id, language, needs_human (exact match with the label)
  all_correct  share of tickets with every scored field right: what an agent would never have to fix
  latency      p50 / p95 per ticket, including time queued behind the rate limiter; wall_s for the whole run
  tokens/cost  per ticket, and the cost of 1,000 tickets at a reference paid price

Sentiment isn't scored: people disagree on it too often for a label to mean much.
"""

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from nimbus.llm import Usage
from nimbus.support.schema import Ticket
from nimbus.support.triage import Triager, TriageResult

FIELDS = ("category", "urgency", "order_id", "language", "needs_human")


@dataclass(frozen=True)
class LabelledTicket:
  ticket: Ticket
  expected: dict


def load_tickets(path: Path) -> list[LabelledTicket]:
  return [LabelledTicket(Ticket(id=t["id"], text=t["text"]), t["expected"]) for t in json.loads(path.read_text())]


def field_matches(result: TriageResult, expected: dict) -> dict[str, bool]:
  got = result.triage.model_dump() if result.triage else {}
  return {f: result.triage is not None and got.get(f) == expected[f] for f in FIELDS}


@dataclass
class TriageReport:
  prompt_version: str
  method: str
  model: str
  tickets: int
  valid: float
  accuracy: dict[str, float]
  all_correct: float
  latency_p50_ms: float
  latency_p95_ms: float
  wall_s: float  # the whole run: what a backlog of this size takes under the rate limit
  usage: Usage
  results: list[tuple[LabelledTicket, TriageResult, dict[str, bool]]]

  def summary(self) -> dict:
    per = self.usage.per_call()
    return {
      "prompt_version": self.prompt_version,
      "method": self.method,
      "model": self.model,
      "tickets": self.tickets,
      "valid": round(self.valid, 3),
      **{f"{f}_acc": round(v, 3) for f, v in self.accuracy.items()},
      "all_correct": round(self.all_correct, 3),
      "latency_p50_ms": round(self.latency_p50_ms),
      "latency_p95_ms": round(self.latency_p95_ms),
      "wall_s": round(self.wall_s, 1),
      "input_tokens": round(per["input_tokens"]),
      "output_tokens": round(per["output_tokens"]),
      "cost_per_1k_usd": round(per["cost_usd"] * 1000, 3),
    }


def evaluate(triager: Triager, labelled: list[LabelledTicket], model: str, concurrency: int = 4) -> TriageReport:
  started = time.perf_counter()
  results = triager.triage_many([lt.ticket for lt in labelled], concurrency)
  wall_s = time.perf_counter() - started
  scored = [(lt, r, field_matches(r, lt.expected)) for lt, r in zip(labelled, results, strict=True)]
  n = len(scored) or 1
  latencies = sorted(r.latency_ms for _, r, _ in scored) or [0.0]
  return TriageReport(
    prompt_version=triager.prompt_version,
    method=triager.method,
    model=model,
    tickets=len(scored),
    valid=sum(1 for _, r, _ in scored if r.triage) / n,
    accuracy={f: sum(1 for *_, m in scored if m[f]) / n for f in FIELDS},
    all_correct=sum(1 for *_, m in scored if all(m.values())) / n,
    latency_p50_ms=statistics.median(latencies),
    latency_p95_ms=latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
    wall_s=wall_s,
    usage=_usage(r for _, r, _ in scored),
    results=scored,
  )


def _usage(results) -> Usage:
  usage = Usage()
  for r in results:
    usage.add(r.raw)
  return usage
