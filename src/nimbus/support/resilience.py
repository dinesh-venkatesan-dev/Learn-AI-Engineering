"""The outage drill: run labelled tickets through the support pipeline under a given condition and score it.

success      share of tickets that got a triage and a reply (what customers experience)
category     triage category accuracy, over all tickets (a failed ticket counts as wrong)
latency      p50 / p95 per ticket, retries and fallbacks included
answered_by  which model produced each call: proof the fallback actually ran
tokens/cost  per ticket, at a reference paid price
"""

import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.runnables import Runnable

from nimbus.llm.usage import Usage
from nimbus.support.evaluation import LabelledTicket
from nimbus.support.pipeline import PipelineResult, run


@dataclass
class DrillReport:
  condition: str
  tickets: int
  success: float
  category_acc: float
  latency_p50_ms: float
  latency_p95_ms: float
  answered_by: dict[str, int]
  usage: Usage
  results: list[tuple[LabelledTicket, PipelineResult]]

  def summary(self) -> dict:
    per = self.usage.cost_usd() / max(self.tickets, 1)
    return {
      "condition": self.condition,
      "tickets": self.tickets,
      "success": round(self.success, 3),
      "category_acc": round(self.category_acc, 3),
      "latency_p50_ms": round(self.latency_p50_ms),
      "latency_p95_ms": round(self.latency_p95_ms),
      "answered_by": dict(self.answered_by),
      "tokens_per_ticket": round((self.usage.input_tokens + self.usage.output_tokens) / max(self.tickets, 1)),
      "cost_per_1k_usd": round(per * 1000, 3),
    }


def drill(condition: str, pipeline: Runnable, labelled: list[LabelledTicket], concurrency: int = 3) -> DrillReport:
  with ThreadPoolExecutor(max_workers=concurrency) as pool:
    results = list(pool.map(lambda lt: run(pipeline, lt.ticket.id, lt.ticket.text), labelled))
  pairs = list(zip(labelled, results, strict=True))
  n = len(pairs) or 1
  latencies = sorted(r.ms for _, r in pairs) or [0.0]
  calls = [c for _, r in pairs for c in r.calls]
  return DrillReport(
    condition=condition,
    tickets=len(pairs),
    success=sum(1 for _, r in pairs if r.triage is not None and r.reply) / n,
    category_acc=sum(1 for lt, r in pairs if r.triage and r.triage.category == lt.expected["category"]) / n,
    latency_p50_ms=statistics.median(latencies),
    latency_p95_ms=latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
    answered_by=Counter(c.model for c in calls),
    usage=Usage(
      input_tokens=sum(c.input_tokens for c in calls),
      output_tokens=sum(c.output_tokens for c in calls),
      calls=len(calls),
    ),
    results=pairs,
  )
