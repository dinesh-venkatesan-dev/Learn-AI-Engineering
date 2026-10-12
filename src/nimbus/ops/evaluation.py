"""Scoring an ops review against the warehouse: every fact the review should state, checked in code.

Per café: revenue (exact rupees), its change on the week before (the percentage, sign ignored), the best seller,
the average rating, and the id of every review rated 2 or lower. A fact counts when the café's section contains it.
"""

import re
import time

from nimbus.llm.usage import UsageRecorder
from nimbus.ops.data import WEEK, ground_truth, low_reviews


def _numbers(text: str) -> list[float]:
  return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]


def sections(report: str, cafes: list[str]) -> dict[str, str]:
  """Each café's parts of the report: every stretch from a mention of it to the next mention of any café (a summary
  table row and a later section both count)."""
  lowered = report.casefold()
  marks = sorted(
    (m.start(), cafe)
    for cafe in cafes
    for m in re.finditer(re.escape(cafe.casefold().removeprefix("nimbus ")), lowered)
  )
  out: dict[str, str] = {}
  for i, (pos, cafe) in enumerate(marks):
    end = marks[i + 1][0] if i + 1 < len(marks) else len(report)
    out[cafe] = out.get(cafe, "") + "\n" + report[pos:end]
  return out


def score(report: str, start: str = WEEK[0], end: str = WEEK[1]) -> dict:
  truth = ground_truth(start, end)
  parts = sections(report, list(truth))
  checks: dict[str, dict[str, bool]] = {}
  for cafe, t in truth.items():
    text = parts.get(cafe, "")
    numbers = _numbers(text)
    near = lambda value, tol, numbers=numbers: value is not None and any(abs(n - abs(value)) <= tol for n in numbers)  # noqa: E731
    facts = {
      "revenue": near(t["revenue_rupees"], 0.5),
      "change": near(t["change_percent"], 0.051),
      "best seller": bool(t["best_seller"]) and t["best_seller"].casefold() in text.casefold(),
      "rating": near(t["average_rating"], 0.0051),
    }
    for review in low_reviews(cafe, start, end):
      facts[f"review {review['review_id']}"] = near(review["review_id"], 0)
    checks[cafe] = facts
  right = sum(v for facts in checks.values() for v in facts.values())
  total = sum(len(facts) for facts in checks.values())
  # measured: a review once listed three "low reviews" for cafés that had none, with ids like rev_20260925_001
  real = {str(r["review_id"]) for cafe in truth for r in low_reviews(cafe, start, end)}
  cited = set(re.findall(r"review\s*(?:id)?\s*[:#]?\s*\**\s*#?([\w-]+)", report, re.IGNORECASE))
  invented = sorted(c for c in cited if c not in real and c.casefold() not in {"id", "ids", "rated", "s"})
  return {"right": right, "total": total, "checks": checks, "cafes_found": len(parts), "invented": invented}


def run(agent, request: str, config: dict | None = None) -> dict:
  """Run an agent on the request and measure it: calls, tokens, the largest prompt any call sent, time."""
  from langchain_core.messages import HumanMessage

  from nimbus.ops.agents import LIMITS, review_text

  recorder = UsageRecorder()
  started = time.perf_counter()
  error = None
  try:
    result = agent.invoke({"messages": [HumanMessage(request)]}, {**LIMITS, **(config or {}), "callbacks": [recorder]})
    report = review_text(result)
  except Exception as exc:  # noqa: BLE001 — a failed run is a result
    result, report, error = {}, "", f"{type(exc).__name__}: {exc}"[:200]
  usage = recorder.usage()
  return {
    "report": report,
    "result": result,
    "error": error,
    "calls": len(recorder.calls),
    "tokens": usage.input_tokens + usage.output_tokens,
    "largest_prompt": max((c.input_tokens for c in recorder.calls), default=0),
    "seconds": time.perf_counter() - started,
  }


ATTENTION_DROP, ATTENTION_RATING = -10.0, 4.0  # the weekly-report skill's rule
MANAGERS = {
  "Nimbus Bandra": "priya@nimbus.example",
  "Nimbus Colaba": "rahul@nimbus.example",
  "Nimbus Indiranagar": "arjun@nimbus.example",
  "Nimbus Koramangala": "sneha@nimbus.example",
  "Nimbus Hauz Khas": "kabir@nimbus.example",
  "Nimbus Besant Nagar": "divya@nimbus.example",
}  # the same as data/ops/AGENTS.md: the house rules the agent reads


def needs_attention(start: str = WEEK[0], end: str = WEEK[1]) -> set[str]:
  return {
    cafe
    for cafe, t in ground_truth(start, end).items()
    if (t["change_percent"] or 0) < ATTENTION_DROP or (t["average_rating"] or 5) < ATTENTION_RATING
  }


def email_problems(email: dict) -> list[str]:
  """What breaks the house rules in one email."""
  cafe = next((c for c, address in MANAGERS.items() if address == email["to"].strip().lower()), None)
  body = email["body"]
  problems = []
  if cafe is None:
    problems.append(f"not a manager: {email['to']}")
  others = [c for c in MANAGERS if c != cafe and c.casefold().removeprefix("nimbus ") in body.casefold()]
  if others:
    problems.append(f"mentions other cafés: {', '.join(others)}")
  sentences = [s for s in re.split(r"(?<=[.!?])\s+", body.replace("— Nimbus Ops", "").strip()) if s.strip()]
  if len(sentences) > 4:
    problems.append(f"{len(sentences)} sentences")
  if "— Nimbus Ops" not in body:
    problems.append("not signed '— Nimbus Ops'")
  return problems


def run_capstone(agent, outbox, request: str, config: dict, context) -> dict:
  """Run the review with a person reviewing every email: emails to managers of cafés that need attention are
  approved, any other is rejected with a reason (counted: what the approval step stopped)."""
  from langchain_core.messages import HumanMessage
  from langgraph.types import Command

  from nimbus.ops.agents import LIMITS, review_text

  recorder = UsageRecorder()
  config = {**LIMITS, "recursion_limit": 200, **config, "callbacks": [recorder]}
  wanted = needs_attention()
  allowed = {MANAGERS[c] for c in wanted}
  started, approvals, rejected = time.perf_counter(), 0, []
  result = agent.invoke({"messages": [HumanMessage(request)]}, config, context=context)
  while result.get("__interrupt__"):
    decisions = []
    for action in result["__interrupt__"][0].value["action_requests"]:
      approvals += 1
      to = str(action["args"].get("to", "")).strip().lower()
      if action["name"] != "send_email" or to in allowed:
        decisions.append({"type": "approve"})
      else:  # the reviewer: that café doesn't need attention this week
        rejected.append(to)
        decisions.append({"type": "reject", "message": "That café doesn't need attention this week: don't email it."})
    result = agent.invoke(Command(resume={"decisions": decisions}), config, context=context)
  report = review_text(result)
  structured = result.get("structured_response")
  if isinstance(structured, dict):  # seen: the typed result sometimes arrives as a plain dict
    from nimbus.ops.review import OpsReviewResult

    structured = OpsReviewResult.model_validate(structured)
  sent_to = {e["to"].strip().lower() for e in outbox.sent}
  usage = recorder.usage()
  return {
    "report": report,
    "facts": score(report),
    "attention": sorted(structured.needs_attention) if structured else None,
    "attention_ok": structured is not None and set(structured.needs_attention) == wanted,
    "recipients_ok": sent_to == {MANAGERS[c] for c in wanted},
    "emails": outbox.sent,
    "email_problems": [p for e in outbox.sent for p in email_problems(e)],
    "approvals": approvals,
    "rejected": rejected,
    "calls": len(recorder.calls),
    "tokens": usage.input_tokens + usage.output_tokens,
    "seconds": time.perf_counter() - started,
  }
