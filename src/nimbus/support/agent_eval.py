"""Agent evaluation: labelled scenarios, each a customer and a request, scored on what the agent did and returned.

tools_ok     every expected tool call made (name + listed arguments), and nothing outside the allowed tools
actions_ok   the expected actions are in state["actions"] (and none when the scenario expects none)
outcome_ok   the SupportOutcome fields the scenario specifies (needs_human, resolved, order ids)
answer_ok    the reply contains the required facts (phrasings, whole numbers)
success      all four
steps        model calls per scenario; tokens and cost
"""

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from nimbus.llm.usage import Usage
from nimbus.support.agent import CustomerContext, SupportOutcome, build_support_agent
from nimbus.support.tool_eval import call_matches
from nimbus.support.tools import SupportStore


def load_scenarios(path: Path) -> list[dict]:
  return json.loads(path.read_text())


@dataclass
class AgentRun:
  outcome: SupportOutcome | None
  actions: list[str]
  tool_calls: list[dict]
  steps: int
  usage: Usage
  ms: float
  error: str | None = None
  messages: list = field(default_factory=list)


def run_scenario(model: BaseChatModel, scenario: dict, output: str = "tool") -> AgentRun:
  store = SupportStore()  # a fresh store per scenario: one scenario's actions never leak into the next
  agent = build_support_agent(model, store, output=output)
  context = CustomerContext(scenario["customer_id"], store.customers.get(scenario["customer_id"], "the customer"))
  started = time.perf_counter()
  try:
    result = agent.invoke({"messages": [HumanMessage(scenario["request"])], "actions": []}, context=context)
  except Exception as error:  # noqa: BLE001 — a failed scenario is a result
    return AgentRun(
      None, [], [], 0, Usage(), (time.perf_counter() - started) * 1000, f"{type(error).__name__}: {error}"[:300]
    )
  ais = [m for m in result["messages"] if isinstance(m, AIMessage)]
  usage = Usage()
  for m in ais:
    usage.add(m)
  calls = [{"name": c["name"], "args": c["args"]} for m in ais for c in m.tool_calls if c["name"] != "SupportOutcome"]
  return AgentRun(
    result.get("structured_response"),
    result.get("actions", []),
    calls,
    len(ais),
    usage,
    (time.perf_counter() - started) * 1000,
    messages=result["messages"],
  )


def score(scenario: dict, run: AgentRun) -> dict:
  if run.outcome is None:
    return {"tools_ok": False, "actions_ok": False, "outcome_ok": False, "answer_ok": False, "success": False}
  expected = scenario["expect"]
  allowed = {e["name"] for e in expected} | set(scenario.get("also_ok", []))
  tools_ok = all(any(call_matches(e, c) for c in run.tool_calls) for e in expected) and all(
    c["name"] in allowed for c in run.tool_calls
  )
  actions = " | ".join(run.actions)
  actions_ok = all(a in actions for a in scenario.get("actions", [])) and not (
    scenario.get("no_actions") and run.actions
  )
  fields = scenario.get("outcome", {})
  outcome_ok = all(getattr(run.outcome, k) == v for k, v in fields.items()) and all(
    o in run.outcome.order_ids for o in scenario.get("order_ids", [])
  )
  reply = (
    run.outcome.reply.casefold().replace("\u2019", "'").replace("\u2018", "'")
  )  # curly apostrophes count as straight
  numbers = {int(n) for n in re.findall(r"\d+", reply.replace(",", ""))}
  answer_ok = (
    (not scenario.get("answer_any") or any(a in reply for a in scenario["answer_any"]))
    and all(n in numbers for n in scenario.get("answer_numbers", []))
    # phrase lists kept missing correct wordings ("you don't have a Nimbus Coffee bean subscription"): a pattern
    and (not scenario.get("answer_regex") or re.search(scenario["answer_regex"], reply) is not None)
  )
  return {
    "tools_ok": tools_ok,
    "actions_ok": actions_ok,
    "outcome_ok": outcome_ok,
    "answer_ok": answer_ok,
    "success": tools_ok and actions_ok and outcome_ok and answer_ok,
  }


@dataclass
class AgentReport:
  variant: str
  rows: list[tuple[dict, AgentRun, dict]]

  def summary(self) -> dict:
    n = max(len(self.rows), 1)
    ms = sorted(r.ms for _, r, _ in self.rows) or [0.0]
    rate = lambda key: round(sum(s[key] for *_, s in self.rows) / n, 3)  # noqa: E731
    return {
      "variant": self.variant,
      "scenarios": len(self.rows),
      **{k: rate(k) for k in ("success", "tools_ok", "actions_ok", "outcome_ok", "answer_ok")},
      "steps": round(sum(r.steps for _, r, _ in self.rows) / n, 2),
      "tokens": round(sum(r.usage.input_tokens + r.usage.output_tokens for _, r, _ in self.rows) / n),
      "cost_per_1k_usd": round(sum(r.usage.cost_usd() for _, r, _ in self.rows) / n * 1000, 3),
      "latency_p50_ms": round(statistics.median(ms)),
    }


def evaluate_agent(variant: str, model: BaseChatModel, scenarios: list[dict], output: str = "tool") -> AgentReport:
  rows = []
  for scenario in scenarios:
    run = run_scenario(model, scenario, output)
    rows.append((scenario, run, score(scenario, run)))
  return AgentReport(variant, rows)


def run_desk_scenario(model: BaseChatModel, scenario: dict, ending: str = "tool") -> AgentRun:
  """The same scenario through the support desk graph (Agents Module 4), scored by the same rules."""
  from nimbus.llm.usage import UsageRecorder
  from nimbus.support.desk import ask, build_support_desk

  store = SupportStore()
  desk = build_support_desk(model, store, ending=ending)
  recorder = UsageRecorder()  # counts the final respond call too (it isn't a message in the conversation)
  started = time.perf_counter()
  try:
    result = ask(desk.with_config(callbacks=[recorder]), store, scenario["customer_id"], scenario["request"])
  except Exception as error:  # noqa: BLE001 — a failed scenario is a result
    return AgentRun(
      None, [], [], 0, Usage(), (time.perf_counter() - started) * 1000, f"{type(error).__name__}: {error}"[:300]
    )
  ticket = result["ticket"]
  outcome = SupportOutcome(
    reply=ticket.reply, resolved=ticket.resolved, needs_human=ticket.needs_human, order_ids=ticket.order_ids
  )
  calls = [
    {"name": c["name"], "args": c["args"]}
    for m in result["messages"]
    if isinstance(m, AIMessage)
    for c in m.tool_calls
    if c["name"] != "final_answer"  # the answer itself, like v1's SupportOutcome call
  ]
  return AgentRun(
    outcome,
    result.get("actions", []),
    calls,
    len(recorder.calls),
    recorder.usage(),
    (time.perf_counter() - started) * 1000,
    messages=result["messages"],
  )
