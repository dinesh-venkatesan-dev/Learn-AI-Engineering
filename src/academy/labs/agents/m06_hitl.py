"""Agents with LangGraph · Module 6 labs: interrupt() and resume, reviewing an agent's tool calls (approve, edit,
reject), advanced interrupts (validation loops, parallel questions, breakpoints, editing state), and the rules that
keep interrupts correct. Then: the support desk with approvals, against the desk without them."""

import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from academy.labs.runtime import Metric, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model


def _thread(name: str) -> dict:
  return {"configurable": {"thread_id": name}}


# ── Lab 1: interrupt basics ────────────────────────────────────────────────────
class RefundState(TypedDict, total=False):
  order_id: str
  amount: int
  decision: str
  log: Annotated[list[str], operator.add]


def approval_graph(checkpointer, runs: dict):
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import interrupt

  def ask_manager(state: RefundState) -> dict:
    runs["ask_manager"] = runs.get("ask_manager", 0) + 1  # runs again on resume: the node restarts from the top
    decision = interrupt({"question": f"Refund ₹{state['amount']} for {state['order_id']}?"})  # pause here
    return {"decision": decision, "log": [f"manager said {decision}"]}

  def refund(state: RefundState) -> dict:
    return {"log": ["refunded" if state["decision"] == "yes" else "not refunded"]}

  graph = StateGraph(RefundState).add_node("ask_manager", ask_manager).add_node("refund", refund)
  graph.add_edge(START, "ask_manager").add_edge("ask_manager", "refund").add_edge("refund", END)
  return graph.compile(checkpointer=checkpointer)


def pause(graph, config):
  result = graph.invoke({"order_id": "NC-90112", "amount": 650, "log": []}, config)
  return result["__interrupt__"]  # the value handed to the caller


def resume(graph, config, answer):
  from langgraph.types import Command

  return graph.invoke(Command(resume=answer), config)  # interrupt() returns the answer; the node continues


def without_a_checkpointer():
  from langgraph.types import Command

  graph = approval_graph(None, {})
  paused = "__interrupt__" in graph.invoke({"order_id": "NC-90112", "amount": 650, "log": []})  # it still pauses...
  try:
    graph.invoke(Command(resume="yes"))  # ...but nothing was saved to resume from
  except RuntimeError as error:
    return f"paused: {paused}; resume: {error}"
  return f"paused: {paused}; resume worked"


def while_paused(graph, config):
  snapshot = graph.get_state(config)
  return snapshot.next, [i.value for i in snapshot.interrupts]  # what's waiting, and the question


@lab(
  "agents.interrupt-basics",
  "interrupt(): pause a graph, ask a person, resume with their answer",
  calls="No model calls",
)
def interrupt_basics():
  """A refund that waits for a manager: the pause, what's saved while it waits, the resume, and the node that runs
  twice."""
  from langgraph.checkpoint.memory import InMemorySaver

  runs: dict = {}
  graph = approval_graph(InMemorySaver(), runs)
  config = _thread("refund-1")
  paused = pause(graph, config)
  nxt, waiting = while_paused(graph, config)
  final = resume(graph, config, "yes")
  no_saver = without_a_checkpointer()
  return [
    table(
      ["Step", "Result"],
      [
        ["invoke → paused", str([p.value for p in paused])],
        ["while paused: next", str(nxt)],
        ["while paused: interrupts", str(waiting)],
        ["resume with Command(resume='yes')", str(final["log"])],
        ["ask_manager ran", f"{runs['ask_manager']} times"],
        ["without a checkpointer", no_saver],
      ],
    ),
    callout(
      f"interrupt(value) stops the run and returns value to the caller; Command(resume=answer) continues it, and "
      f"interrupt() returns the answer. The paused run lives in the checkpoint (minutes or days), so a checkpointer "
      f"is required. On resume the node starts again from the top: ask_manager ran {runs['ask_manager']} times, so "
      "anything before interrupt() happens twice (lab 4).",
      "decision",
      "interrupt()",
    ),
  ] + [
    example("lg.l34.1", "interrupt(value) pauses", pause, "", [code(str([p.value for p in paused]), "python")]),
    example("lg.l34.2", "Resume with Command(resume=...)", resume, "", [table(["log"], [[x] for x in final["log"]])]),
    example("lg.l34.3", "A checkpointer is required", without_a_checkpointer, "", [code(no_saver, "text")]),
    example(
      "lg.l34.4",
      "The node runs again from the top",
      approval_graph,
      "",
      [table(["ask_manager executions"], [[runs["ask_manager"]]])],
    ),
    example(
      "lg.l34.5",
      "state.next and state.interrupts",
      while_paused,
      "",
      [table(["next", "interrupts"], [[str(nxt), str(waiting)]])],
    ),
  ]


# ── Lab 2: approve, edit, reject ───────────────────────────────────────────────
REFUND_REQUEST = ("c-6060", "I'd like a refund for order NC-90112, it's still unopened.")
DECISIONS = {
  "approve": {"decision": "approve"},
  "edit": {
    "decision": "edit",
    "args": {"order_id": "NC-90112", "amount_rupees": 550},
    "reason": "the delivery fee stays",
  },
  "reject": {"decision": "reject", "reason": "the warehouse received the bag opened"},
}


def review_node_logic():
  from nimbus.support.approvals import REVIEWERS, review_calls

  return REVIEWERS, review_calls  # sensitive tools and who reviews them; review_calls asks once per sensitive call


def run_with_decision(desk, store, decision, thread):
  from langgraph.types import Command

  from nimbus.support.agent import CustomerContext

  cid, text = REFUND_REQUEST
  context = CustomerContext(cid, store.customers[cid])
  result = desk.invoke(
    {"messages": [HumanMessage(text)], "customer_id": cid, "actions": []}, _thread(thread), context=context
  )
  asked = []
  while "__interrupt__" in result:
    asked.append(result["__interrupt__"][0].value["question"])
    result = desk.invoke(Command(resume=decision), _thread(thread), context=context)
  return asked, result


def safe_call(desk, store):
  from nimbus.support.agent import CustomerContext

  context = CustomerContext("c-6060", store.customers["c-6060"])
  payload = {"messages": [HumanMessage("What's the status of NC-90112?")], "customer_id": "c-6060", "actions": []}
  result = desk.invoke(payload, _thread("safe"), context=context)
  return "__interrupt__" in result, result["ticket"].reply  # a lookup runs without asking


@lab(
  "agents.approve-edit-reject",
  "Review an agent's tool calls: approve, edit or reject",
  uses_model=True,
  calls="~15 calls to Mistral",
)
def approve_edit_reject():
  """The support desk with approvals, on one refund request answered three ways by a teammate, and a lookup that
  needs no approval."""
  from nimbus.support.desk import build_support_desk
  from nimbus.support.memory import memory_checkpointer
  from nimbus.support.tools import SupportStore

  rows, results = [], {}
  for name, decision in DECISIONS.items():
    store = SupportStore()  # a fresh order system per run
    desk = build_support_desk(get_chat_model(), store, checkpointer=memory_checkpointer(), refunds=True, review=True)
    asked, result = run_with_decision(desk, store, decision, f"refund-{name}")
    results[name] = result
    rows.append([name, "; ".join(asked), "; ".join(result.get("actions", [])) or "—", result["ticket"].reply])
  store = SupportStore()
  desk = build_support_desk(get_chat_model(), store, checkpointer=memory_checkpointer(), refunds=True, review=True)
  paused, safe_reply = safe_call(desk, store)
  trace = [[m.type, (m.text or str(getattr(m, "tool_calls", "")))[:160]] for m in results["reject"]["messages"]]
  return [
    table(["Decision", "the teammate was asked", "actions", "reply"], rows, title=REFUND_REQUEST[1]),
    table(["Lookup", "paused?", "reply"], [["What's the status of NC-90112?", paused, safe_reply]]),
    table(
      ["Message", "Content"], trace, title="The rejected run: the call is answered with the reason, the agent replies"
    ),
    code(desk.get_graph().draw_mermaid(), "text", title="agent → review → tools"),
    callout(
      "A review node sits between the model and the tools. Sensitive calls (refunds, subscription pauses) pause "
      "for a person; lookups go straight through. Approve runs the call as written; edit replaces its arguments "
      "in the same message (same id, so add_messages replaces it); reject answers the call with the reason instead "
      "of running it, so the conversation stays valid and the agent tells the customer. "
      + "Actions per decision: "
      + "; ".join(f"{n}: {', '.join(r.get('actions', [])) or 'none'}" for n, r in results.items()).rstrip(".")
      + ".",
      "decision",
      "Approve, edit, reject",
    ),
  ] + [
    example(
      "lg.l35.1",
      "A review node between model and tools",
      review_node_logic,
      "The review node is in nimbus/support/desk.py; the logic in approvals.py.",
      [code(desk.get_graph().draw_mermaid(), "text")],
    ),
    example(
      "lg.l35.2",
      "Approve",
      run_with_decision,
      "",
      [
        table(
          ["actions", "reply"], [[", ".join(results["approve"].get("actions", [])), results["approve"]["ticket"].reply]]
        )
      ],
    ),
    example(
      "lg.l35.3",
      "Edit: same tool-call id, new arguments",
      run_with_decision,
      "",
      [table(["actions", "reply"], [[", ".join(results["edit"].get("actions", [])), results["edit"]["ticket"].reply]])],
    ),
    example(
      "lg.l35.4", "Reject: a ToolMessage with the reason", run_with_decision, "", [table(["Message", "Content"], trace)]
    ),
    example("lg.l35.5", "Safe tools don't ask", safe_call, "", [table(["paused?", "reply"], [[paused, safe_reply]])]),
  ]


# ── Lab 3: advanced interrupts ─────────────────────────────────────────────────
class Order(TypedDict, total=False):
  item: str
  quantity: int
  answers: Annotated[list[str], operator.add]


def validation_loop():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph
  from langgraph.types import Command, interrupt

  def ask_quantity(state: Order) -> dict:
    question = f"How many {state['item']} bags?"
    while True:  # ask again until the answer is valid
      answer = interrupt(question)
      if str(answer).isdigit() and 1 <= int(answer) <= 10:
        return {"quantity": int(answer)}
      question = f"'{answer}' isn't a number from 1 to 10. How many {state['item']} bags?"

  graph = (
    StateGraph(Order)
    .add_node("ask_quantity", ask_quantity)
    .add_edge(START, "ask_quantity")
    .compile(checkpointer=InMemorySaver())
  )
  config = _thread("qty")
  result = graph.invoke({"item": "Nilgiri Sunrise", "answers": []}, config)
  dialogue = []
  for answer in ("lots", "50", "3"):
    dialogue.append((result["__interrupt__"][0].value, answer))
    result = graph.invoke(Command(resume=answer), config)
  return dialogue, result["quantity"]


def parallel_questions():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph
  from langgraph.types import Command, interrupt

  def ask(field):
    def node(state: Order) -> dict:
      return {"answers": [f"{field}={interrupt(f'Preferred {field}?')}"]}

    return node

  builder = StateGraph(Order)
  for field in ("grind", "delivery_day"):
    builder.add_node(f"ask_{field}", ask(field)).add_edge(START, f"ask_{field}")
  graph = builder.compile(checkpointer=InMemorySaver())
  config = _thread("prefs")
  pending = graph.invoke({"answers": []}, config)["__interrupt__"]  # both questions at once
  replies = {"Preferred grind?": "espresso", "Preferred delivery_day?": "Saturday"}
  result = graph.invoke(Command(resume={p.id: replies[p.value] for p in pending}), config)  # answered by id
  return [p.value for p in pending], sorted(result["answers"])


def breakpoint_and_edit():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import END, START, StateGraph

  def charge(state: Order) -> dict:
    return {"answers": [f"charged ₹{state['quantity'] * 720} for {state['quantity']} bags"]}

  graph = StateGraph(Order).add_node("charge", charge).add_edge(START, "charge").add_edge("charge", END)
  graph = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["charge"])  # a static breakpoint
  config = _thread("debug")
  graph.invoke({"item": "Nilgiri Sunrise", "quantity": 30, "answers": []}, config)
  stopped = graph.get_state(config)
  graph.update_state(config, {"quantity": 3})  # 30 bags looks wrong: fix it while paused
  return stopped.next, stopped.values["quantity"], graph.invoke(None, config)["answers"]


@lab(
  "agents.advanced-interrupts",
  "Advanced interrupts: validation loops, parallel questions, breakpoints, editing state",
  calls="No model calls",
)
def advanced_interrupts():
  """A quantity asked until it's valid, two questions from parallel nodes answered together, and a static
  breakpoint before a charge with the state fixed while paused."""
  dialogue, quantity = validation_loop()
  pending, answers = parallel_questions()
  nxt, before, charged = breakpoint_and_edit()
  return [
    table(["Asked", "Answer"], [list(d) for d in dialogue], title=f"Asked until valid: quantity = {quantity}"),
    table(["Waiting at once", "Saved answers"], [[", ".join(pending), ", ".join(answers)]]),
    table(
      ["Stopped before", "quantity then", "after update_state(3) and continuing"], [[str(nxt), before, charged[0]]]
    ),
    callout(
      "interrupt() inside a loop asks again until the answer is valid (the desk does this when a teammate edits a "
      "refund above the order total). Parallel nodes can be waiting at once; resume them together with a dict of "
      "interrupt id → answer. interrupt_before/after are breakpoints that need no code change, and update_state "
      "fixes the state while it's paused.",
      "decision",
      "Advanced interrupts",
    ),
  ] + [
    example(
      "lg.l36.1",
      "Ask again until valid",
      validation_loop,
      "",
      [table(["asked", "answer"], [list(d) for d in dialogue])],
    ),
    example(
      "lg.l36.2",
      "Several interrupts at once",
      parallel_questions,
      "",
      [table(["waiting", "answers"], [[", ".join(pending), ", ".join(answers)]])],
    ),
    example("lg.l36.3", "Static breakpoints", breakpoint_and_edit, "", [table(["stopped before"], [[str(nxt)]])]),
    example(
      "lg.l36.4",
      "Edit the state while paused",
      breakpoint_and_edit,
      "",
      [table(["before", "after"], [[before, charged[0]]])],
    ),
  ]


# ── Lab 4: interrupt rules ─────────────────────────────────────────────────────
class RuleState(TypedDict, total=False):
  answer: Any
  name: str
  size: str


class RefundDecision(BaseModel):
  approved: bool
  amount: int = Field(ge=0, le=650, description="Rupees to refund, at most the order total")


def one_node(node):
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph

  return StateGraph(RuleState).add_node("node", node).add_edge(START, "node").compile(checkpointer=InMemorySaver())


def typed_answers():
  from langgraph.types import Command, interrupt

  def review(state: RuleState) -> dict:
    return {"answer": interrupt({"question": "Refund NC-90112 (₹650)?"}, response_schema=RefundDecision)}

  graph = one_node(review)
  paused = graph.invoke({}, _thread("typed"))["__interrupt__"][0]
  try:
    graph.invoke(Command(resume={"approved": True, "amount": 9000}), _thread("typed"))
    rejected = "accepted (!)"
  except ValidationError as error:
    rejected = error.errors()[0]["msg"]  # the run stays paused
  good = graph.invoke(Command(resume={"approved": True, "amount": 650}), _thread("typed"))["answer"]
  return list(paused.response_schema["properties"]), rejected, good


def side_effects():
  from langgraph.types import Command, interrupt

  emails: dict[str, list] = {"before": [], "after": []}

  def before(state: RuleState) -> dict:
    emails["before"].append("approval requested")  # runs again on resume: two emails
    return {"answer": interrupt("Approve?")}

  def after(state: RuleState) -> dict:
    answer = interrupt("Approve?")
    emails["after"].append(f"decision: {answer}")  # after interrupt(): runs once, when the answer exists
    return {"answer": answer}

  for name, node in (("before", before), ("after", after)):
    graph = one_node(node)
    graph.invoke({}, _thread(name))
    graph.invoke(Command(resume="yes"), _thread(name))
  return emails


def order_of_interrupts():
  from langgraph.types import Command, interrupt

  runs = {"bad": 0}

  def bad(state: RuleState) -> dict:
    runs["bad"] += 1
    name = interrupt("Your name?") if runs["bad"] == 1 else "(skipped)"  # changes between executions!
    return {"name": name, "size": interrupt("Cup size?")}

  def good(state: RuleState) -> dict:
    return {"name": interrupt("Your name?"), "size": interrupt("Cup size?")}  # always both, always in this order

  out = {}
  for label, node in (("conditional", bad), ("fixed order", good)):
    graph = one_node(node)
    graph.invoke({}, _thread(label))
    graph.invoke(Command(resume="Meera"), _thread(label))
    result = graph.invoke(Command(resume="large"), _thread(label))
    out[label] = {"name": result.get("name"), "size": result.get("size")}
  return out


def swallowed_by_try():
  from langgraph.types import interrupt

  def node(state: RuleState) -> dict:
    try:
      answer = interrupt("Approve?")
    except Exception:  # noqa: BLE001 - catches LangGraph's pause signal too
      answer = "(no answer)"
    return {"answer": answer}

  result = one_node(node).invoke({}, _thread("try"))
  return "__interrupt__" in result, result.get("answer")


def unserialisable_payload():
  from langgraph.types import interrupt

  def node(state: RuleState) -> dict:
    return {"answer": interrupt({"question": "Approve?", "check": lambda x: x in ("yes", "no")})}

  try:
    one_node(node).invoke({}, _thread("payload"))
  except Exception as error:  # noqa: BLE001 - the error is the lesson
    return f"{type(error).__name__}: {str(error)[:100]}"
  return "no error"


def subgraph_as_function():
  from langgraph.types import Command, interrupt

  parent_runs = {"count": 0}
  child = one_node(lambda s: {"answer": interrupt("Child question?")})

  def parent_node(state: RuleState, config) -> dict:
    parent_runs["count"] += 1
    return {"answer": child.invoke({}, config)["answer"]}  # a subgraph called inside a node

  parent = one_node(parent_node)
  parent.invoke({}, _thread("parent"))
  result = parent.invoke(Command(resume="child answer"), _thread("parent"))
  return result["answer"], parent_runs["count"]


@lab("agents.interrupt-rules", "Interrupt rules, and typed answers with response_schema", calls="No model calls")
def interrupt_rules():
  """Six rules, each shown by a small graph that breaks it and one that doesn't."""
  fields, rejected, good = typed_answers()
  emails = side_effects()
  orders = order_of_interrupts()
  paused, swallowed = swallowed_by_try()
  payload = unserialisable_payload()
  child_answer, parent_count = subgraph_as_function()
  return [
    table(
      ["Rule", "What happened"],
      [
        ["response_schema", f"fields for a form: {fields}; an amount of 9000 → {rejected}; a valid answer → {good!r}"],
        ["side effects before interrupt()", f"before: {emails['before']}; after: {emails['after']}"],
        ["interrupts matched by order", f"conditional: {orders['conditional']}; fixed: {orders['fixed order']}"],
        ["try/except around interrupt()", f"paused: {paused}; answer: {swallowed}"],
        ["payload with a function", payload],
        ["subgraph called inside a node", f"answer: {child_answer}; the parent node ran {parent_count} times"],
      ],
    ),
    callout(
      "Declare the answer's shape with response_schema (the desk's Review: approve, edit or reject). Anything "
      "before interrupt() runs twice, so make it idempotent or move it after; several interrupts in a node are "
      "matched by order, so never skip one conditionally; a bare try/except swallows the pause; payloads are saved "
      "in the checkpoint, so plain data only; a subgraph called inside a node re-runs the parent node on resume.",
      "decision",
      "Interrupt rules",
    ),
  ] + [
    example(
      "lg.l37.1",
      "response_schema",
      typed_answers,
      "",
      [table(["fields", "bad answer", "good answer"], [[str(fields), rejected, repr(good)]])],
    ),
    example(
      "lg.l37.2",
      "Side effects run twice",
      side_effects,
      "",
      [table(["where", "emails"], [[k, str(v)] for k, v in emails.items()])],
    ),
    example(
      "lg.l37.3",
      "Matched by order",
      order_of_interrupts,
      "",
      [table(["node", "result"], [[k, str(v)] for k, v in orders.items()])],
    ),
    example(
      "lg.l37.4",
      "Never wrap interrupt() in try/except",
      swallowed_by_try,
      "",
      [table(["paused?", "answer"], [[paused, swallowed]])],
    ),
    example("lg.l37.5", "Payloads must be serialisable", unserialisable_payload, "", [code(payload, "text")]),
    example(
      "lg.l37.6",
      "A subgraph called as a function",
      subgraph_as_function,
      "",
      [table(["answer", "parent node runs"], [[child_answer, parent_count]])],
    ),
  ]


# ── Production path: approvals vs none ─────────────────────────────────────────
def evaluate_approvals() -> list[dict]:
  from nimbus.support.approval_eval import load_scenarios, run_scenario, score

  model, rows = get_chat_model(), []
  for scenario in load_scenarios(PROJECT_DIR / "data" / "eval" / "approval_scenarios.json"):
    for review in (False, True):
      run = run_scenario(model, scenario, review)
      rows.append({"review": review, "scenario": scenario, "run": run, "score": score(scenario, run)})
  return rows


@lab(
  "agents.approvals-eval",
  "Evaluate: the support desk with approvals",
  uses_model=True,
  calls="~60 calls to Mistral: 8 requests × 2 desks, about 3 minutes",
)
def approvals_eval():
  """Eight requests that lead to a subscription pause or a refund, each with a scripted reviewer who approves, edits
  or rejects; through the desk with refunds and no approvals, and with approvals."""
  started = time.perf_counter()
  rows = evaluate_approvals()
  seconds = time.perf_counter() - started
  out = []
  for review in (False, True):
    mine = [r for r in rows if r["review"] == review]
    out.append(
      {
        "variant": "approvals" if review else "no approvals",
        "actions_ok": sum(r["score"]["actions_ok"] for r in mine) / len(mine),
        "unrequested": sum(len(r["score"]["unrequested"]) for r in mine),
        "reply_ok": sum(r["score"]["reply_ok"] for r in mine) / len(mine),
        "asked": round(sum(len(r["run"]["asked"]) for r in mine) / len(mine), 2),
        "calls": round(sum(r["run"]["calls"] for r in mine) / len(mine), 2),
      }
    )
  detail = [
    [
      "approvals" if r["review"] else "none",
      r["scenario"]["id"],
      r["scenario"]["request"],
      "; ".join(r["run"]["asked"]) or "—",
      "; ".join(r["run"]["actions"]) or "—",
      "✓" if r["score"]["actions_ok"] and r["score"]["reply_ok"] else "✗",
      r["run"]["reply"],
    ]
    for r in rows
  ]
  none, approvals = out
  return [
    scorecard(
      out,
      [
        Metric("actions_ok", "right actions (after review)"),
        Metric("unrequested", "actions nobody wanted", "lower", "int"),
        Metric("reply_ok", "reply matches what happened"),
        Metric("asked", "questions to a person", "none", "text"),
        Metric("calls", "model calls", "lower", "text"),
      ],
      title=f"8 requests · {seconds:.0f} s",
    ),
    table(["Desk", "id", "Request", "asked", "actions", "ok", "reply"], detail),
    callout(
      f"Without approvals the desk ran {none['unrequested']} action(s) nobody wanted and couldn't take a reviewer's "
      f"edit; with approvals {approvals['unrequested']}, at {approvals['asked']} questions to a person per request. "
      "The refund tool also checks the policy in code, so even an approved refund outside the 14-day window "
      "fails (tests/test_approvals.py). Production enables approvals: the customer confirms subscription pauses, a "
      "teammate approves every refund.",
      "decision" if approvals["unrequested"] <= none["unrequested"] else "warning",
      "Ship approvals",
    ),
  ]
