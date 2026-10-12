"""Support graph v1: the support pipeline as an explicit LangGraph graph.

    START ─┬─ triage ──────────┬─ draft_reply → END
           └─ look_up_order ───┘

The LCEL pipeline (LangChain track, Module 2) runs triage, then the reply. As a graph, the order lookup starts at
the same time as triage (it only needs the ticket text), the reply waits for both, and it now knows the order's
real status. Every node returns a partial update; `steps` collects what each node did (operator.add), so a run
can be streamed and inspected node by node.
"""

import operator
import re
from datetime import date
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.types import Command, Send

from nimbus.support.pipeline import FIXED_REPLIES, reply_step, triage_step
from nimbus.support.schema import TicketTriage
from nimbus.support.tools import REFUND_WINDOW_DAYS, SupportStore

ORDER_ID = re.compile(r"\bNC-\d{4,6}\b", re.IGNORECASE)


class SupportState(TypedDict, total=False):
  ticket: str
  history: list
  triage: TicketTriage
  order: dict | None  # the order record, when the ticket names one that exists
  order_note: str  # what the reply may say about it
  reply: str
  steps: Annotated[list[str], operator.add]  # every node adds a line: the run's trail


def order_note(order_id: str | None, order: dict | None) -> str:
  if order_id is None:
    return ""
  if order is None:
    return f"No order {order_id} exists: ask the customer to check the number."
  delivered = f", delivered {order['delivered_on']}" if order.get("delivered_on") else ""
  return f"Order {order['order_id']} on file: status {order['status']}, ordered {order['ordered_on']}{delivered}."


def build_support_graph(
  model: BaseChatModel,
  store: SupportStore,
  fallback: BaseChatModel | None = None,
  prompt_version: str = "v2",
  method: str = "json_schema",
):
  triage_chain = triage_step(model, prompt_version, method)
  reply_chain = reply_step(model)
  if fallback is not None:
    triage_chain = triage_chain.with_fallbacks([triage_step(fallback, prompt_version, method)])
    reply_chain = reply_chain.with_fallbacks([reply_step(fallback)])

  def triage(state: SupportState, config: RunnableConfig) -> dict:
    result = triage_chain.invoke({"ticket": state["ticket"]}, config=config)
    return {"triage": result, "steps": [f"triage: {result.category}, {result.urgency}"]}

  def look_up_order(state: SupportState) -> dict:
    match = ORDER_ID.search(state["ticket"])
    order_id = match.group(0).upper() if match else None
    order = store.order(order_id) if order_id else None
    note = order_note(order_id, order)
    return {"order": order, "order_note": note, "steps": [f"look_up_order: {note or 'no order id in the ticket'}"]}

  def draft_reply(state: SupportState, config: RunnableConfig) -> dict:
    fixed = FIXED_REPLIES.get(state["triage"].category)
    if fixed:  # safety tickets: the approved message, never a draft
      return {"reply": fixed, "steps": ["draft_reply: fixed safety message"]}
    ticket = state["ticket"] + (f"\n\n[Facts for the reply: {state['order_note']}]" if state.get("order_note") else "")
    reply = reply_chain.invoke(
      {"ticket": ticket, "triage": state["triage"], "history": state.get("history", [])}, config=config
    )
    return {"reply": reply, "steps": ["draft_reply"]}

  graph = StateGraph(SupportState)
  graph.add_node("triage", triage)
  graph.add_node("look_up_order", look_up_order)
  graph.add_node("draft_reply", draft_reply)
  graph.add_edge(START, "triage")  # two edges from START: both nodes run in the first step
  graph.add_edge(START, "look_up_order")
  graph.add_edge(["triage", "look_up_order"], "draft_reply")  # waits for both
  graph.add_edge("draft_reply", END)
  return graph.compile(name="support_graph")


def support_graph(settings=None, version: str = "v3"):
  """The production support graph (v3 by default; v1 and v2 kept for comparison)."""
  from nimbus.config import get_settings
  from nimbus.llm import get_chat_model

  s = settings or get_settings()
  model, fallback = get_chat_model(s), get_chat_model(s, role="fallback")
  if version == "v3":
    from nimbus.rag.ingestion.embeddings import get_embeddings
    from nimbus.rag.retriever import Retriever

    retriever = Retriever(s, get_embeddings(s), model)
    return build_support_graph_v3(model, SupportStore(), retriever, fallback, s.triage_prompt_version, s.triage_method)
  build = build_support_graph_v2 if version == "v2" else build_support_graph
  return build(model, SupportStore(), fallback, s.triage_prompt_version, s.triage_method)


# ── Support graph v2 (Agents Module 2): routing, Send, a checked loop, input/output schemas ─────────────────────
#
#   START ─┬─ triage ─(route)─┬─ safety_reply ───────────────────────────→ END
#          │                  └─ draft_reply → check_reply ─(ok)─────────→ END
#          │                                     └─(problems, budget left)─┘ back to draft_reply
#          └─(Send per order id)→ look_up_order ×N   (results merged with operator.add)

STATUS_WORDS = {
  "delivered": ["delivered"],
  "shipped": ["shipped", "on its way", "dispatched", "in transit"],
  "processing": ["processing", "being prepared", "not yet shipped", "hasn't shipped", "being processed"],
  "packed": ["packed"],
  "cancelled": ["cancelled", "canceled"],
}
MAX_DRAFTS = 2
LINK = re.compile(r"https?://|www\.|\]\([^)]*\)", re.IGNORECASE)  # URLs and Markdown links, including "[shop](link)"
CLAIMED_ACTION = re.compile(
  r"\b(i'?ve|i have|we'?ve|we have|i|we)\s+(just\s+|already\s+|gone ahead and\s+)*"
  r"(cancel+ed|refunded|processed|issued|credited|reversed)\b",
  re.IGNORECASE,
)


class SupportInput(TypedDict):
  ticket: str
  history: list


class SupportOutput(TypedDict, total=False):
  triage: TicketTriage
  reply: str
  needs_review: bool  # the reply still failed its checks after the last draft: a person should look
  steps: Annotated[list[str], operator.add]


class SupportStateV2(SupportInput, SupportOutput, total=False):
  orders: Annotated[list[dict], operator.add]  # one entry per order id, from parallel Send tasks
  feedback: str
  drafts: int
  remaining_steps: RemainingSteps  # filled in by LangGraph: steps left before the recursion limit


class OrderTask(TypedDict):
  order_id: str


def reply_problems(reply: str, triage: TicketTriage, orders: list[dict]) -> list[str]:
  """What a reply must get right, checked in code (no model): the real order status, and a sane length."""
  problems = []
  text = reply.casefold()
  for o in (o for o in orders if o.get("record")):
    record = o["record"]
    if triage.language == "en":
      words = STATUS_WORDS.get(record["status"], [record["status"]])
      if not any(w in text for w in words):
        problems.append(f"say that order {record['order_id']} is {record['status']}")
    elif record["order_id"].casefold() not in text:
      problems.append(f"mention order {record['order_id']}")
  # This assistant can't cancel or refund anything: it must never say it did (models write curly apostrophes too)
  if CLAIMED_ACTION.search(reply.replace("\u2019", "'")):
    problems.append("don't say you cancelled, refunded or processed anything: say what happens next instead")
  if LINK.search(reply):  # the facts never contain URLs: any link in a reply is invented
    problems.append("don't add links or URLs")
  if len(reply.split()) > 160:
    problems.append("keep it under 160 words")
  return problems


def build_support_graph_v2(
  model: BaseChatModel,
  store: SupportStore,
  fallback: BaseChatModel | None = None,
  prompt_version: str = "v2",
  method: str = "json_schema",
):
  triage_chain = triage_step(model, prompt_version, method)
  reply_chain = reply_step(model)
  if fallback is not None:
    triage_chain = triage_chain.with_fallbacks([triage_step(fallback, prompt_version, method)])
    reply_chain = reply_chain.with_fallbacks([reply_step(fallback)])

  def triage(state: SupportStateV2, config: RunnableConfig) -> dict:
    result = triage_chain.invoke({"ticket": state["ticket"]}, config=config)
    return {"triage": result, "steps": [f"triage: {result.category}, {result.urgency}"]}

  def find_orders(state: SupportStateV2) -> list[Send]:
    ids = dict.fromkeys(m.upper() for m in ORDER_ID.findall(state["ticket"]))  # unique, in order
    return [Send("look_up_order", {"order_id": order_id}) for order_id in ids]  # one task per order, run in parallel

  def look_up_order(task: OrderTask) -> dict:  # receives ITS OWN input, not the whole state
    record = store.order(task["order_id"])
    note = order_note(task["order_id"], record)
    return {
      "orders": [{"order_id": task["order_id"], "record": record, "note": note}],
      "steps": [f"look_up_order: {note}"],
    }

  def route(state: SupportStateV2) -> str:
    return "safety_reply" if state["triage"].category in FIXED_REPLIES else "draft_reply"

  def safety_reply(state: SupportStateV2) -> dict:
    return {
      "reply": FIXED_REPLIES[state["triage"].category],
      "needs_review": True,
      "steps": ["safety_reply: fixed message"],
    }

  def draft_reply(state: SupportStateV2, config: RunnableConfig) -> dict:
    facts = " ".join(o["note"] for o in state.get("orders", []))
    extra = f"\n\n[Facts for the reply: {facts}]" if facts else ""
    if state.get("feedback"):
      extra += f"\n[The previous draft was sent back: {state['feedback']}.]"
    reply = reply_chain.invoke(
      {"ticket": state["ticket"] + extra, "triage": state["triage"], "history": state.get("history", [])}, config=config
    )
    drafts = state.get("drafts", 0) + 1
    return {"reply": reply, "drafts": drafts, "steps": [f"draft_reply: draft {drafts}"]}

  def check_reply(state: SupportStateV2) -> Command[Literal["draft_reply", "__end__"]]:
    problems = reply_problems(state["reply"], state["triage"], state.get("orders", []))
    budget_left = state["drafts"] < MAX_DRAFTS and state["remaining_steps"] > 3
    if problems and budget_left:  # update the state AND choose the next node, in one return
      return Command(
        update={"feedback": "; ".join(problems), "steps": [f"check_reply: {'; '.join(problems)}"]}, goto="draft_reply"
      )
    return Command(
      update={
        "needs_review": bool(problems),
        "steps": [f"check_reply: {'ok' if not problems else 'still: ' + '; '.join(problems)}"],
      },
      goto=END,
    )

  graph = StateGraph(SupportStateV2, input_schema=SupportInput, output_schema=SupportOutput)
  graph.add_node("triage", triage)
  graph.add_node("look_up_order", look_up_order)
  graph.add_node("safety_reply", safety_reply)
  graph.add_node("draft_reply", draft_reply)
  graph.add_node("check_reply", check_reply)
  graph.add_edge(START, "triage")
  graph.add_conditional_edges(START, find_orders, ["look_up_order"])  # Send: as many lookups as order ids
  graph.add_conditional_edges("triage", route, ["safety_reply", "draft_reply"])
  graph.add_edge("safety_reply", END)
  graph.add_edge("draft_reply", "check_reply")
  return graph.compile(name="support_graph_v2")


# ── Support graph v3 (Agents Module 3): routing to specialised handlers ────────────────────────────────────────
#
#   triage ─(route)─┬─ safety_reply ───────────────────────────────→ END
#                   ├─ orders_facts ──┐
#                   ├─ refund_facts ──┼→ draft_reply → check_reply ─→ END (or back to draft_reply)
#                   └─ knowledge_facts┘   (each handler gathers ONLY the facts its route needs)

ROUTES = {
  "order_problem": "orders_facts",
  "cancel_order": "orders_facts",
  "refund_return": "refund_facts",
  "product_question": "knowledge_facts",
  "cafe_experience": "knowledge_facts",
  "subscription": "knowledge_facts",
  "other": "knowledge_facts",
}


class SupportStateV3(SupportStateV2, total=False):
  facts: list[str]  # what the reply may rely on, gathered by the route's handler
  sources: list[str]  # where the facts came from (files, order records)


def refund_decision(store: SupportStore, order: dict | None, order_id: str) -> str:
  """The refund policy applied in code: the model is told the decision, it never makes it."""
  if order is None:
    return f"No order {order_id} exists."
  if order["status"] != "delivered" or not order.get("delivered_on"):
    return (
      f"{order_id} is {order['status']}: refunds apply to delivered orders (it can be cancelled instead if not packed)."
    )
  days = (store.today - date.fromisoformat(order["delivered_on"])).days
  if order.get("opened"):
    return f"{order_id} was opened: opened bags can't be returned; a teammate can review faulty beans."
  if days > REFUND_WINDOW_DAYS:
    return f"{order_id} was delivered {days} days ago, outside the {REFUND_WINDOW_DAYS}-day window: not refundable."
  return (
    f"{order_id} is eligible: unopened, delivered {days} days ago. A teammate confirms the ₹{order['total']} refund."
  )


def build_support_graph_v3(
  model: BaseChatModel,
  store: SupportStore,
  retriever,
  fallback: BaseChatModel | None = None,
  prompt_version: str = "v2",
  method: str = "json_schema",
):
  """v2's routing, Send lookups and checked loop, with a handler per kind of ticket that gathers only its facts:
  order records, the refund decision (computed in code), or public documents from the RAG retriever."""
  triage_chain = triage_step(model, prompt_version, method)
  reply_chain = reply_step(model)
  if fallback is not None:
    triage_chain = triage_chain.with_fallbacks([triage_step(fallback, prompt_version, method)])
    reply_chain = reply_chain.with_fallbacks([reply_step(fallback)])

  def triage(state: SupportStateV3, config: RunnableConfig) -> dict:
    result = triage_chain.invoke({"ticket": state["ticket"]}, config=config)
    return {"triage": result, "steps": [f"triage: {result.category}, {result.urgency}"]}

  def find_orders(state: SupportStateV3) -> list[Send]:
    ids = dict.fromkeys(m.upper() for m in ORDER_ID.findall(state["ticket"]))
    return [Send("look_up_order", {"order_id": order_id}) for order_id in ids]

  def look_up_order(task: OrderTask) -> dict:
    record = store.order(task["order_id"])
    note = order_note(task["order_id"], record)
    return {
      "orders": [{"order_id": task["order_id"], "record": record, "note": note}],
      "steps": [f"look_up_order: {note}"],
    }

  def route(state: SupportStateV3) -> str:
    category = state["triage"].category
    return "safety_reply" if category in FIXED_REPLIES else ROUTES.get(category, "knowledge_facts")

  def safety_reply(state: SupportStateV3) -> dict:
    return {
      "reply": FIXED_REPLIES[state["triage"].category],
      "needs_review": True,
      "steps": ["safety_reply: fixed message"],
    }

  def orders_facts(state: SupportStateV3) -> dict:
    notes = [o["note"] for o in state.get("orders", [])] or ["The customer didn't give an order number: ask for it."]
    return {"facts": notes, "sources": [o["order_id"] for o in state.get("orders", [])], "steps": ["orders_facts"]}

  def refund_facts(state: SupportStateV3) -> dict:
    decisions = [refund_decision(store, o["record"], o["order_id"]) for o in state.get("orders", [])]
    decisions = decisions or ["No order number given: ask for it; unopened bags can be returned within 14 days."]
    return {
      "facts": decisions,
      "sources": [o["order_id"] for o in state.get("orders", [])],
      "steps": [f"refund_facts: {'; '.join(decisions)}"],
    }

  def knowledge_facts(state: SupportStateV3, config: RunnableConfig) -> dict:
    _, hits = retriever.search(state["ticket"], k=3, audience="public")  # customers only see public documents
    facts = [d.page_content for d, _ in hits]
    sources = [d.metadata["source"] for d, _ in hits]
    return {"facts": facts, "sources": sources, "steps": [f"knowledge_facts: {', '.join(sources)}"]}

  def draft_reply(state: SupportStateV3, config: RunnableConfig) -> dict:
    facts = "\n".join(f"- {f}" for f in state.get("facts", []))
    # measured in Agents 03: without the second sentence, replies promised actions and times nobody had confirmed
    extra = (
      f"\n\n[Facts for the reply; use only these, don't add others:\n{facts}\nOffer only the next step you were "
      "given; never say something is done or will happen at a certain time.]"
      if facts
      else ""
    )
    if state.get("feedback"):
      extra += f"\n[The previous draft was sent back: {state['feedback']}.]"
    reply = reply_chain.invoke(
      {"ticket": state["ticket"] + extra, "triage": state["triage"], "history": state.get("history", [])}, config=config
    )
    drafts = state.get("drafts", 0) + 1
    return {"reply": reply, "drafts": drafts, "steps": [f"draft_reply: draft {drafts}"]}

  def check_reply(state: SupportStateV3) -> Command[Literal["draft_reply", "__end__"]]:
    problems = reply_problems(state["reply"], state["triage"], state.get("orders", []))
    budget_left = state["drafts"] < MAX_DRAFTS and state["remaining_steps"] > 3
    if problems and budget_left:
      return Command(
        update={"feedback": "; ".join(problems), "steps": [f"check_reply: {'; '.join(problems)}"]}, goto="draft_reply"
      )
    verdict = "ok" if not problems else "still: " + "; ".join(problems)
    return Command(update={"needs_review": bool(problems), "steps": [f"check_reply: {verdict}"]}, goto=END)

  graph = StateGraph(SupportStateV3, input_schema=SupportInput, output_schema=SupportOutput)
  for name, fn in (
    ("triage", triage),
    ("look_up_order", look_up_order),
    ("safety_reply", safety_reply),
    ("orders_facts", orders_facts),
    ("refund_facts", refund_facts),
    ("knowledge_facts", knowledge_facts),
    ("draft_reply", draft_reply),
    ("check_reply", check_reply),
  ):
    graph.add_node(name, fn)
  graph.add_edge(START, "triage")
  graph.add_conditional_edges(START, find_orders, ["look_up_order"])
  graph.add_conditional_edges("triage", route, ["safety_reply", "orders_facts", "refund_facts", "knowledge_facts"])
  graph.add_edge("safety_reply", END)
  for handler in ("orders_facts", "refund_facts", "knowledge_facts"):
    graph.add_edge(handler, "draft_reply")
  graph.add_edge("draft_reply", "check_reply")
  return graph.compile(name="support_graph_v3")
