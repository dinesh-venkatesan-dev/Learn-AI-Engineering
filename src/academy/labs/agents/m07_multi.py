"""Agents with LangGraph · Module 7 labs: subgraphs and Send, subgraphs in depth, supervisors, routers, handoffs,
skills, hierarchical teams and the Functional API. Then: three ways to build the Nimbus assistant, measured."""

import asyncio
import operator
import random
import time
from functools import lru_cache
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model


def _thread(name: str) -> dict:
  return {"configurable": {"thread_id": name}}


@lru_cache
def _parts():
  """The retriever and the three specialists, built once per server start."""
  from nimbus.assistant.specialists import build_specialists
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.rag.retriever import Retriever

  settings, model = get_settings(), get_chat_model()
  retriever = Retriever(settings, get_embeddings(settings), model)
  return retriever, build_specialists(settings, model, retriever)


# ── Lab 1: subgraphs and Send ──────────────────────────────────────────────────
PRICES = {"Oat Latte": 220, "Cold Brew": 200, "Vegan Brownie": 170, "Cappuccino": 180}


class Basket(TypedDict, total=False):
  items: list[str]
  lines: Annotated[list[str], operator.add]  # the reducer collects every parallel result
  total: int


def map_reduce_with_send():
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Send

  def fan_out(state: Basket):
    return [Send("price_line", {"item": item}) for item in state["items"]]  # one task per item, in parallel

  def price_line(task: dict) -> dict:
    return {"lines": [f"{task['item']}: ₹{PRICES[task['item']]}"]}

  def add_up(state: Basket) -> dict:
    return {"total": sum(int(line.split("₹")[1]) for line in state["lines"])}

  graph = StateGraph(Basket).add_node("price_line", price_line).add_node("add_up", add_up)
  graph.add_conditional_edges(START, fan_out, ["price_line"])
  graph.add_edge("price_line", "add_up").add_edge("add_up", END)
  return graph.compile().invoke({"items": ["Oat Latte", "Cold Brew", "Vegan Brownie"], "lines": []})


class OrderFlow(TypedDict, total=False):
  items: list[str]
  amount: int
  log: Annotated[list[str], operator.add]  # shared with the payment subgraph
  receipt: str


class PaymentState(TypedDict, total=False):
  amount: int
  log: Annotated[list[str], operator.add]
  card_ok: bool  # private to the subgraph


def payment_subgraph(approve: bool = False):
  from langgraph.graph import START, StateGraph
  from langgraph.types import interrupt

  def check_card(state: PaymentState) -> dict:
    return {"card_ok": True, "log": ["card checked"]}

  def charge(state: PaymentState) -> dict:
    if approve and interrupt(f"Charge ₹{state['amount']}?") != "yes":
      return {"log": ["declined"]}
    return {"log": [f"charged ₹{state['amount']}"]}

  graph = StateGraph(PaymentState).add_node("check_card", check_card).add_node("charge", charge)
  return graph.add_edge(START, "check_card").add_edge("check_card", "charge").compile()


def shared_keys(payment, checkpointer=None):
  from langgraph.graph import END, START, StateGraph

  def price(state: OrderFlow) -> dict:
    return {"amount": sum(PRICES[i] for i in state["items"]), "log": ["priced"]}

  def receipt(state: OrderFlow) -> dict:
    return {"receipt": f"{len(state['items'])} items, ₹{state['amount']}"}

  graph = StateGraph(OrderFlow).add_node("price", price).add_node("payment", payment).add_node("receipt", receipt)
  graph.add_edge(START, "price").add_edge("price", "payment").add_edge("payment", "receipt").add_edge("receipt", END)
  return graph.compile(checkpointer=checkpointer)  # the subgraph is a node; overlapping keys (amount, log) flow


class TicketFlow(TypedDict, total=False):
  ticket: str
  answer: str


def different_schemas(specialists):
  from langgraph.graph import START, StateGraph

  def ask_account(state: TicketFlow) -> dict:
    # a different schema: translate in, call the desk graph, translate out
    payload = {"messages": [HumanMessage(state["ticket"])], "customer_id": "c-7007", "actions": []}
    from nimbus.support.agent import CustomerContext

    result = specialists.desk.invoke(payload, context=CustomerContext("c-7007", "Vikram P."))
    return {"answer": result["ticket"].reply}

  graph = StateGraph(TicketFlow).add_node("ask_account", ask_account).add_edge(START, "ask_account").compile()
  return graph.invoke({"ticket": "When will my order ship?"})


def look_inside_a_paused_subgraph():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.types import Command

  graph = shared_keys(payment_subgraph(approve=True), checkpointer=InMemorySaver())  # one checkpointer for both
  config = _thread("order-9")
  graph.invoke({"items": ["Oat Latte", "Cold Brew"], "log": []}, config)
  parent = graph.get_state(config, subgraphs=True)
  inner = parent.tasks[0].state  # the subgraph's own snapshot
  resumed = graph.invoke(Command(resume="yes"), config)
  return parent.next, inner.next, inner.values, resumed


@lab(
  "agents.subgraphs",
  "Subgraphs and Send: composing graphs and fanning out work",
  uses_model=True,
  calls="~2 calls to Mistral",
)
def subgraphs():
  """A basket priced line by line with Send, a payment graph used as a node of an order graph (shared keys), the
  support desk called from a node of another graph (different schemas), and a subgraph paused for approval."""
  _, specialists = _parts()
  basket = map_reduce_with_send()
  order = shared_keys(payment_subgraph()).invoke({"items": ["Oat Latte", "Cold Brew"], "log": []})
  desk = different_schemas(specialists)
  parent_next, inner_next, inner_values, resumed = look_inside_a_paused_subgraph()
  return [
    table(
      ["Send", "Result"],
      [["lines (one task each)", ", ".join(basket["lines"])], ["total (the reduce)", basket["total"]]],
    ),
    table(
      ["Order graph", "Result"],
      [["log", str(order["log"])], ["receipt", order["receipt"]], ["card_ok in the parent?", "card_ok" in order]],
    ),
    table(["Different schemas", "Result"], [["the desk's answer, translated back", desk["answer"]]]),
    table(
      ["Paused inside the subgraph", "Result"],
      [
        ["parent next", str(parent_next)],
        ["subgraph next", str(inner_next)],
        ["subgraph state", str(inner_values)],
        ["after resume", str(resumed["log"])],
      ],
    ),
    callout(
      "Send runs one node many times in parallel and a reducer collects the results. A compiled graph is a node like "
      "any other: with shared keys it reads and writes them directly (private keys stay inside); with a different "
      "schema, call it from a node and translate both ways (the Nimbus assistant calls the desk like that). One "
      "checkpointer on the parent saves the subgraph's progress too, and get_state(subgraphs=True) shows where it "
      "paused."
      + (
        f" Watch shared keys with a reducer: the log reads {order['log']}, because the subgraph returns its whole "
        "state (the inherited entries included) and the parent's operator.add appends it again. Give subgraphs "
        "their own key for what they add, or no reducer on keys they return whole."
        if order["log"].count("priced") > 1
        else ""
      ),
      "decision",
      "Composing graphs",
    ),
  ] + [
    example("lc.l52.1", "Send: map", map_reduce_with_send, "", [table(["line"], [[x] for x in basket["lines"]])]),
    example("lc.l52.2", "A reducer: reduce", map_reduce_with_send, "", [table(["total"], [[basket["total"]]])]),
    example("lc.l52.3", "A compiled graph as a node", shared_keys, "", [table(["log"], [[x] for x in order["log"]])]),
    example(
      "lg.l38.1",
      "Shared keys",
      shared_keys,
      "",
      [table(["receipt", "card_ok in parent"], [[order["receipt"], "card_ok" in order]])],
    ),
    example(
      "lg.l38.2",
      "Different schemas: call it in a node",
      different_schemas,
      "",
      [answer(desk["answer"], "from the desk")],
    ),
    example(
      "lg.l38.3",
      "One checkpointer for all",
      look_inside_a_paused_subgraph,
      "",
      [table(["after resume"], [[str(resumed["log"])]])],
    ),
    example(
      "lg.l38.4",
      "get_state(subgraphs=True)",
      look_inside_a_paused_subgraph,
      "",
      [
        table(
          ["parent next", "subgraph next", "subgraph state"], [[str(parent_next), str(inner_next), str(inner_values)]]
        )
      ],
    ),
  ]


# ── Lab 2: subgraphs in depth ──────────────────────────────────────────────────
class Notes(TypedDict):
  topic: str
  notes: Annotated[list[str], operator.add]


class Chat(TypedDict):
  topic: str
  reply: str


def persistence_modes():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph

  out = {}
  for mode in (None, True, False):  # set on the SUBGRAPH's compile()
    notes = StateGraph(Notes).add_node("note", lambda s: {"notes": [s["topic"]]}).add_edge(START, "note")
    notes = notes.compile(checkpointer=mode)

    def ask(state: Chat, notes=notes) -> dict:
      return {"reply": f"notes so far: {notes.invoke({'topic': state['topic']})['notes']}"}

    chat = StateGraph(Chat).add_node("ask", ask).add_edge(START, "ask").compile(checkpointer=InMemorySaver())
    for topic in ("latte", "mocha"):
      result = chat.invoke({"topic": topic}, _thread("same"))
    out[{None: "None (per invocation)", True: "True (per thread)", False: "False (stateless)"}[mode]] = result["reply"]
  return out


class Stock(TypedDict, total=False):
  stock: str
  answer: str
  charged: str


def interrupt_in_subgraph():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph
  from langgraph.types import Command, interrupt

  out = {}
  for mode in (None, False):
    counts = {"lookup": 0}

    def lookup(state: Stock, counts=counts) -> dict:
      counts["lookup"] += 1  # slow and costly
      return {"stock": "12 bags left"}

    def confirm(state: Stock) -> dict:
      return {"answer": interrupt(f"{state['stock']}: reorder now?")}

    sub = StateGraph(Stock).add_node("lookup", lookup).add_node("confirm", confirm)
    sub = sub.add_edge(START, "lookup").add_edge("lookup", "confirm").compile(checkpointer=mode)
    parent = StateGraph(Stock).add_node("order", sub).add_edge(START, "order").compile(checkpointer=InMemorySaver())
    parent.invoke({}, _thread(str(mode)))
    parent.invoke(Command(resume="yes"), _thread(str(mode)))
    out["per invocation (None)" if mode is None else "stateless (False)"] = counts["lookup"]
  return out


def crash_and_retry():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph
  from langgraph.types import RetryPolicy

  counts = {"lookup": 0, "charge": 0}

  def lookup(state: Stock) -> dict:
    counts["lookup"] += 1
    return {"stock": "12 bags left"}

  def charge(state: Stock) -> dict:
    counts["charge"] += 1
    if counts["charge"] == 1:
      raise ConnectionError("payment API down")  # fails the first time only
    return {"charged": "yes"}

  def sub(retry=None):
    graph = StateGraph(Stock).add_node("lookup", lookup).add_node("charge", charge, retry_policy=retry)
    return graph.add_edge(START, "lookup").add_edge("lookup", "charge").compile()

  def parent(subgraph, **options):
    graph = StateGraph(Stock).add_node("order", subgraph, **options).add_edge(START, "order")
    return graph.compile(checkpointer=InMemorySaver())

  out = {}
  graph = parent(sub())
  try:
    graph.invoke({}, _thread("crash"))
  except ConnectionError:
    out["after the crash: parent state"] = str(graph.get_state(_thread("crash")).values)
  graph.invoke(None, _thread("crash"))  # resume continues inside the subgraph
  out["resume: lookup, charge runs"] = f"{counts['lookup']}, {counts['charge']}"
  retry = RetryPolicy(max_attempts=3, initial_interval=0.05)
  for label, graph in (
    ("retry on the subgraph's node", parent(sub(), retry_policy=retry)),
    ("retry on the inner node", parent(sub(retry))),
  ):
    counts.update(lookup=0, charge=0)
    graph.invoke({}, _thread(label))
    out[f"{label}: lookup, charge runs"] = f"{counts['lookup']}, {counts['charge']}"
  return out


@lab(
  "agents.subgraphs-depth", "Subgraphs in depth: memory modes, interrupts, failures and retries", calls="No model calls"
)
def subgraphs_depth():
  """A note-taking subgraph in each persistence mode, an interrupt inside a subgraph, a crash inside one, and where a
  retry policy belongs."""
  modes = persistence_modes()
  lookups = interrupt_in_subgraph()
  crash = crash_and_retry()
  return [
    table(["Subgraph checkpointer", "after 'latte' then 'mocha' in one thread"], list(modes.items())),
    table(["Interrupt inside a subgraph", "times the lookup ran"], list(lookups.items())),
    table(["Failure", "Result"], list(crash.items())),
    callout(
      "Set the subgraph's own compile(checkpointer=...): None (default) starts fresh each call but is saved while it "
      "runs, so an interrupt resumes inside it (the lookup ran once); True remembers across calls in the same thread; "
      "False is a plain function (the lookup ran again). A crash inside leaves the parent untouched and a resume "
      "continues inside. A retry on the subgraph's node re-runs the whole subgraph: put it on the step that fails.",
      "decision",
      "Subgraphs in depth",
    ),
  ] + [
    example(
      "lg.l39.1", "Three persistence modes", persistence_modes, "", [table(["mode", "reply"], list(modes.items()))]
    ),
    example(
      "lg.l39.2",
      "Interrupts inside a subgraph",
      interrupt_in_subgraph,
      "",
      [table(["mode", "lookups"], list(lookups.items()))],
    ),
    example("lg.l39.3", "A crash inside a subgraph", crash_and_retry, "", [table(["", ""], list(crash.items())[:2])]),
    example("lg.l39.4", "Where a RetryPolicy goes", crash_and_retry, "", [table(["", ""], list(crash.items())[2:])]),
  ]


# ── Lab 3: supervisors ─────────────────────────────────────────────────────────
QUESTION_TWO_PARTS = ("c-6060", "How long do I have to return unopened beans, and can I return NC-90112?")


def subagents_as_tools(specialists):
  from langchain.agents import create_agent

  @tool
  def ask_knowledge(question: str) -> str:
    """Ask the knowledge specialist about Nimbus policies, cafés and products."""
    return specialists.ask("knowledge", "c-6060", question)

  @tool
  def ask_account(question: str) -> str:
    """Ask the account specialist about this customer's orders, subscription, points and refunds."""
    return specialists.ask("account", "c-6060", question)

  supervisor = create_agent(
    get_chat_model(),
    [ask_knowledge, ask_account],  # each subagent wrapped as a tool
    system_prompt="You answer Nimbus customers by asking your specialists. Combine their answers briefly.",
  )
  return supervisor.invoke({"messages": [HumanMessage(QUESTION_TWO_PARTS[1])]})


def langgraph_supervisor(specialists):
  from nimbus.assistant.architectures import build_supervisor

  graph, ask = build_supervisor(get_chat_model(), specialists)
  return graph, graph.invoke({"question": QUESTION_TWO_PARTS[1], "customer_id": "c-6060", "notes": [], "asked": []})


@lab("agents.supervisor", "A supervisor that delegates to specialists", uses_model=True, calls="~15 calls to Mistral")
def supervisor():
  """A two-part question (a policy and an order) through a create_agent supervisor whose tools are subagents, and
  through the assistant's LangGraph supervisor loop."""
  _, specialists = _parts()
  tool_style = subagents_as_tools(specialists)
  graph, loop = langgraph_supervisor(specialists)
  calls = [c["name"] for m in tool_style["messages"] for c in getattr(m, "tool_calls", [])]
  return [
    table(
      ["Supervisor", "asked", "reply"],
      [
        ["create_agent + subagents as tools", ", ".join(calls), tool_style["messages"][-1].text],
        ["LangGraph loop", " → ".join(loop["asked"]), loop["reply"]],
      ],
    ),
    table(["Shared notes (one list, labelled by specialist)"], [[n] for n in loop["notes"]]),
    code(graph.get_graph().draw_mermaid(), "text", title="supervisor → specialist → supervisor … → FINISH"),
    callout(
      "Specialists are small agents with their own prompts and tools; wrapped as tools, another agent can call them. "
      "A LangGraph supervisor makes the choice explicit: a structured decision (next specialist and task, or FINISH "
      "with the reply), a loop back after each answer, and shared notes labelled with who said what. Code caps the "
      f"rounds and refuses repeats. Here the loop asked {len(loop['asked'])} time(s); the evaluation measures what "
      "that costs.",
      "decision",
      "Supervisors",
    ),
  ] + [
    example(
      "lc.l36.1",
      "Specialist subagents",
      subagents_as_tools,
      "The specialists are in nimbus/assistant/specialists.py.",
      [table(["tools"], [[c] for c in calls])],
    ),
    example(
      "lc.l36.2",
      "A subagent wrapped as a tool",
      subagents_as_tools,
      "",
      [answer(tool_style["messages"][-1].text, "reply")],
    ),
    example(
      "lc.l36.3",
      "The supervisor decides and combines",
      subagents_as_tools,
      "",
      [table(["asked"], [[", ".join(calls)]])],
    ),
    example(
      "lg.l40.1", "Specialists", langgraph_supervisor, "", [table(["specialist asked"], [[a] for a in loop["asked"]])]
    ),
    example(
      "lg.l40.2",
      "A structured decision",
      langgraph_supervisor,
      "Decision(next, task, reply) in nimbus/assistant/architectures.py.",
      [table(["asked"], [[a] for a in loop["asked"]])],
    ),
    example(
      "lg.l40.3", "The loop to FINISH", langgraph_supervisor, "", [code(graph.get_graph().draw_mermaid(), "text")]
    ),
    example("lg.l40.4", "Shared notes", langgraph_supervisor, "", [table(["notes"], [[n] for n in loop["notes"]])]),
  ]


# ── Lab 4: routers ─────────────────────────────────────────────────────────────
def router_plan(specialists, question):
  from nimbus.assistant.architectures import build_router

  graph, _ = build_router(get_chat_model(), specialists)
  return graph.invoke({"question": question, "customer_id": "c-5005", "answers": []})


def router_as_a_tool(specialists):
  from langchain.agents import create_agent
  from langgraph.checkpoint.memory import InMemorySaver

  from nimbus.assistant.architectures import build_router

  _, ask = build_router(get_chat_model(), specialists)

  @tool
  def ask_nimbus(question: str) -> str:
    """Answer any question about Nimbus Coffee or this customer's account (a stateless router of specialists)."""
    return ask("c-5005", question).reply

  chat = create_agent(
    get_chat_model(),
    [ask_nimbus],
    system_prompt="Use ask_nimbus with self-contained questions. Be brief.",
    checkpointer=InMemorySaver(),
  )
  replies = []
  for text in ("Where is my order NC-78123?", "And how many points will it earn?"):  # the second needs the first
    result = chat.invoke({"messages": [HumanMessage(text)]}, _thread("chat"))
    calls = (
      [c["args"].get("question") for c in getattr(result["messages"][-3], "tool_calls", [])]
      if len(result["messages"]) > 2
      else []
    )
    replies.append((text, calls, result["messages"][-1].text))
  return replies


@lab(
  "agents.router",
  "A router: classify the question, send it to the right specialists",
  uses_model=True,
  calls="~15 calls to Mistral",
)
def router():
  """The assistant's router on a two-part question (two specialists in parallel), on small talk (none), and wrapped
  as a tool of a chat agent that remembers the conversation."""
  _, specialists = _parts()
  two = router_plan(specialists, "Where is my order NC-78123, and is the cold brew vegan?")
  none = router_plan(specialists, "Hi! Thanks for the help last week.")
  chat = router_as_a_tool(specialists)
  return [
    table(
      ["Message", "plan (one call)", "answers", "reply"],
      [
        [q, str(r["plan"]), str([a["specialist"] for a in r.get("answers", [])]), r["reply"]]
        for q, r in (
          ("Where is my order NC-78123, and is the cold brew vegan?", two),
          ("Hi! Thanks for the help last week.", none),
        )
      ],
    ),
    table(
      ["Turn", "asked the router", "reply"],
      [[t, str(c), r] for t, c, r in chat],
      title="The stateless router as a tool of a chat agent with memory",
    ),
    callout(
      "A router makes ONE structured call that decides which specialists are needed and what to ask each; they run "
      "in parallel (Send) and a final call combines their answers, skipped when only one answered. When nothing "
      "fits (small talk) it answers at once and no specialist runs. The router has no memory: put it behind a chat "
      "agent with a checkpointer, which turns follow-ups into self-contained questions.",
      "decision",
      "Routers",
    ),
  ] + [
    example(
      "lc.l37.1",
      "Specialists per area",
      router_plan,
      "",
      [table(["specialist", "answer"], [[a["specialist"], a["answer"]] for a in two["answers"]])],
    ),
    example("lc.l37.2", "One call plans", router_plan, "", [code(str(two["plan"]), "python")]),
    example("lc.l37.3", "Parallel specialists, one combine", router_plan, "", [answer(two["reply"], "combined")]),
    example(
      "lc.l37.4",
      "Nothing fits: answer at once",
      router_plan,
      "",
      [table(["plan", "reply"], [[str(none["plan"]), none["reply"]]])],
    ),
    example(
      "lc.l37.5",
      "The router as a tool, with memory around it",
      router_as_a_tool,
      "",
      [table(["turn", "asked", "reply"], [[t, str(c), r] for t, c, r in chat])],
    ),
  ]


# ── Lab 5: handoffs ────────────────────────────────────────────────────────────
def step_handoffs():
  from typing import NotRequired

  from langchain.agents import AgentState, create_agent
  from langchain.agents.middleware import wrap_model_call
  from langchain.tools import ToolRuntime
  from langchain_core.messages import SystemMessage
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.types import Command

  from nimbus.support.tools import SupportStore

  store = SupportStore()

  class StepState(AgentState):
    current_step: NotRequired[str]  # "identify" → "diagnose" → "resolve"
    order_id: NotRequired[str]
    issue: NotRequired[str]

  def move_to(step: str, note: str, runtime: ToolRuntime, **fields) -> Command:
    return Command(
      update={"current_step": step, **fields, "messages": [ToolMessage(note, tool_call_id=runtime.tool_call_id)]}
    )

  @tool
  def record_order(order_id: str, runtime: ToolRuntime) -> Command | str:
    """Record which order the customer is talking about."""
    order = store.order(order_id)
    if order is None:
      return f"No order {order_id}. Ask the customer to check it."  # stay in this step
    return move_to("diagnose", f"{order_id}: {order['status']}.", runtime, order_id=order_id.upper())

  @tool
  def record_issue(issue: Literal["damaged", "wrong_item", "late"], runtime: ToolRuntime) -> Command:
    """Record what is wrong with the order, in the customer's words mapped to one kind."""
    return move_to("resolve", f"Issue recorded: {issue}.", runtime, issue=issue)

  @tool
  def send_replacement(runtime: ToolRuntime) -> str:
    """Send a free replacement for the order."""
    return f"Replacement for {runtime.state['order_id']} dispatched, arriving in 2 days."

  steps = {
    "identify": (
      "Find out which order this is about; call record_order with its id. You can't fix anything yet.",
      [record_order],
    ),
    "diagnose": (
      "The order is {order_id}. Ask what's wrong; call record_issue with what the customer SAID.",
      [record_issue],
    ),
    "resolve": (
      "Order {order_id}, issue {issue}. Offer a replacement; send it only after the customer agrees.",
      [send_replacement],
    ),
  }

  @wrap_model_call
  def apply_step(request, handler):
    current = request.state.get("current_step", "identify")
    prompt, tools = steps[current]  # the active step decides the prompt AND the tools
    system = SystemMessage("You are Nimbus Coffee support. Be brief. " + prompt.format(**request.state))
    return handler(request.override(system_message=system, tools=tools, model_settings={"parallel_tool_calls": False}))

  agent = create_agent(
    get_chat_model(),
    [record_order, record_issue, send_replacement],
    middleware=[apply_step],
    state_schema=StepState,
    checkpointer=InMemorySaver(),
  )
  turns = [
    "Hi, there's a problem with my coffee order.",
    "It's NC-90112.",
    "The bag arrived torn open.",
    "A replacement please.",
  ]
  rows = []
  for text in turns:
    result = agent.invoke({"messages": [HumanMessage(text)]}, _thread("steps"))
    saved = agent.get_state(_thread("steps")).values
    rows.append([text, saved.get("current_step", "identify"), result["messages"][-1].text])
  used = [c["name"] for m in result["messages"] for c in getattr(m, "tool_calls", [])]
  return rows, used


def swarm():
  from langchain.agents import create_agent
  from langchain.tools import ToolRuntime
  from langgraph.graph import START, MessagesState, StateGraph
  from langgraph.types import Command

  from nimbus.support.tools import SupportStore, build_tools

  store = SupportStore()
  tools = {t.name: t for t in build_tools(store)}

  def handoff_to(agent: str, description: str):
    @tool(f"transfer_to_{agent}", description=description)
    def transfer(runtime: ToolRuntime) -> Command:
      note = ToolMessage(f"Transferred to {agent}.", tool_call_id=runtime.tool_call_id)  # answers its own call
      return Command(goto=agent, graph=Command.PARENT, update={"messages": [*runtime.state["messages"], note]})

    return transfer

  barista = create_agent(
    get_chat_model(),
    [tools["cafe_menu_item"], handoff_to("support", "Transfer to support for orders and refunds.")],
    system_prompt=(
      "You are Nimbus's barista: café menu questions. For orders or refunds, transfer to support. Be brief."
    ),
    name="barista",
  )
  support = create_agent(
    get_chat_model(),
    [tools["get_order"], handoff_to("barista", "Transfer to the barista for café menu questions.")],
    system_prompt=(
      "You are Nimbus's support agent for orders. For café menu questions, transfer to the barista. Be brief."
    ),
    name="support",
  )
  graph = (
    StateGraph(MessagesState)
    .add_node("barista", barista, destinations=("support",))
    .add_node("support", support, destinations=("barista",))
  )
  graph = graph.add_edge(START, "barista").compile()  # conversations start with the barista; no supervisor
  result = graph.invoke({"messages": [HumanMessage("Hi! Where is my order NC-78123? It should have arrived by now.")]})
  return graph, [
    [m.type, getattr(m, "name", "") or "", m.text[:150] or str([c["name"] for c in getattr(m, "tool_calls", [])])]
    for m in result["messages"]
  ]


@lab(
  "agents.handoffs",
  "Handoffs: one conversation moving through steps, and agents passing control",
  uses_model=True,
  calls="~12 calls to Mistral",
)
def handoffs():
  """A damaged-order conversation in three steps (identify, diagnose, resolve), each with its own prompt and tools,
  and a barista agent that hands an order question to support with no supervisor in between."""
  rows, used = step_handoffs()
  graph, trace = swarm()
  return [
    table(
      ["Customer", "step after the turn", "reply"], rows, title="Steps: the active one decides the prompt and the tools"
    ),
    table(["Message", "agent", "content"], trace, title="A swarm: barista → support"),
    code(graph.get_graph().draw_mermaid(), "text"),
    callout(
      "Handoffs keep one conversation and move it between steps or agents. With create_agent, a state field says "
      "which step is active and middleware swaps that step's prompt and tools in on every call: send_replacement "
      f"doesn't exist until the order and the issue are known (tools used: {', '.join(used)}). In a graph, a handoff "
      "tool returns Command(goto=..., graph=Command.PARENT) and answers its own tool call. No supervisor means fewer "
      "calls, but nobody oversees the whole conversation.",
      "decision",
      "Handoffs",
    ),
  ] + [
    example(
      "lc.l38.1",
      "A state field says which step is active",
      step_handoffs,
      "",
      [table(["turn", "step"], [[r[0], r[1]] for r in rows])],
    ),
    example(
      "lc.l38.2",
      "Middleware swaps prompt and tools",
      step_handoffs,
      "",
      [table(["turn", "reply"], [[r[0], r[2]] for r in rows])],
    ),
    example(
      "lc.l38.3", "Handoff tools return a Command", step_handoffs, "", [table(["tools used"], [[", ".join(used)]])]
    ),
    example(
      "lc.l38.4", "Steps unlock tools", step_handoffs, "", [table(["tools used, in order"], [[", ".join(used)]])]
    ),
    example(
      "lc.l38.5",
      "The customer talks to each step across turns",
      step_handoffs,
      "",
      [table(["turn", "step"], [[r[0], r[1]] for r in rows])],
    ),
    example(
      "lg.l41.1", "Command(goto, graph=Command.PARENT)", swarm, "", [table(["message", "agent", "content"], trace)]
    ),
    example("lg.l41.2", "No supervisor: a swarm", swarm, "", [code(graph.get_graph().draw_mermaid(), "text")]),
    example(
      "lg.l41.3",
      "The handoff answers its own tool call",
      swarm,
      "",
      [table(["message", "content"], [[r[0], r[2]] for r in trace if r[0] == "tool"])],
    ),
    example(
      "lg.l41.4",
      "Supervisor vs handoffs",
      swarm,
      "Lab 3 is the supervisor; the evaluation measures it.",
      [
        table(
          ["", "supervisor", "handoffs"],
          [["control", "central", "peer to peer"], ["extra calls", "one decision per round", "none"]],
        )
      ],
    ),
  ]


# ── Lab 6: skills ──────────────────────────────────────────────────────────────
def skill_folder():
  from nimbus.assistant.skills import load_skills

  return {name: (s.description, sorted(p.name for p in s.folder.iterdir())) for name, s in load_skills().items()}


def progressive_disclosure():
  from nimbus.assistant.skills import load_skills, skills_prompt

  skills = load_skills()
  return skills_prompt(skills), sum(len(s.instructions) for s in skills.values())  # descriptions only, until needed


def run_cafe(specialists, question):
  result = specialists.cafe.invoke({"messages": [HumanMessage(question)], "skills_loaded": []})
  used = [(c["name"], c["args"]) for m in result["messages"] for c in getattr(m, "tool_calls", [])]
  return used, result.get("skills_loaded", []), result["messages"][-1].text


@lab("agents.skills", "Skills: instructions loaded only when needed", uses_model=True, calls="~10 calls to Mistral")
def skills():
  """The café specialist with three skills: an allergen question (the skill points to a table file), a catering
  quote (the skill unlocks a tool), and a question no skill covers."""
  _, specialists = _parts()
  folders = skill_folder()
  prompt, full = progressive_disclosure()
  runs = {
    q: run_cafe(specialists, q)
    for q in (
      "Does the banana bread contain nuts?",
      "Can you quote coffee catering for 30 people next Friday?",
      "How much is a cappuccino?",
    )
  }
  return [
    table(
      ["Skill", "description (always in the prompt)", "files"], [[n, d, ", ".join(f)] for n, (d, f) in folders.items()]
    ),
    table(
      ["Question", "tools used", "skills loaded", "reply"],
      [[q, "; ".join(f"{n}({a})" for n, a in u), ", ".join(s), r] for q, (u, s, r) in runs.items()],
    ),
    callout(
      f"A skill is a folder: SKILL.md's one-line description is always in the prompt ({len(prompt)} characters for "
      f"all three); its full instructions ({full} characters) are loaded only when a question needs them, and can "
      "point to more files. Loading a skill is recorded in state, so middleware can unlock tools: catering_quote is "
      "offered only after the catering skill is loaded.",
      "decision",
      "Skills",
    ),
  ] + [
    example(
      "lc.l39.1",
      "A skill is a folder",
      skill_folder,
      "",
      [table(["skill", "files"], [[n, ", ".join(f)] for n, (_, f) in folders.items()])],
    ),
    example("lc.l39.2", "Progressive disclosure", progressive_disclosure, "", [code(prompt, "markdown")]),
    example(
      "lc.l39.3",
      "The agent picks the skill",
      run_cafe,
      "",
      [table(["question", "skills"], [[q, ", ".join(s)] for q, (_, s, _) in runs.items()])],
    ),
    example(
      "lc.l39.4",
      "Skills unlock tools",
      run_cafe,
      "unlock_tools in nimbus/assistant/skills.py.",
      [table(["question", "tools"], [[q, "; ".join(n for n, _ in u)] for q, (u, _, _) in runs.items()])],
    ),
    example(
      "lc.l39.5",
      "Extra files, read only if needed",
      run_cafe,
      "",
      [answer(runs["Does the banana bread contain nuts?"][2], "allergen answer")],
    ),
  ]


# ── Lab 7: hierarchical teams ──────────────────────────────────────────────────
def team(specialists, name, members):
  """A team: a small supervisor over a few specialists, compiled into one graph."""
  from langgraph.graph import END, START, StateGraph
  from pydantic import BaseModel

  class Pick(BaseModel):
    member: str
    task: str

  class TeamState(TypedDict, total=False):
    customer_id: str
    task: str
    report: str

  picker = get_chat_model().with_structured_output(Pick, method="json_schema")

  def lead(state: TeamState) -> dict:
    pick = picker.invoke(
      f"Team {name}. Members: {members}. Pick the member for this task and restate it.\n{state['task']}"
    )
    member = pick.member if pick.member in members else next(iter(members))
    return {"report": f"{member}: " + specialists.ask(member, state["customer_id"], pick.task)}

  return StateGraph(TeamState).add_node("lead", lead).add_edge(START, "lead").add_edge("lead", END).compile(name=name)


def top_supervisor(specialists):
  from langgraph.graph import END, START, StateGraph
  from pydantic import BaseModel

  teams = {
    "customer_team": team(specialists, "customer_team", {"account": "orders, subscription", "cafe": "café menu"}),
    "info_team": team(specialists, "info_team", {"knowledge": "policies, cafés, products"}),
  }

  class Route(BaseModel):
    team: Literal["customer_team", "info_team", "FINISH"]
    task: str
    reply: str = ""

  class TopState(TypedDict, total=False):
    question: str
    customer_id: str
    reports: Annotated[list[str], operator.add]
    sent: Annotated[list[str], operator.add]
    route: dict
    guarded: str  # set when a code guard overrode the model's routing
    reply: str

  decide = get_chat_model().with_structured_output(Route, method="json_schema")

  def top(state: TopState) -> dict:
    route = decide.invoke(
      "Teams: customer_team (orders, subscription, café menu), info_team (policies, cafés, products). Question: "
      f"{state['question']}\nReports so far: {state.get('reports')}\nRoute the next part to a team, or FINISH with "
      "the reply."
    )
    if route.team != "FINISH" and route.team in state.get("sent", []):
      wanted = route.team
      route = Route(team="FINISH", task="", reply=" ".join(state.get("reports", [])))  # never re-send to a team
      return {"route": route.model_dump(), "reply": route.reply, "guarded": f"the top asked for {wanted} again"}
    return {"route": route.model_dump(), "reply": route.reply}

  def run_team(name):
    def node(state: TopState) -> dict:
      out = teams[name].invoke(
        {"customer_id": state["customer_id"], "task": state["route"]["task"]}
      )  # its steps stay inside
      return {"reports": [out["report"]], "sent": [name]}

    return node

  graph = StateGraph(TopState).add_node("top", top)
  for name in teams:
    graph.add_node(name, run_team(name)).add_edge(name, "top")
  graph.add_edge(START, "top").add_conditional_edges(
    "top", lambda s: END if s["route"]["team"] == "FINISH" else s["route"]["team"], [*teams, END]
  )
  graph = graph.compile()
  result = graph.invoke(
    {
      "question": "Where is my order NC-78123, and how fast is delivery to metro cities?",
      "customer_id": "c-5005",
      "reports": [],
      "sent": [],
    }
  )
  return graph, result


@lab(
  "agents.hierarchical", "Hierarchical teams: supervisors of supervisors", uses_model=True, calls="~10 calls to Mistral"
)
def hierarchical():
  """Two teams (customer, information), each a small supervisor graph over specialists, under a top supervisor that
  routes between teams and never sends to a team twice."""
  _, specialists = _parts()
  graph, result = top_supervisor(specialists)
  return [
    table(
      ["Team", "report (what the top sees)"], [[s, r] for s, r in zip(result["sent"], result["reports"], strict=False)]
    ),
    answer(result["reply"] or " ".join(result["reports"]), "Reply"),
    code(graph.get_graph().draw_mermaid(), "text", title="top → team → top … → FINISH"),
    callout(
      "A team is a supervisor graph over a few specialists, compiled into one graph; the top supervisor routes "
      "between teams and sees only their reports, never their inner steps. Code guards (never re-send to a team "
      "that reported) stop routing loops. Worth it with many agents: one supervisor choosing among ten or more gets "
      "unreliable. "
      + (
        f"Here the guard fired ({result['guarded']}) after the teams {result['sent']} had reported, so a part the "
        "top never routed went unanswered: a guard prevents loops, it can't repair the routing."
        if result.get("guarded")
        else f"Here the top sent the question to {result['sent']} and finished."
      ),
      "decision",
      "Hierarchical teams",
    ),
  ] + [
    example(
      "lg.l42.1", "A team = a supervisor graph", team, "", [table(["teams sent to"], [[s] for s in result["sent"]])]
    ),
    example(
      "lg.l42.2", "The top routes between teams", top_supervisor, "", [code(graph.get_graph().draw_mermaid(), "text")]
    ),
    example(
      "lg.l42.3",
      "Each level sees only what it needs; guards stop loops",
      top_supervisor,
      "",
      [table(["report"], [[r] for r in result["reports"]])],
    ),
    example(
      "lg.l42.4",
      "When to use it",
      top_supervisor,
      "Many agents; not three.",
      [table(["teams", "specialists"], [[2, 3]])],
    ),
  ]


# ── Lab 8: the Functional API ──────────────────────────────────────────────────
def functional_basics():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.func import entrypoint, task
  from langgraph.types import Command, interrupt

  runs = {"price": 0}

  @task
  def price(item: str) -> int:
    runs["price"] += 1
    return PRICES[item]

  @entrypoint(checkpointer=InMemorySaver())
  def order(items: list[str], *, previous: dict | None = None) -> dict:
    futures = [price(i) for i in items]  # tasks run in parallel; .result() waits
    total = sum(f.result() for f in futures)
    approved = interrupt(f"Charge ₹{total}?") if total > 300 else "auto"  # interrupt works the same way
    return {"total": total, "approved": approved, "orders_before": (previous or {}).get("orders_before", 0) + 1}

  first = order.invoke(["Oat Latte", "Cold Brew"], _thread("t"))
  paused = first["__interrupt__"][0].value
  done = order.invoke(Command(resume="yes"), _thread("t"))  # resumed: finished tasks are NOT re-run
  priced = runs["price"]  # 2: one per item, none again on resume
  again = order.invoke(["Cappuccino"], _thread("t"))  # `previous`: last return value on this thread
  return paused, done, again, priced


def functional_depth():
  from langchain_core.runnables import RunnableConfig
  from langgraph.cache.memory import InMemoryCache
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.errors import NodeTimeoutError
  from langgraph.func import entrypoint, task
  from langgraph.graph import START, StateGraph
  from langgraph.store.base import BaseStore
  from langgraph.store.memory import InMemoryStore
  from langgraph.types import CachePolicy, Command, RetryPolicy, StreamWriter, interrupt

  out: dict[str, Any] = {}
  store = InMemoryStore()
  store.put(("prefs",), "c-1042", {"drink": "oat latte"})

  @entrypoint(checkpointer=InMemorySaver(), store=store)
  def greet(customer: str, *, store: BaseStore, writer: StreamWriter, config: RunnableConfig) -> str:
    writer({"progress": f"looking up {customer}"})  # injected: store, writer, config
    usual = store.get(("prefs",), customer)
    return f"Your usual {usual.value['drink'] if usual else 'drink'}? (thread {config['configurable']['thread_id']})"

  out["injected"] = list(greet.stream("c-1042", _thread("g"), stream_mode=["custom", "values"]))

  @entrypoint(checkpointer=InMemorySaver())
  def tab(amount: int, *, previous: int | None = None) -> entrypoint.final[str, int]:
    total = (previous or 0) + amount
    return entrypoint.final(value=f"added ₹{amount}, total ₹{total}", save=total)  # return text, keep the number

  out["final"] = [tab.invoke(a, _thread("tab")) for a in (180, 220)]

  @task
  def draw() -> int:
    return random.randint(1, 1000)

  @entrypoint(checkpointer=InMemorySaver())
  def raffle_bad(_: Any) -> dict:
    ticket = random.randint(1, 1000)  # NOT in a task: re-rolled when the entrypoint re-runs on resume
    return {"asked": ticket, "announced": ticket, "ok": interrupt(f"Announce {ticket}?")}

  @entrypoint(checkpointer=InMemorySaver())
  def raffle_good(_: Any) -> dict:
    ticket = draw().result()  # in a task: replayed from the checkpoint on resume
    return {"asked": ticket, "announced": ticket, "ok": interrupt(f"Announce {ticket}?")}

  for name, flow in (("outside a task", raffle_bad), ("inside a task", raffle_good)):
    question = flow.invoke("go", _thread(name))["__interrupt__"][0].value
    out[f"determinism: random {name}"] = (
      f"asked '{question}', announced {flow.invoke(Command(resume=True), _thread(name))['announced']}"
    )

  attempts = {"n": 0}

  @task(retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05, retry_on=ConnectionError))
  def rate() -> float:
    attempts["n"] += 1
    if attempts["n"] < 3:
      raise ConnectionError("rates API down")
    return 0.012

  @task(cache_policy=CachePolicy(ttl=60))
  def slow_translation(item: str) -> str:
    time.sleep(0.5)
    return {"cappuccino": "カプチーノ"}[item]

  @entrypoint(cache=InMemoryCache())
  def price_card(item: str) -> dict:
    return {"name": slow_translation(item).result(), "usd_per_rupee": rate().result()}

  timings, tries = [], []
  for _ in range(2):
    started, before = time.perf_counter(), attempts["n"]
    price_card.invoke("cappuccino")
    timings.append(round(time.perf_counter() - started, 2))
    tries.append(attempts["n"] - before)
  out["retry and cache"] = (
    f"rate attempts per call {tries} (the first retried twice); seconds per call {timings} (the translation was "
    "cached the second time)"
  )

  calls = {"n": 0}

  @task(timeout=0.3, retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.05, retry_on=NodeTimeoutError))
  async def supplier(item: str) -> str:
    calls["n"] += 1
    await asyncio.sleep(1 if calls["n"] == 1 else 0.05)  # the first attempt hangs
    return f"{item}: in stock"

  @entrypoint()
  async def stock(item: str) -> str:
    return await supplier(item)

  out["timeout"] = f"{asyncio.run(stock.ainvoke('Coorg Dawn'))} after {calls['n']} attempts (the first timed out)"

  class Tax(TypedDict):
    amount: float

  tax = (
    StateGraph(Tax).add_node("tax", lambda s: {"amount": round(s["amount"] * 1.05, 2)}).add_edge(START, "tax").compile()
  )

  @entrypoint()
  def checkout(subtotal: float) -> dict:
    return {"subtotal": subtotal, "with_tax": tax.invoke({"amount": subtotal})["amount"]}  # a Graph-API graph inside

  out["mixing APIs"] = checkout.invoke(400)
  return out


@lab(
  "agents.functional-api",
  "The Functional API: plain Python control flow with LangGraph's features",
  calls="No model calls",
)
def functional_api():
  """An order with @task and @entrypoint (parallel tasks, an interrupt, resume, `previous`), then injected parameters,
  entrypoint.final, determinism, retries, caching, timeouts, and a Graph-API graph called from an entrypoint."""
  paused, done, again, price_runs = functional_basics()
  depth = functional_depth()
  return [
    table(
      ["Step", "Result"],
      [
        ["paused", paused],
        ["resumed", str(done)],
        ["price tasks run for 2 items, paused and resumed", price_runs],
        ["next order on the thread (previous)", str(again)],
      ],
    ),
    table(["Feature", "Result"], [[k, str(v)] for k, v in depth.items()]),
    callout(
      "The Functional API keeps LangGraph's checkpoints, interrupts, streaming, retries and caching but lets you "
      "write plain if/for/while: @task is a unit of work (a future; tasks run in parallel), @entrypoint the workflow. "
      f"On resume the entrypoint re-runs from the top and finished tasks replay their saved results (the two price "
      f"tasks ran {price_runs} times in total), so randomness, clock reads and side effects belong inside tasks.",
      "decision",
      "Functional API",
    ),
  ] + [
    example(
      "lg.l43.1", "@task: futures, run in parallel", functional_basics, "", [table(["total"], [[done["total"]]])]
    ),
    example("lg.l43.2", "@entrypoint: plain control flow", functional_basics, "", [code(str(done), "python")]),
    example(
      "lg.l43.3",
      "A checkpointer: no repeated work on resume",
      functional_basics,
      "",
      [table(["price tasks run"], [[price_runs]])],
    ),
    example("lg.l43.4", "interrupt() in an entrypoint", functional_basics, "", [code(str(paused), "text")]),
    example("lg.l43.5", "`previous`", functional_basics, "", [code(str(again), "python")]),
    example(
      "lg.l44.1", "Injected store, writer, config", functional_depth, "", [code(str(depth["injected"]), "python")]
    ),
    example(
      "lg.l44.2",
      "entrypoint.final(value, save)",
      functional_depth,
      "",
      [table(["calls"], [[x] for x in depth["final"]])],
    ),
    example(
      "lg.l44.3",
      "Determinism",
      functional_depth,
      "",
      [table(["", ""], [[k, v] for k, v in depth.items() if k.startswith("determinism")])],
    ),
    example(
      "lg.l44.4", "RetryPolicy and CachePolicy per task", functional_depth, "", [code(depth["retry and cache"], "text")]
    ),
    example("lg.l44.5", "Timeouts for async tasks", functional_depth, "", [code(depth["timeout"], "text")]),
    example("lg.l44.6", "Mixing APIs", functional_depth, "", [code(str(depth["mixing APIs"]), "python")]),
  ]


# ── Production path: three assistants, measured ───────────────────────────────
def evaluate_assistants() -> list[dict]:
  from nimbus.assistant.architectures import build_router, build_single, build_supervisor
  from nimbus.assistant.evaluation import load_questions, run_question

  retriever, specialists = _parts()
  model = get_chat_model()
  builds = {
    "single agent": build_single(model, specialists, retriever)[1],
    "router": build_router(model, specialists)[1],
    "supervisor": build_supervisor(model, specialists)[1],
  }
  rows = []
  for question in load_questions(PROJECT_DIR / "data" / "eval" / "assistant_questions.json"):
    for name, ask in builds.items():
      rows.append({"architecture": name, "question": question, **run_question(ask, question)})
  return rows


@lab(
  "agents.assistant-eval",
  "Evaluate: the Nimbus assistant, three ways",
  uses_model=True,
  calls="~200 calls to Mistral: 14 questions × 3 architectures, about 10 minutes",
)
def assistant_eval():
  """14 questions (knowledge, account, café, two-part, small talk) through one agent with every tool, a router, and
  a supervisor, all using the same specialists and tools."""
  started = time.perf_counter()
  rows = evaluate_assistants()
  seconds = time.perf_counter() - started
  out = []
  for name in ("single agent", "router", "supervisor"):
    mine = [r for r in rows if r["architecture"] == name]
    multi = [r for r in mine if r["question"]["kind"] == "multi"]
    ms = sorted(r["ms"] for r in mine)
    out.append(
      {
        "variant": name,
        "correct": sum(r["ok"] for r in mine) / len(mine),
        "multi": f"{sum(r['ok'] for r in multi)}/{len(multi)}",
        "calls": round(sum(r["calls"] for r in mine) / len(mine), 2),
        "tokens": round(sum(r["tokens"] for r in mine) / len(mine)),
        "p50_ms": round(ms[len(ms) // 2]),
        "errors": sum(r["error"] is not None for r in mine),
      }
    )
  misses = [
    [r["architecture"], r["question"]["id"], r["question"]["question"], ", ".join(r["used"]), r["error"] or r["reply"]]
    for r in rows
    if not r["ok"]
  ]
  best = max(out, key=lambda o: (o["correct"], -o["calls"]))
  return [
    scorecard(
      out,
      [
        Metric("correct", "correct"),
        Metric("multi", "two-part questions", "none", "text"),
        Metric("calls", "model calls / question", "lower", "text"),
        Metric("tokens", "tokens / question", "lower", "int"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
        Metric("errors", "errors", "lower", "int"),
      ],
      title=f"14 questions · {seconds:.0f} s",
    ),
    table(["Architecture", "id", "Question", "used", "reply"], misses, title="Every miss"),
    callout(
      f"{best['variant']} answered {best['correct']:.0%} at {best['calls']} model calls per question. "
      + "; ".join(
        f"{o['variant']}: {o['correct']:.0%}, {o['calls']} calls, p50 {o['p50_ms'] / 1000:.1f} s" for o in out
      )
      + ". More agents is not better by default: with three specialists and a model that handles a dozen tools, the "
      "architecture to ship is the cheapest one that answers correctly.",
      "decision",
      f"Ship the {best['variant']}",
    ),
  ]
