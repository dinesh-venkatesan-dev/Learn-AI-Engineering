"""Agents with LangGraph · Module 1 labs: state, nodes, edges and reducers. No model calls: graphs are plain code."""

import json
import operator
import time

from langgraph.errors import InvalidUpdateError

from academy.labs.runtime import Input, Metric, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR

MENU = {
  "Cold Brew": 200,
  "Oat Latte": 220,
  "Filter Coffee": 120,
  "Cappuccino": 180,
  "Banana Bread": 150,
  "Croissant": 140,
}
FOOD = {"Banana Bread", "Croissant"}


# ── Lab 1: state, nodes, edges and reducers ─────────────────────────────────────
# Each learning point is a small function: the page shows its body, and the lab runs it.


def state_without_reducers():
  from typing import TypedDict

  class OrderState(TypedDict):
    order: str  # what the customer typed
    items: list[str]  # no reducer: a node's update REPLACES the list
    notes: list[str]
    total: int

  return OrderState


def state_with_notes_reducer():
  import operator
  from typing import Annotated, TypedDict

  class OrderState(TypedDict):
    order: str
    items: list[str]  # still no reducer
    notes: Annotated[list[str], operator.add]  # a REDUCER: updates are appended to what's there
    total: int

  return OrderState


def state_with_reducers():
  import operator
  from typing import Annotated, TypedDict

  class OrderState(TypedDict):
    order: str
    items: Annotated[list[str], operator.add]  # current + update
    notes: Annotated[list[str], operator.add]
    total: int  # no reducer: the latest value wins, which is right for a total

  return OrderState


def state_with_custom_reducer():
  import operator
  from typing import Annotated, TypedDict

  def edit_order(current: list[str], update: list[str]) -> list[str]:
    """Any function (current, update) -> new value. Here "-Oat Latte" removes one Oat Latte."""
    items = list(current)
    for item in update:
      if item.startswith("-"):
        if item[1:] in items:
          items.remove(item[1:])
      else:
        items.append(item)
    return items

  class OrderState(TypedDict):
    order: str
    items: Annotated[list[str], edit_order]  # your own merge rule
    notes: Annotated[list[str], operator.add]
    total: int

  return OrderState


def write_the_nodes():
  # A node is a plain function: it reads the state and returns a PARTIAL update, only the keys it changes
  def take_order(state):
    return {"items": [i.strip() for i in state["order"].split(",") if i.strip()]}

  def suggest_food(state):
    if any(i in FOOD for i in state["items"]):
      return {"notes": ["already has food"]}
    return {"items": ["Banana Bread"], "notes": ["suggested Banana Bread"]}

  def check_loyalty(state):
    return {"notes": ["gold member: double points"]}  # writes `notes` in the same step as suggest_food

  def add_up(state):
    unknown = [i for i in state["items"] if i not in MENU]
    total = sum(MENU.get(i, 0) for i in state["items"])
    return {"total": total, "notes": [f"not on the menu: {', '.join(unknown)}"] if unknown else []}

  return take_order, suggest_food, check_loyalty, add_up


def wire_the_graph(OrderState, take_order, suggest_food, check_loyalty, add_up):  # noqa: N803
  from langgraph.graph import END, START, StateGraph

  builder = StateGraph(OrderState)
  builder.add_node("take_order", take_order)
  builder.add_node("suggest_food", suggest_food)
  builder.add_node("check_loyalty", check_loyalty)
  builder.add_node("add_up", add_up)
  builder.add_edge(START, "take_order")
  builder.add_edge("take_order", "suggest_food")  # two edges out of one node:
  builder.add_edge("take_order", "check_loyalty")  # both run, in the same step
  builder.add_edge(["suggest_food", "check_loyalty"], "add_up")  # add_up waits for both
  builder.add_edge("add_up", END)
  graph = builder.compile()  # checks the wiring and returns something you can run
  return graph


def a_typo_is_caught(OrderState, take_order):  # noqa: N803
  from langgraph.graph import START, StateGraph

  builder = StateGraph(OrderState)
  builder.add_node("take_order", take_order)
  builder.add_edge(START, "take_order")
  builder.add_edge("take_order", "add_upp")  # a node that doesn't exist
  try:
    builder.compile()
    error = None
  except ValueError as caught:  # found at compile time, before anything runs
    error = caught
  return error


def run_the_graph(graph, order):
  final = graph.invoke({"order": order, "notes": []})  # runs to END, returns the final state
  return final


def watch_each_step(graph, order):
  steps = []
  for update in graph.stream({"order": order, "notes": []}, stream_mode="updates"):
    steps.append(update)  # {node name: the partial update it returned}
  return steps


def two_writers_without_a_reducer(graph, order):
  from langgraph.errors import InvalidUpdateError

  try:
    graph.invoke({"order": order, "notes": []})
    error = None
  except InvalidUpdateError as caught:  # suggest_food and check_loyalty both wrote `notes` in one step
    error = caught
  return error


def messages_with_add_messages():
  from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
  from langgraph.graph.message import add_messages

  after = {}
  history = add_messages([], [HumanMessage("A cold brew, please", id="1")])
  after["append a question"] = history
  history = add_messages(history, [AIMessage("That's ₹200", id="2")])
  after["append the reply"] = history
  history = add_messages(history, [AIMessage("That's ₹180 with your discount", id="2")])  # same id
  after["same id: REPLACE the reply"] = history
  history = add_messages(history, [RemoveMessage(id="1")])
  after["RemoveMessage: delete the question"] = history
  return after


def replace_despite_a_reducer(OrderState, take_order):  # noqa: N803
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Overwrite

  def clear_order(state):
    return {"items": Overwrite([])}  # skip the reducer: really replace, here with an empty order

  builder = StateGraph(OrderState)
  builder.add_node("take_order", take_order)
  builder.add_node("clear_order", clear_order)
  builder.add_edge(START, "take_order")
  builder.add_edge("take_order", "clear_order")
  builder.add_edge("clear_order", END)
  cleared = builder.compile().invoke({"order": "Cold Brew, Croissant", "notes": []})
  return cleared


def _expected(order: str) -> tuple[list[str], int]:
  """What the order should be: removals applied, food suggested when there's none."""
  items: list[str] = []
  for item in (i.strip() for i in order.split(",") if i.strip()):
    if item.startswith("-"):
      if item[1:] in items:
        items.remove(item[1:])
    else:
      items.append(item)
  if not any(i in FOOD for i in items):
    items.append("Banana Bread")
  return items, sum(MENU.get(i, 0) for i in items)


@lab(
  "agents.graph-basics",
  "Graphs: state, nodes, edges and reducers",
  inputs=[
    Input(
      "order",
      "Order",
      "text",
      default="Cold Brew, Oat Latte, Filter Coffee, -Oat Latte",
      help=f"Comma separated. A leading - removes an item. Menu: {', '.join(MENU)}.",
    ),
  ],
)
def graph_basics(order: str):
  """One order graph, four state schemas: what each reducer choice does to the same run."""
  nodes = write_the_nodes()
  variants = [
    ("no reducers", state_without_reducers()),
    ("operator.add on notes only", state_with_notes_reducer()),
    ("operator.add on both lists", state_with_reducers()),
    ("custom reducer: edit_order", state_with_custom_reducer()),
  ]
  graphs = {label: wire_the_graph(state, *nodes) for label, state in variants}
  expected_items, expected_total = _expected(order)
  rows, finals = [], {}
  for label, graph in graphs.items():
    try:
      final = run_the_graph(graph, order)
    except InvalidUpdateError as error:
      finals[label] = None
      rows.append({"variant": label, "result": "✗ error", "items": "—", "total": None, "notes": str(error)[:90]})
      continue
    finals[label] = final
    right = final["items"] == expected_items and final["total"] == expected_total
    rows.append(
      {
        "variant": label,
        "result": "✓ right" if right else "✗ wrong order",
        "items": ", ".join(final["items"]),
        "total": final["total"],
        "notes": "; ".join(final["notes"]),
      }
    )
  custom = graphs["custom reducer: edit_order"]
  steps = watch_each_step(custom, order)
  plain = graphs["no reducers"]
  collision = two_writers_without_a_reducer(plain, order)
  typo = a_typo_is_caught(state_with_reducers(), nodes[0])
  history = messages_with_add_messages()
  replaced = run_the_graph(graphs["operator.add on notes only"], "Cold Brew, Oat Latte")
  cleared = replace_despite_a_reducer(state_with_reducers(), nodes[0])
  added = finals["operator.add on both lists"]
  return [
    scorecard(
      rows,
      [
        Metric("result", "Result", "none", "text", "Compared with the order the customer meant"),
        Metric("items", "Final items", "none", "text"),
        Metric("total", "Total ₹", "none", "int"),
        Metric("notes", "Notes / error", "none", "text"),
      ],
      title=f"The same graph, four state schemas · expected: {', '.join(expected_items)} = ₹{expected_total}",
    ),
    table(
      ["Step", "Node", "Partial update it returned"],
      [
        [i + 1, node, json.dumps(update, ensure_ascii=False)]
        for i, chunk in enumerate(steps)
        for node, update in chunk.items()
      ],
      title="Step by step (custom reducer): each node returns only what it changes",
    ),
    code(custom.get_graph().draw_mermaid(), "text", title="The graph (Mermaid)"),
    callout(
      "Give a key a reducer when more than one node adds to it (lists of items, notes, messages), and none when "
      "the latest value is the truth (a total, a status). Without one, two nodes writing the same key in one step "
      "is an error, and a later node silently replaces earlier work. When the merge rule is part of your domain "
      "(removals, de-duplication, caps), write it as a custom reducer so every node gets it for free.",
      "decision",
      "Which reducer",
    ),
  ] + [
    example(
      "lc.l48.1",
      "State: a TypedDict every node shares",
      state_without_reducers,
      "The state schema lists the keys of the graph's shared memory and their types. A TypedDict only documents "
      "them: nothing is checked at run time.",
      [
        table(["Key", "Type"], [[k, getattr(v, "__name__", str(v))] for k, v in variants[0][1].__annotations__.items()])
      ],
    ),
    example(
      "lc.l48.2",
      "Nodes: state in, partial update out",
      write_the_nodes,
      "Each node returns only the keys it changes. LangGraph merges that update into the state before the next "
      'step. Below: what each node returned in this run (from `stream_mode="updates"`).',
      [table(["Node", "Returned"], [[n, json.dumps(u, ensure_ascii=False)] for c in steps for n, u in c.items()])],
    ),
    example(
      "lc.l48.2",
      "Watching the partial updates",
      watch_each_step,
      '`stream(stream_mode="updates")` yields one chunk per finished node: `{node: update}`. The fastest way to '
      "see what each node actually did.",
      [code(json.dumps(steps, indent=2, ensure_ascii=False), "json")],
    ),
    example(
      "lc.l48.3",
      "Edges: START → nodes → END",
      wire_the_graph,
      "`add_edge(a, b)` always runs `b` after `a`. Two edges out of `take_order` run both nodes in the same step; "
      "a list as the source makes `add_up` wait for both. `compile()` returns the runnable graph.",
      [code(custom.get_graph().draw_mermaid(), "text")],
    ),
    example(
      "lc.l48.3",
      "compile() checks the wiring",
      a_typo_is_caught,
      "A misspelt node name fails when you compile, not halfway through a customer's request.",
      [code(f"{type(typo).__name__}: {typo}", "text")],
    ),
    example(
      "lg.l05.1",
      "No reducer: the new value replaces the old one",
      state_with_notes_reducer,
      "Run with the order *Cold Brew, Oat Latte* (no food). `items` has no reducer, so `suggest_food`'s update "
      '`{"items": ["Banana Bread"]}` replaced the whole order instead of adding to it. No error: just a wrong '
      "order.",
      [table(["Key", "Final value"], [["items", ", ".join(replaced["items"])], ["total", f"₹{replaced['total']}"]])],
    ),
    example(
      "lg.l05.2",
      "operator.add: updates accumulate",
      state_with_reducers,
      "`Annotated[list[str], operator.add]` makes the update `current + update`. Every node's items are kept, but "
      "a removal like `-Oat Latte` is just another string to `+`.",
      [
        table(
          ["Key", "Final value"],
          [["items", ", ".join(added["items"])], ["total", f"₹{added['total']}"], ["notes", "; ".join(added["notes"])]],
        )
      ],
    ),
    example(
      "lg.l05.3",
      "A custom reducer: your own merge rule",
      state_with_custom_reducer,
      "Any function `(current, update) -> new value` works. `edit_order` treats `-Oat Latte` as a removal, so "
      "every node that writes `items` gets the rule without knowing about it.",
      [
        table(
          ["Key", "Final value"],
          [
            ["items", ", ".join(finals["custom reducer: edit_order"]["items"])],
            ["total", f"₹{finals['custom reducer: edit_order']['total']}"],
          ],
        )
      ],
    ),
    example(
      "lg.l05.4",
      "add_messages: append, replace by id, delete",
      messages_with_add_messages,
      "The reducer behind every chat and agent graph (`MessagesState` uses it). A message with an existing id "
      "replaces it, and `RemoveMessage(id)` deletes one.",
      [
        table(
          ["After", "Messages (type · id · text)"],
          [[step, " | ".join(f"{m.type} · {m.id} · {m.text}" for m in ms)] for step, ms in history.items()],
        )
      ],
    ),
    example(
      "lc.l48.5",
      "Why chat graphs use add_messages",
      messages_with_add_messages,
      "With `operator.add`, a corrected reply would be appended next to the wrong one and nothing could be "
      "deleted. `add_messages` merges by id, which is what a conversation needs.",
      [
        code(
          json.dumps([{"type": m.type, "id": m.id, "text": m.text} for m in list(history.values())[-1]], indent=2),
          "json",
          title="The final history",
        )
      ],
    ),
    example(
      "lg.l05.5",
      "Overwrite: bypass the reducer",
      replace_despite_a_reducer,
      "`items` uses `operator.add`, so returning `[]` would add nothing. `Overwrite([])` replaces the value: here, "
      "clearing the order.",
      [table(["Key", "Final value"], [["items", repr(cleared["items"])]])],
    ),
    example(
      "lg.l05.6",
      "Two writers in one step need a reducer",
      two_writers_without_a_reducer,
      "`suggest_food` and `check_loyalty` run in the same step and both return `notes`. Without a reducer "
      "LangGraph can't know how to combine them, so it stops instead of guessing.",
      [code(f"{type(collision).__name__}: {str(collision).splitlines()[0]}", "text")],
    ),
    example(
      "lc.l48.4",
      "Reducers, side by side",
      run_the_graph,
      "The same `invoke` on the four graphs above. The reducer you choose per key decides the result.",
      [
        table(
          ["State schema", "Result", "items", "total"],
          [[r["variant"], r["result"], r["items"], r["total"]] for r in rows],
        )
      ],
    ),
  ]


# ── Lab 2: why LangGraph ────────────────────────────────────────────────────────
PRODUCTS = [
  "Nimbus Coffee's cold brew",
  "the Nilgiri Sunrise blend",
  "Nimbus's oat latte",
  "the Monsoon Malabar Reserve",
  "a Nimbus coffee subscription",
]
MAX_WORDS, MAX_ATTEMPTS = 3, 3


def a_chain_runs_once(model, product):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  chain = (
    ChatPromptTemplate.from_template("Write a tagline for {product}. {feedback} Reply with the tagline only.")
    | model
    | StrOutputParser()
  )
  tagline = chain.invoke({"product": product, "feedback": ""}).strip().strip('*"')
  return chain, tagline  # if it's too long, the chain can't notice or try again


def a_graph_that_loops(chain):
  from typing import TypedDict

  from langgraph.graph import END, START, StateGraph

  class State(TypedDict):
    product: str
    tagline: str
    attempts: int
    feedback: str

  def write(state):
    text = chain.invoke({"product": state["product"], "feedback": state["feedback"]}).strip().strip('*"')
    return {"tagline": text, "attempts": state["attempts"] + 1}

  def check(state):
    words = len(state["tagline"].split())
    return {"feedback": "" if words <= MAX_WORDS else f"Your last tagline had {words} words; use at most {MAX_WORDS}."}

  def route(state):
    if not state["feedback"]:
      return "done"
    return "give_up" if state["attempts"] >= MAX_ATTEMPTS else "retry"  # a safety limit on the loop

  builder = StateGraph(State)
  builder.add_node("write", write)
  builder.add_node("check", check)
  builder.add_edge(START, "write")
  builder.add_edge("write", "check")
  builder.add_conditional_edges("check", route, {"retry": "write", "done": END, "give_up": END})  # the loop
  return builder.compile()


WHAT_ELSE = [
  ("branching, loops, parallel steps, Send", "Module 2: Control flow"),
  ("workflow patterns: routing, orchestrator-workers, evaluator-optimizer", "Module 3"),
  ("agents that call tools in a loop, streaming", "Module 4"),
  ("memory that survives restarts, time travel", "Module 5"),
  ("pausing for a person, then resuming", "Module 6"),
  ("subgraphs and teams of agents", "Module 7"),
  ("retries, timeouts, tests, serving", "Module 8"),
]


@lab(
  "agents.why-langgraph",
  "Why LangGraph: a chain runs once, a graph can check and try again",
  uses_model=True,
  calls="5 to ~20 calls to Mistral",
)
def why_langgraph():
  """Five taglines with a 3-word limit: a chain gets one try; a graph writes, checks and loops back (at most 3)."""
  from nimbus.llm import get_chat_model  # not cached: temperature 0.7, variety is the point

  model = get_chat_model(temperature=0.7)
  rows, last_graph = [], None
  for product in PRODUCTS:
    chain, tagline = a_chain_runs_once(model, product)
    graph = a_graph_that_loops(chain)
    final = graph.invoke({"product": product, "tagline": "", "attempts": 0, "feedback": ""})
    last_graph = graph
    rows.append(
      [product, tagline, len(tagline.split()) <= MAX_WORDS, final["tagline"], final["attempts"], not final["feedback"]]
    )
  chain_ok, graph_ok = sum(r[2] for r in rows), sum(r[5] for r in rows)
  return [
    scorecard(
      [
        {"variant": "chain (one pass)", "within_limit": f"{chain_ok}/{len(rows)}", "calls": len(rows)},
        {
          "variant": "graph (write → check → retry, max 3)",
          "within_limit": f"{graph_ok}/{len(rows)}",
          "calls": sum(r[4] for r in rows),
        },
      ],
      [
        Metric("within_limit", f"taglines of ≤ {MAX_WORDS} words", "none", "text"),
        Metric("calls", "model calls", "lower", "int"),
      ],
      title="Same model, same prompt",
    ),
    table(
      ["Product", "chain", "≤3?", "graph", "attempts", "≤3?"],
      [[r[0], r[1], "✓" if r[2] else "✗", r[3], r[4], "✓" if r[5] else "✗"] for r in rows],
    ),
    code(last_graph.get_graph().draw_mermaid(), "text", title="The graph"),
    callout(
      f"The chain met the limit {chain_ok}/{len(rows)} times and had no way to notice when it didn't. The graph "
      f"checked its own output and looped back with feedback: {graph_ok}/{len(rows)}, for "
      f"{sum(r[4] for r in rows) - len(rows)} "
      "extra calls, with a limit so it can't loop forever. That loop (state shared between steps, edges that can go "
      "back) is what LangGraph adds; every agent is built on it.",
      "decision",
      "Why a graph",
    ),
  ] + [
    example(
      "lg.l01.1",
      "A chain runs once",
      a_chain_runs_once,
      "",
      [table(["product", "tagline", "words"], [[r[0], r[1], len(r[1].split())] for r in rows])],
    ),
    example(
      "lg.l01.2",
      "State + nodes + edges, and edges can loop",
      a_graph_that_loops,
      "",
      [code(last_graph.get_graph().draw_mermaid(), "text")],
    ),
    example(
      "lg.l01.3",
      "Write → check → rewrite, with a safety limit",
      a_graph_that_loops,
      "",
      [table(["product", "attempts", "final", "within limit"], [[r[0], r[4], r[3], r[5]] for r in rows])],
    ),
    example(
      "lg.l01.4",
      "What else LangGraph adds",
      a_graph_that_loops,
      "Where this track covers each feature.",
      [table(["feature", "where"], WHAT_ELSE)],
    ),
  ]


# ── Lab 3: state schemas ────────────────────────────────────────────────────────
PRICES = {"Cappuccino": 180, "Oat Latte": 220, "Cold Brew": 200}


def one_node_graph(schema, node):
  from langgraph.graph import START, StateGraph

  return StateGraph(schema).add_node("node", node).add_edge(START, "node").compile()


def typeddict_state():
  from typing import TypedDict

  class Order(TypedDict):
    item: str
    quantity: int
    total: float

  def price(state):
    return {"total": PRICES[state["item"]] * state["quantity"]}

  good = one_node_graph(Order, price).invoke({"item": "Cappuccino", "quantity": 2, "total": 0})
  bad = one_node_graph(Order, price).invoke({"item": "Cappuccino", "quantity": "2", "total": 0})  # not checked!
  return good, bad


def pydantic_state():
  from pydantic import BaseModel, ValidationError

  class Order(BaseModel):
    item: str
    quantity: int = 1  # a default
    total: float = 0.0

  def price(state):
    return {"total": PRICES[state.item] * state.quantity}  # attributes, not keys

  graph = one_node_graph(Order, price)
  defaulted = graph.invoke({"item": "Oat Latte"})
  coerced = graph.invoke({"item": "Cappuccino", "quantity": "2"})  # "2" → 2
  try:
    graph.invoke({"item": "Cappuccino", "quantity": "two"})
    rejected = None
  except ValidationError as error:
    rejected = str(error).splitlines()[0]
  return defaulted, coerced, rejected


def dataclass_state():
  from dataclasses import dataclass, field

  @dataclass
  class Order:
    item: str
    quantity: int = 1
    notes: list[str] = field(default_factory=list)

  def add_note(state):
    return {"notes": [*state.notes, f"{state.quantity} x {state.item}"]}

  return one_node_graph(Order, add_note).invoke({"item": "Cold Brew"})


def partial_updates():
  from typing import TypedDict

  class Order(TypedDict):
    item: str
    quantity: int
    total: float

  def price(state):
    return {"total": PRICES[state["item"]] * state["quantity"]}  # only the key it changes

  return one_node_graph(Order, price).invoke({"item": "Oat Latte", "quantity": 3, "total": 0})


def messages_state():
  from langchain_core.messages import AIMessage, HumanMessage
  from langgraph.graph import MessagesState

  def reply(state):
    return {"messages": [AIMessage(f"You said: {state['messages'][-1].text}")]}  # appended (add_messages)

  return one_node_graph(MessagesState, reply).invoke({"messages": [HumanMessage("Hi Nimbus!")]})


@lab("agents.state-schemas", "State schemas: TypedDict, Pydantic, dataclass and MessagesState")
def state_schemas():
  """The same one-node order graph with each kind of state schema: what each checks, defaults and returns."""
  good, bad = typeddict_state()
  defaulted, coerced, rejected = pydantic_state()
  dc = dataclass_state()
  partial = partial_updates()
  msgs = messages_state()
  rows = [
    {
      "variant": "TypedDict",
      "checks_types": "no",
      "defaults": "no",
      "wrong_input": f"quantity '2' → total {str(bad['total'])[:20]}…",
    },
    {
      "variant": "Pydantic BaseModel",
      "checks_types": "yes (coerces '2' → 2)",
      "defaults": "yes",
      "wrong_input": f"'two' → {rejected}",
    },
    {"variant": "dataclass", "checks_types": "no", "defaults": "yes", "wrong_input": "—"},
  ]
  return [
    scorecard(
      rows,
      [
        Metric("checks_types", "validates input", "none", "text"),
        Metric("defaults", "defaults", "none", "text"),
        Metric("wrong_input", "with a wrong input", "none", "text"),
      ],
      title="Three ways to declare state",
    ),
    callout(
      "A TypedDict documents the keys but checks nothing: quantity '2' times 180 repeated the string. Pydantic "
      "validates (and coerces) what a node reads and supports defaults, at a small cost per step; a dataclass gives "
      "defaults without validation. Use Pydantic at the boundary where outside data enters, TypedDict inside. "
      "MessagesState is the ready-made schema for conversations.",
      "decision",
      "Which schema",
    ),
  ] + [
    example(
      "lg.l02.1",
      "TypedDict: keys and types, no checks",
      typeddict_state,
      "",
      [table(["input quantity", "total"], [["2", good["total"]], ["'2' (a string)", str(bad["total"])[:40] + "…"]])],
    ),
    example(
      "lg.l02.2",
      "Pydantic: validated, with defaults",
      pydantic_state,
      "",
      [
        table(
          ["input", "result"],
          [
            ["no quantity", f"total {defaulted['total']} (quantity defaulted to 1)"],
            ["quantity '2'", f"total {coerced['total']}"],
            ["quantity 'two'", rejected],
          ],
        )
      ],
    ),
    example(
      "lg.l02.3",
      "Dataclass: defaults without Pydantic",
      dataclass_state,
      "",
      [table(["result"], [[json.dumps(dc, default=str)]])],
    ),
    example(
      "lg.l02.4",
      "Nodes return partial updates",
      partial_updates,
      "item and quantity weren't returned, so they keep their values.",
      [code(json.dumps(partial), "json")],
    ),
    example(
      "lg.l02.5",
      "MessagesState for chat",
      messages_state,
      "",
      [table(["type", "text"], [[m.type, m.text] for m in msgs["messages"]])],
    ),
  ]


# ── Lab 4: nodes ────────────────────────────────────────────────────────────────
def function_nodes():
  from typing import TypedDict

  from langgraph.graph import START, StateGraph

  class State(TypedDict):
    text: str
    log: str

  seen = []

  def shout(state):
    return {"text": state["text"].upper()}

  def observe(state):  # returns nothing: no update
    seen.append(state["text"])

  graph = StateGraph(State).add_sequence([shout, observe]).add_edge(START, "shout").compile()
  return graph.invoke({"text": "hello", "log": ""}), seen


def config_and_runtime():
  from dataclasses import dataclass
  from typing import TypedDict

  from langchain_core.runnables import RunnableConfig
  from langgraph.graph import START, StateGraph
  from langgraph.runtime import Runtime

  class State(TypedDict):
    log: str

  @dataclass
  class Context:
    customer_id: str  # per-run facts that aren't state: who is asking

  def who_and_where(state, config: RunnableConfig, runtime: Runtime[Context]):
    thread = config["configurable"].get("thread_id", "-")  # declare a parameter, LangGraph passes it in
    return {"log": f"customer={runtime.context.customer_id} thread={thread} tags={config.get('tags')}"}

  graph = StateGraph(State, context_schema=Context).add_node(who_and_where).add_edge(START, "who_and_where").compile()
  return graph.invoke(
    {"log": ""}, config={"configurable": {"thread_id": "t-1"}, "tags": ["support"]}, context=Context("c-1042")
  )


def a_runnable_node():
  from typing import TypedDict

  from langchain_core.runnables import RunnableLambda
  from langgraph.graph import START, StateGraph

  class State(TypedDict):
    text: str
    log: str

  count = RunnableLambda(lambda state: {"log": f"{len(state['text'].split())} words"})  # any Runnable, e.g. a chain
  return (
    StateGraph(State)
    .add_node("count", count)
    .add_edge(START, "count")
    .compile()
    .invoke({"text": "graphs are made of nodes", "log": ""})
  )


def async_nodes():
  import asyncio
  from typing import TypedDict

  from langgraph.graph import START, StateGraph

  class State(TypedDict):
    order_id: str
    log: str

  async def look_up(state):
    await asyncio.sleep(0.2)  # an HTTP call: the event loop serves others meanwhile
    return {"log": f"looked up {state['order_id']}"}

  graph = StateGraph(State).add_node(look_up).add_edge(START, "look_up").compile()
  return asyncio.run(graph.ainvoke({"order_id": "NC-1004", "log": ""}))


def node_names():
  from typing import TypedDict

  from langgraph.graph import START, StateGraph

  class State(TypedDict):
    text: str

  def clean(state):
    return {"text": state["text"].strip()}

  def shout(state):
    return {"text": state["text"].upper()}

  graph = StateGraph(State).add_sequence([clean, shout]).add_edge(START, "clean").compile()  # names from functions
  named = StateGraph(State).add_node("tidy_up", clean).add_edge(START, "tidy_up").compile()  # or explicit names
  return list(graph.get_graph().nodes), list(named.get_graph().nodes)


@lab("agents.nodes", "Nodes: functions, config and context, Runnables, async")
def nodes_lab():
  """Every way to write a node, run."""
  result, seen = function_nodes()
  runtime = config_and_runtime()
  runnable = a_runnable_node()
  started = time.perf_counter()
  async_result = async_nodes()
  async_ms = (time.perf_counter() - started) * 1000
  auto, named = node_names()
  return [
    table(
      ["Node kind", "Result"],
      [
        ["plain function (and one that returns nothing)", f"{json.dumps(result)}; observe saw {seen}"],
        ["reads config and runtime context", runtime["log"]],
        ["a Runnable", runnable["log"]],
        ["async (ainvoke)", f"{async_result['log']} in {async_ms:.0f} ms"],
        ["names", f"from functions: {auto}; explicit: {named}"],
      ],
    ),
    callout(
      "A node is a function from state to a partial update (or nothing). Ask for `config` or `runtime` by declaring "
      "the parameter: per-run facts like the customer id belong in the runtime context, not the state. Any Runnable "
      "works as a node, so existing chains drop in, and async nodes free the event loop during I/O.",
      "decision",
      "Nodes",
    ),
  ] + [
    example(
      "lg.l03.1",
      "Function nodes, and one that changes nothing",
      function_nodes,
      "",
      [table(["result", "observe saw"], [[json.dumps(result), str(seen)]])],
    ),
    example("lg.l03.2", "config and runtime context", config_and_runtime, "", [table(["log"], [[runtime["log"]]])]),
    example("lg.l03.3", "A Runnable as a node", a_runnable_node, "", [table(["log"], [[runnable["log"]]])]),
    example(
      "lg.l03.4",
      "Async nodes with ainvoke()",
      async_nodes,
      "",
      [table(["log", "time"], [[async_result["log"], f"{async_ms:.0f} ms"]])],
    ),
    example(
      "lg.l03.5",
      "Names and add_sequence",
      node_names,
      "",
      [table(["from function names", "explicit"], [[str(auto), str(named)]])],
    ),
  ]


# ── Lab 5: edges ────────────────────────────────────────────────────────────────
def sequence_vs_fan_out(delay):
  import time
  from typing import Annotated, TypedDict

  from langgraph.graph import END, START, StateGraph

  class State(TypedDict):
    done: Annotated[list[str], operator.add]

  def node(name):
    def run(state):
      time.sleep(delay)  # a lookup that takes a while
      return {"done": [name]}

    return run

  line = StateGraph(State).add_sequence([("stock", node("stock")), ("price", node("price"))])
  line.add_edge(START, "stock")
  fan = StateGraph(State)
  fan.add_node("stock", node("stock"))
  fan.add_node("price", node("price"))
  fan.add_edge(START, "stock")  # two edges from START: both run in the same step, in parallel
  fan.add_edge(START, "price")
  fan.add_edge(["stock", "price"], END)
  timings = {}
  for label, graph in (
    ("add_sequence (one after the other)", line.compile()),
    ("fan-out (both at once)", fan.compile()),
  ):
    started = time.perf_counter()
    graph.invoke({"done": []})
    timings[label] = (time.perf_counter() - started) * 1000
  return timings


def wiring_errors():
  from typing import TypedDict

  from langgraph.graph import END, START, StateGraph

  class State(TypedDict):
    x: int

  errors = {}
  typo = StateGraph(State).add_node("a", lambda s: {"x": 1}).add_edge(START, "a").add_edge("a", "b")  # no node b
  orphan = (
    StateGraph(State)
    .add_node("a", lambda s: {"x": 1})
    .add_node("b", lambda s: {"x": 2})
    .add_edge(START, "a")
    .add_edge("a", END)
  )
  for name, builder in (("edge to a missing node", typo), ("a node nothing reaches", orphan)):
    try:
      builder.compile()
      errors[name] = "compiled"
    except ValueError as error:
      errors[name] = f"{type(error).__name__}: {error}"
  return errors


@lab(
  "agents.edges",
  "Edges: sequences, fan-out, and wiring checked at compile time",
  inputs=[Input("delay", "Each node takes (seconds)", "number", default=0.3, min=0.1, max=1.0)],
)
def edges_lab(delay: float):
  """Two slow nodes in a line and fanned out; then two wiring mistakes compile() catches."""
  timings = sequence_vs_fan_out(delay)
  errors = wiring_errors()
  rows = [{"variant": k, "ms": round(v)} for k, v in timings.items()]
  return [
    scorecard(rows, [Metric("ms", "wall time", "lower", "ms")], title=f"Two nodes of {delay:.1f} s each"),
    table(["Mistake", "compile()"], [[k, v] for k, v in errors.items()]),
    callout(
      "Normal edges always run; add_sequence is shorthand for a straight line. Two edges out of one node run both "
      "targets in the same step (here in about the time of one). compile() checks the wiring before anything runs: "
      "an edge to a missing node fails immediately. A node that nothing reaches is "
      + (
        "also rejected." if "Error" in errors["a node nothing reaches"] else "accepted, so read your graph's diagram."
      ),
      "decision",
      "Edges",
    ),
  ] + [
    example(
      "lg.l04.1",
      "Normal edges: always taken",
      sequence_vs_fan_out,
      "",
      [table(["graph", "ms"], [[r["variant"], r["ms"]] for r in rows])],
    ),
    example(
      "lg.l04.2",
      "add_sequence",
      sequence_vs_fan_out,
      "One call adds the nodes and the edges between them.",
      [table(["graph", "ms"], [[rows[0]["variant"], rows[0]["ms"]]])],
    ),
    example(
      "lg.l04.3",
      "Two outgoing edges: both run, in parallel",
      sequence_vs_fan_out,
      "",
      [table(["graph", "ms"], [[rows[1]["variant"], rows[1]["ms"]]])],
    ),
    example(
      "lg.l04.4",
      "compile() checks the wiring",
      wiring_errors,
      "",
      [table(["mistake", "result"], [[k, v] for k, v in errors.items()])],
    ),
  ]


# ── Lab 6: running and inspecting ───────────────────────────────────────────────
def an_order_graph():
  from typing import Annotated, TypedDict

  from langgraph.graph import START, StateGraph

  class State(TypedDict):
    items: list[str]
    subtotal: int
    total: float
    log: Annotated[list[str], operator.add]

  def add_up(state):
    return {"subtotal": sum(PRICES[i] for i in state["items"]), "log": ["add_up"]}

  def add_tax(state):
    return {"total": round(state["subtotal"] * 1.05, 2), "log": ["add_tax"]}

  return StateGraph(State).add_sequence([add_up, add_tax]).add_edge(START, "add_up").compile()


def run_every_way(graph, order):
  final = graph.invoke(order)  # the final state
  values = list(graph.stream(order, stream_mode="values"))  # the whole state after every step
  updates = list(graph.stream(order, stream_mode="updates"))  # only what each node changed
  tasks = list(graph.stream(order, stream_mode="tasks"))  # node start and finish events
  many = graph.batch([{**order, "items": ["Oat Latte"]}, {**order, "items": ["Cappuccino", "Cappuccino"]}])
  return final, values, updates, tasks, many


@lab("agents.running", "Running and inspecting graphs: invoke, stream modes, batch, diagrams")
def running():
  """One small order graph run every way LangGraph offers."""
  graph = an_order_graph()
  order = {"items": ["Cappuccino", "Cold Brew"], "subtotal": 0, "total": 0.0, "log": []}
  final, values, updates, tasks, many = run_every_way(graph, order)
  try:
    ascii_art = graph.get_graph().draw_ascii()
  except ImportError:
    ascii_art = "(draw_ascii needs the grandalf package; Mermaid below needs nothing)"
  return [
    table(
      ["Way", "What you get"],
      [
        ["invoke", json.dumps(final)],
        ["stream values", f"{len(values)} full states"],
        ["stream updates", json.dumps(updates)],
        [
          "stream tasks",
          f"{len(tasks)} events: " + ", ".join(("finish " if "result" in t else "start ") + t["name"] for t in tasks),
        ],
        ["batch (2 orders)", str([r["total"] for r in many])],
      ],
    ),
    code(graph.get_graph().draw_mermaid(), "text", title="Mermaid"),
    callout(
      "invoke returns the final state; stream shows the run as it happens: `values` for the whole state after each "
      "step (debugging), `updates` for what each node changed (progress in a UI), `tasks` for when nodes start and "
      "finish (timing). batch runs many inputs. The Mermaid text renders anywhere, which is how this track draws "
      "graphs.",
      "decision",
      "Inspecting",
    ),
  ] + [
    example("lg.l06.1", "invoke(): the final state", run_every_way, "", [code(json.dumps(final, indent=2), "json")]),
    example("lg.l06.2", "stream_mode='values'", run_every_way, "", [code(json.dumps(values, indent=2)[:900], "json")]),
    example("lg.l06.3", "stream_mode='updates'", run_every_way, "", [code(json.dumps(updates, indent=2), "json")]),
    example(
      "lg.l06.4",
      "stream_mode='tasks'",
      run_every_way,
      "",
      [table(["event", "node"], [["finish" if "result" in t else "start", t["name"]] for t in tasks])],
    ),
    example(
      "lg.l06.5",
      "batch(): many inputs",
      run_every_way,
      "",
      [
        table(
          ["items", "total"],
          [[str(i), r["total"]] for i, r in zip((["Oat Latte"], ["Cappuccino", "Cappuccino"]), many, strict=True)],
        )
      ],
    ),
    example(
      "lg.l06.6",
      "ASCII and Mermaid",
      an_order_graph,
      "",
      [code(ascii_art, "text", "draw_ascii()"), code(graph.get_graph().draw_mermaid(), "text", "draw_mermaid()")],
    ),
  ]


# ── Production path: the support graph ─────────────────────────────────────────
STATUS_WORDS = {
  "delivered": ["delivered"],
  "shipped": ["shipped", "on its way", "dispatched", "in transit"],
  "processing": ["processing", "being prepared", "not yet shipped", "hasn't shipped", "being processed"],
  "packed": ["packed"],
}


def evaluate_support_graph() -> list[dict]:
  import re as _re

  from nimbus.llm import get_chat_model
  from nimbus.llm.usage import UsageRecorder
  from nimbus.support.evaluation import load_tickets
  from nimbus.support.graph import build_support_graph
  from nimbus.support.pipeline import build_pipeline
  from nimbus.support.tools import SupportStore

  settings_model = get_chat_model()
  store = SupportStore()
  pipeline = build_pipeline(settings_model)
  graph = build_support_graph(settings_model, store)
  rows = []
  for labelled in load_tickets(PROJECT_DIR / "data" / "eval" / "tickets.json"):
    ticket = labelled.ticket
    match = _re.search(r"NC-\d+", ticket.text)
    order = store.order(match.group(0)) if match else None
    for name, run in (
      ("pipeline", lambda t, r: pipeline.invoke({"ticket": t, "history": []}, {"callbacks": [r]})),
      ("graph", lambda t, r: graph.invoke({"ticket": t, "history": [], "steps": []}, {"callbacks": [r]})),
    ):
      rec = UsageRecorder()
      started = time.perf_counter()
      out = run(ticket.text, rec)
      ms = (time.perf_counter() - started) * 1000
      reply = (out.get("reply") or "").casefold()
      rows.append(
        {
          "system": name,
          "ticket": ticket.id,
          "category_ok": out["triage"].category == labelled.expected["category"],
          "has_order": order is not None and out["triage"].category != "allergy_safety" and ticket.text.isascii(),
          "status_stated": order is not None
          and any(w in reply for w in STATUS_WORDS.get(order["status"], [order["status"]])),
          "ms": ms,
          "calls": len(rec.calls),
        }
      )
  return rows


@lab(
  "agents.support-graph-eval",
  "Evaluate: the support graph vs the LCEL pipeline",
  uses_model=True,
  calls="~80 calls to Mistral (20 tickets × 2 calls × 2 systems): about 3 minutes",
)
def support_graph_eval():
  """The same 20 labelled tickets through the LangChain-track pipeline and through the graph."""
  started = time.perf_counter()
  rows = evaluate_support_graph()
  seconds = time.perf_counter() - started
  out = []
  for name in ("pipeline", "graph"):
    mine = [r for r in rows if r["system"] == name]
    orders = [r for r in mine if r["has_order"]]
    ms = sorted(r["ms"] for r in mine)
    out.append(
      {
        "variant": "LCEL pipeline (LangChain M2)" if name == "pipeline" else "support graph (production)",
        "category": round(sum(r["category_ok"] for r in mine) / len(mine), 3),
        "status": f"{sum(r['status_stated'] for r in orders)}/{len(orders)}",
        "calls": round(sum(r["calls"] for r in mine) / len(mine), 2),
        "p50_ms": round(ms[len(ms) // 2]),
      }
    )
  return [
    scorecard(
      out,
      [
        Metric("category", "category right", "higher", "ratio"),
        Metric(
          "status", "reply states the real order status", "none", "text", "English tickets naming an existing order"
        ),
        Metric("calls", "model calls / ticket", "lower", "text"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"20 labelled tickets · {seconds:.0f} s",
    ),
    callout(
      f"Same model, same prompts, same two calls per ticket: category {out[0]['category']:.2f} vs "
      f"{out[1]['category']:.2f} (run-to-run variation, not the graph). The graph adds an order lookup that runs "
      f"beside triage, so replies can state the real status: {out[1]['status']} did, vs {out[0]['status']} for the "
      f"chain, which never saw the order (median latency {out[0]['p50_ms'] / 1000:.1f} s vs "
      f"{out[1]['p50_ms'] / 1000:.1f} s: the reply prompt carries the order facts). Having the facts doesn't "
      "guarantee the reply uses them; Module 2 adds a check that sends a draft back when it doesn't.",
      "decision",
      "Ship the support graph",
    ),
  ]
