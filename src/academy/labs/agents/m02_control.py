"""Agents with LangGraph · Module 2 labs: conditional edges, loops and the recursion limit, parallel branches and
supersteps, Command, Send, input/output/private state, managed values and runtime info, and Pregel underneath.
Then: support graph v2 against v1."""

import json
import operator
import time
import warnings
from typing import Annotated, Literal, TypedDict

from academy.labs import cache
from academy.labs.runtime import Input, Metric, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model

PRICES = {"Cappuccino": 180, "Oat Latte": 220, "Cold Brew": 200, "Vegan Brownie": 170}


# ── Lab 1: conditional edges ────────────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


class Ticket(TypedDict, total=False):
  text: str
  kind: str
  path: Annotated[list[str], operator.add]


def routing_function():
  from langgraph.graph import END, START, StateGraph

  def classify(state):
    kind = "refund" if "refund" in state["text"].lower() else "order" if "NC-" in state["text"] else "other"
    return {"kind": kind, "path": ["classify"]}

  def route(state) -> str:  # reads the state, returns the NEXT node's name
    return {"refund": "refunds", "order": "orders"}.get(state["kind"], END)  # END: finish early

  graph = StateGraph(Ticket)
  graph.add_node("classify", classify)
  graph.add_node("refunds", lambda s: {"path": ["refunds"]})
  graph.add_node("orders", lambda s: {"path": ["orders"]})
  graph.add_edge(START, "classify")
  graph.add_conditional_edges("classify", route, ["refunds", "orders", END])  # the possible targets
  return graph.compile()


def a_path_map():
  from langgraph.graph import END, START, StateGraph

  def label(state) -> str:
    return "money" if "refund" in state["text"].lower() else "info"  # a label, not a node name

  graph = StateGraph(Ticket)
  graph.add_node("refunds", lambda s: {"path": ["refunds"]})
  graph.add_node("faq", lambda s: {"path": ["faq"]})
  graph.add_conditional_edges(START, label, {"money": "refunds", "info": "faq"})  # labels → nodes, drawable
  graph.add_edge("refunds", END)
  graph.add_edge("faq", END)
  return graph.compile()


def several_branches_at_once():
  from langgraph.graph import END, START, StateGraph

  def fan(state) -> list[str]:
    return ["log_it", "notify"] if "urgent" in state["text"].lower() else ["log_it"]  # a LIST: all run

  graph = StateGraph(Ticket)
  graph.add_node("log_it", lambda s: {"path": ["log_it"]})
  graph.add_node("notify", lambda s: {"path": ["notify"]})
  graph.add_conditional_edges(START, fan, ["log_it", "notify"])
  graph.add_edge("log_it", END)
  graph.add_edge("notify", END)
  return graph.compile()


def llm_driven_branch(model):
  from langgraph.graph import END, START, StateGraph
  from pydantic import BaseModel

  class Route(BaseModel):
    kind: Literal["refund", "order", "other"]

  def classify(state):
    kind = cache.structured(
      model, Route, f"Classify this customer message as refund, order or other:\n{state['text']}"
    ).kind
    return {"kind": kind, "path": [f"classify → {kind}"]}  # the model's output decides the path

  graph = StateGraph(Ticket)
  graph.add_node("classify", classify)
  graph.add_node("refunds", lambda s: {"path": ["refunds"]})
  graph.add_node("orders", lambda s: {"path": ["orders"]})
  graph.add_edge(START, "classify")
  graph.add_conditional_edges(
    "classify", lambda s: {"refund": "refunds", "order": "orders"}.get(s["kind"], END), ["refunds", "orders", END]
  )
  return graph.compile()


@lab(
  "agents.conditional-edges",
  "Conditional edges: the state chooses the next node",
  uses_model=True,
  calls="4 calls to Mistral (cached after the first run)",
)
def conditional_edges():
  """Rule-based routing, a path map, several branches at once, END early, and a model choosing the branch."""
  messages = [
    "I want a refund for my opened bag",
    "Where is NC-1003?",
    "Love your cafés!",
    "URGENT: wrong charge on my card",
  ]
  rule = routing_function()
  mapped = a_path_map()
  multi = several_branches_at_once()
  model = get_chat_model()
  smart = llm_driven_branch(model)
  rows = [
    [
      m,
      " → ".join(rule.invoke({"text": m, "path": []})["path"]),
      " → ".join(mapped.invoke({"text": m, "path": []})["path"]),
      ", ".join(sorted(multi.invoke({"text": m, "path": []})["path"])),
      " → ".join(smart.invoke({"text": m, "path": []})["path"]),
    ]
    for m in messages
  ]
  return [
    table(["Message", "rules (END for other)", "path map", "list of branches", "model decides"], rows),
    code(rule.get_graph().draw_mermaid(), "text", title="Rule-based routing: dashed edges are conditional"),
    callout(
      "A conditional edge is a function from state to the next node's name (or END to stop). A path map turns "
      "labels into node names and lets LangGraph draw every possible branch; returning a list runs several "
      "branches at once. Rules are free and predictable; a model decides better on messy text ('wrong charge' is "
      "money, but has no 'refund' in it) at a call per decision.",
      "decision",
      "Conditional edges",
    ),
  ] + [
    example(
      "lc.l49.1",
      "add_conditional_edges",
      routing_function,
      "",
      [table(["message", "path"], [[r[0], r[1]] for r in rows])],
    ),
    example(
      "lc.l49.2",
      "Loops: a conditional edge back",
      routing_function,
      "Lab 2 builds a loop; the support graph's check_reply loops back to draft_reply.",
      [code(rule.get_graph().draw_mermaid(), "text")],
    ),
    example(
      "lc.l49.3",
      "The model's output decides the path",
      llm_driven_branch,
      "",
      [table(["message", "path"], [[r[0], r[4]] for r in rows])],
    ),
    example(
      "lg.l07.1",
      "A routing function returns a node name",
      routing_function,
      "",
      [table(["message", "path"], [[r[0], r[1]] for r in rows])],
    ),
    example(
      "lg.l07.2",
      "A path map: labels → nodes",
      a_path_map,
      "",
      [table(["message", "path"], [[r[0], r[2]] for r in rows])],
    ),
    example(
      "lg.l07.3",
      "A list runs several branches",
      several_branches_at_once,
      "",
      [table(["message", "branches"], [[r[0], r[3]] for r in rows])],
    ),
    example(
      "lg.l07.4",
      "END finishes early",
      routing_function,
      "'Love your cafés!' is neither: the router returns END after classify.",
      [table(["message", "path"], [[rows[2][0], rows[2][1]]])],
    ),
  ]


# ── Lab 2: loops and the recursion limit ────────────────────────────────────────
class Brew(TypedDict):
  cups: int
  target: int
  attempts: int


def a_loop_with_an_exit(max_attempts):
  from langgraph.graph import END, START, StateGraph

  def brew(state):
    return {"cups": state["cups"] + 1, "attempts": state["attempts"] + 1}

  def again(state) -> str:
    if state["cups"] >= state["target"]:
      return "done"
    return "stop" if state["attempts"] >= max_attempts else "brew"  # an exit even when the goal isn't reached

  graph = StateGraph(Brew).add_node("brew", brew).add_edge(START, "brew")
  graph.add_conditional_edges("brew", again, {"brew": "brew", "done": END, "stop": END})
  return graph.compile()


def a_loop_without_an_exit():
  from langgraph.graph import START, StateGraph

  graph = StateGraph(Brew).add_node("brew", lambda s: {"cups": s["cups"] + 1}).add_edge(START, "brew")
  graph.add_conditional_edges("brew", lambda s: "brew", ["brew"])  # never stops on its own
  return graph.compile()


def recursion_limit(graph, limit):
  from langgraph.errors import GraphRecursionError

  try:
    graph.invoke({"cups": 0, "target": 5, "attempts": 0}, {"recursion_limit": limit})  # per run
    return None
  except GraphRecursionError as error:  # handle it: report, fall back, or ask a person
    return str(error).splitlines()[0]


def the_default_limit():
  from langgraph._internal._config import DEFAULT_RECURSION_LIMIT

  return DEFAULT_RECURSION_LIMIT  # 10,007 in LangGraph 1.x (older versions and many tutorials: 25)


@lab(
  "agents.loops",
  "Loops and the recursion limit",
  inputs=[Input("limit", "recursion_limit", "number", default=25, min=3, max=200)],
)
def loops(limit: int):
  """A loop with an exit, one without, and what the recursion limit does about it."""
  bounded = a_loop_with_an_exit(3)
  done = bounded.invoke({"cups": 0, "target": 2, "attempts": 0})
  stopped = bounded.invoke({"cups": 0, "target": 50, "attempts": 0})
  started = time.perf_counter()
  error = recursion_limit(a_loop_without_an_exit(), limit)
  ms = (time.perf_counter() - started) * 1000
  default = the_default_limit()
  return [
    table(
      ["Run", "Result"],
      [
        ["target 2, at most 3 attempts", json.dumps(done)],
        ["target 50, at most 3 attempts", json.dumps(stopped)],
        [f"no exit, recursion_limit={limit}", f"{error} ({ms:.0f} ms)"],
        ["the default limit", f"{default:,}"],
      ],
    ),
    callout(
      f"A loop is a conditional edge pointing back. Give it an exit in the state (count attempts): the bounded loop "
      f"stopped after 3 attempts at {stopped['cups']} cups. Without one, the recursion limit is the last line of "
      f"defence, and LangGraph 1.x's default is {default:,} steps, not the 25 older tutorials mention: a runaway "
      "agent loop would make thousands of model calls first. Set recursion_limit per run and handle "
      "GraphRecursionError.",
      "decision",
      "Loops",
    ),
  ] + [
    example(
      "lg.l08.1", "A loop is an edge back", a_loop_with_an_exit, "", [code(bounded.get_graph().draw_mermaid(), "text")]
    ),
    example(
      "lg.l08.2",
      "An exit condition in the state",
      a_loop_with_an_exit,
      "",
      [table(["target", "result"], [[2, json.dumps(done)], [50, json.dumps(stopped)]])],
    ),
    example(
      "lg.l08.3",
      "The recursion limit, and its real default",
      the_default_limit,
      "",
      [table(["DEFAULT_RECURSION_LIMIT"], [[f"{default:,}"]])],
    ),
    example(
      "lg.l08.4",
      "recursion_limit per run, GraphRecursionError handled",
      recursion_limit,
      "",
      [table(["limit", "error"], [[limit, error]])],
    ),
  ]


# ── Lab 3: parallel branches and supersteps ─────────────────────────────────────
class Work(TypedDict):
  done: Annotated[list[str], operator.add]


def fan_out_fan_in(delay):
  from langgraph.graph import END, START, StateGraph

  def task(name):
    def run(state):
      time.sleep(delay)
      return {"done": [name]}

    return run

  graph = StateGraph(Work)
  for name in ("stock", "price", "loyalty"):
    graph.add_node(name, task(name))
    graph.add_edge(START, name)  # fan out: all three in the first superstep
  graph.add_node("join", lambda s: {"done": [f"join saw {sorted(s['done'])}"]})
  graph.add_edge(["stock", "price", "loyalty"], "join")  # fan in: waits for all three
  graph.add_edge("join", END)
  return graph.compile()


def supersteps(graph):
  steps = []
  for event in graph.stream({"done": []}, stream_mode="debug"):  # debug events carry the superstep number
    if event["type"] == "task":
      steps.append((event["step"], event["payload"]["name"]))
  return steps  # nodes ready together run together; their writes apply together


def uneven_branches():
  from langgraph.graph import END, START, StateGraph

  graph = StateGraph(Work)
  graph.add_node("quick", lambda s: {"done": ["quick"]})
  graph.add_node("slow_1", lambda s: {"done": ["slow_1"]})
  graph.add_node("slow_2", lambda s: {"done": ["slow_2"]})
  graph.add_node("join", lambda s: {"done": [f"join after {len(s['done'])} writes"]})
  graph.add_edge(START, "quick")
  graph.add_edge(START, "slow_1")
  graph.add_edge("slow_1", "slow_2")
  graph.add_edge(["quick", "slow_2"], "join")  # waits for BOTH incoming edges, however long each branch is
  graph.add_edge("join", END)
  return graph.compile()


def deferred_node():
  from langgraph.graph import END, START, StateGraph

  graph = StateGraph(Work)
  graph.add_node("a", lambda s: {"done": ["a"]})
  graph.add_node("b1", lambda s: {"done": ["b1"]})
  graph.add_node("b2", lambda s: {"done": ["b2"]})
  graph.add_node("summary", lambda s: {"done": [f"summary of {len(s['done'])}"]}, defer=True)  # runs last, once
  graph.add_edge(START, "a")
  graph.add_edge(START, "b1")
  graph.add_edge("b1", "b2")
  graph.add_edge("a", "summary")
  graph.add_edge("b2", "summary")
  graph.add_edge("summary", END)
  return graph.compile()


@lab(
  "agents.parallel",
  "Parallel branches and supersteps",
  inputs=[Input("delay", "Each task takes (seconds)", "number", default=0.3, min=0.1, max=1.0)],
)
def parallel(delay: float):
  """Fan-out/fan-in timed, the supersteps shown, branches of different lengths, and a deferred node."""
  graph = fan_out_fan_in(delay)
  started = time.perf_counter()
  result = graph.invoke({"done": []})
  ms = (time.perf_counter() - started) * 1000
  steps = supersteps(graph)
  uneven = uneven_branches().invoke({"done": []})
  uneven_steps = supersteps(uneven_branches())
  deferred = deferred_node().invoke({"done": []})
  return [
    table(["superstep", "node"], steps, title=f"Three {delay:.1f} s tasks: {ms:.0f} ms in total"),
    table(
      ["Graph", "Result"],
      [
        ["fan-out → join", json.dumps(result["done"])],
        ["uneven branches", json.dumps(uneven["done"])],
        ["deferred summary", json.dumps(deferred["done"])],
      ],
    ),
    callout(
      f"Nodes that become ready together run together in one superstep: three {delay:.1f} s tasks took about one "
      "task's time. A join waits for every incoming edge, however long each branch is. defer=True runs a node only "
      "after everything else has finished, once: the place for a summary of all branches.",
      "decision",
      "Parallelism",
    ),
  ] + [
    example(
      "lg.l09.1",
      "Fan-out / fan-in",
      fan_out_fan_in,
      "",
      [table(["result", "ms"], [[json.dumps(result["done"]), round(ms)]])],
    ),
    example("lg.l09.2", "Supersteps", supersteps, "", [table(["superstep", "node"], steps)]),
    example(
      "lg.l09.3",
      "Branches of different lengths",
      uneven_branches,
      "",
      [table(["superstep", "node"], uneven_steps), table(["result"], [[json.dumps(uneven["done"])]])],
    ),
    example(
      "lg.l09.4", "defer=True: run once, last", deferred_node, "", [table(["result"], [[json.dumps(deferred["done"])]])]
    ),
  ]


# ── Lab 4: Command ──────────────────────────────────────────────────────────────
class Order(TypedDict, total=False):
  item: str
  quantity: int
  total: int
  path: Annotated[list[str], operator.add]


def command_from_a_node():
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Command

  def check_stock(state) -> Command[Literal["price", "sold_out"]]:  # destinations from the return type
    in_stock = state["item"] != "Cold Brew"
    return Command(
      update={"path": [f"check_stock: {'in stock' if in_stock else 'sold out'}"]},
      goto="price" if in_stock else "sold_out",
    )

  graph = StateGraph(Order)
  graph.add_node("check_stock", check_stock)
  graph.add_node("price", lambda s: {"total": PRICES[s["item"]] * s["quantity"], "path": ["price"]})
  graph.add_node("sold_out", lambda s: {"path": ["sold_out"]})
  graph.add_edge(START, "check_stock")
  graph.add_edge("price", END)
  graph.add_edge("sold_out", END)
  return graph.compile()


def destinations_argument():
  from langgraph.graph import START, StateGraph
  from langgraph.types import Command

  def check(state):  # no type hint on the return…
    return Command(goto="price")

  graph = StateGraph(Order)
  graph.add_node("check", check, destinations=("price", "sold_out"))  # …so tell LangGraph where it may go
  graph.add_node("price", lambda s: {"path": ["price"]})
  graph.add_node("sold_out", lambda s: {"path": ["sold_out"]})
  graph.add_edge(START, "check")
  return graph.compile()


@lab("agents.command", "Command: update the state and choose the next node, together")
def command_lab():
  """A node that returns Command, and the destinations LangGraph needs to draw it."""
  graph = command_from_a_node()
  runs = {item: graph.invoke({"item": item, "quantity": 2, "path": []}) for item in ("Oat Latte", "Cold Brew")}
  dest = destinations_argument()
  return [
    table(["Item", "path", "total"], [[i, " → ".join(r["path"]), r.get("total", "—")] for i, r in runs.items()]),
    code(dest.get_graph().draw_mermaid(), "text", title="Drawn from destinations=: both possible targets"),
    callout(
      "Command(update=…, goto=…) is a node deciding where to go while it updates the state: one place for logic that "
      "would otherwise be split between the node and a routing function. Use a conditional edge when the routing "
      "is a separate decision (the same router after several nodes); use Command when the node that does the work "
      "knows where to go next (the support graph's check_reply). Declare destinations so the graph can be drawn and "
      "checked.",
      "decision",
      "Command vs conditional edges",
    ),
  ] + [
    example(
      "lg.l10.1",
      "Command(update=…, goto=…)",
      command_from_a_node,
      "",
      [table(["item", "path"], [[i, " → ".join(r["path"])] for i, r in runs.items()])],
    ),
    example("lg.l10.2", "destinations=", destinations_argument, "", [code(dest.get_graph().draw_mermaid(), "text")]),
    example(
      "lg.l10.3",
      "When to use which",
      command_from_a_node,
      "",
      [
        table(
          ["Use", "When"],
          [
            ["conditional edge", "routing is its own decision, reused after several nodes"],
            ["Command", "the node doing the work knows the next step (and may update state with it)"],
          ],
        )
      ],
    ),
  ]


# ── Lab 5: Send ─────────────────────────────────────────────────────────────────
class Basket(TypedDict):
  items: list[str]
  lines: Annotated[list[str], operator.add]  # the "reduce": every task's result is appended


class Line(TypedDict):
  item: str  # what ONE mapped task receives


def map_reduce():
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Send

  def fan_out(state) -> list:
    return [Send("price_line", {"item": item}) for item in state["items"]]  # one task per item, made at run time

  def price_line(state: Line) -> dict:  # its own input, not the whole basket
    return {"lines": [f"{state['item']}: {PRICES[state['item']]}"]}

  def total(state) -> dict:
    return {"lines": [f"TOTAL: {sum(int(x.split(': ')[1]) for x in state['lines'])}"]}

  graph = StateGraph(Basket)
  graph.add_node("price_line", price_line)
  graph.add_node("total", total)
  graph.add_conditional_edges(START, fan_out, ["price_line"])
  graph.add_edge("price_line", "total")  # runs once, after ALL the tasks
  graph.add_edge("total", END)
  return graph.compile()


def send_to_different_nodes():
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Send

  def by_kind(state) -> list:
    return [Send("bake" if item == "Vegan Brownie" else "brew", {"item": item}) for item in state["items"]]

  graph = StateGraph(Basket)
  graph.add_node("brew", lambda s: {"lines": [f"brewed {s['item']}"]})
  graph.add_node("bake", lambda s: {"lines": [f"baked {s['item']}"]})
  graph.add_conditional_edges(START, by_kind, ["brew", "bake"])
  graph.add_edge("brew", END)
  graph.add_edge("bake", END)
  return graph.compile()


@lab(
  "agents.send",
  "Send: as many parallel tasks as the data needs (map-reduce)",
  inputs=[Input("cappuccinos", "Extra cappuccinos", "number", default=2, min=0, max=8)],
)
def send_lab(cappuccinos: int):
  """A basket whose size is only known at run time, priced line by line in parallel, then totalled."""
  graph = map_reduce()
  baskets = [["Cappuccino", "Vegan Brownie"], ["Oat Latte", "Cold Brew", *["Cappuccino"] * cappuccinos]]
  results = [graph.invoke({"items": b, "lines": []})["lines"] for b in baskets]
  routed = send_to_different_nodes().invoke({"items": ["Cappuccino", "Vegan Brownie", "Cold Brew"], "lines": []})[
    "lines"
  ]
  return [
    table(
      ["Basket", "tasks", "lines"],
      [[", ".join(b), len(b), " | ".join(r)] for b, r in zip(baskets, results, strict=True)],
    ),
    table(["Send to different nodes"], [[" | ".join(routed)]]),
    callout(
      "Fixed edges need the branches known when you write the graph. Send creates them at run time: one task per "
      "item, each with its own small input, all in the same superstep; a reducer collects their results. The "
      "support graph uses it to look up every order a ticket mentions at once.",
      "decision",
      "Send",
    ),
  ] + [
    example(
      "lg.l11.1",
      "Send creates branches at run time",
      map_reduce,
      "",
      [table(["items", "tasks"], [[len(b), len(b)] for b in baskets])],
    ),
    example(
      "lg.l11.2",
      "Each Send carries its own input",
      map_reduce,
      "`price_line` reads `Line` (one item), not the whole basket.",
      [table(["lines"], [[" | ".join(results[0])]])],
    ),
    example(
      "lg.l11.3",
      "A reducer collects the results",
      map_reduce,
      "`lines` has `operator.add`, so every task's line is kept, then `total` adds them up.",
      [table(["result"], [[" | ".join(results[1])]])],
    ),
    example(
      "lg.l11.4", "Send to different nodes", send_to_different_nodes, "", [table(["lines"], [[" | ".join(routed)]])]
    ),
  ]


# ── Lab 6: input, output and private state ──────────────────────────────────────
COST_TO_MAKE = {"Cappuccino": 45, "Oat Latte": 70}  # internal: customers must never see it


def three_views_of_one_graph():
  from langgraph.graph import END, START, StateGraph

  class InputState(TypedDict):
    item: str
    quantity: int

  class OutputState(TypedDict):
    price: int

  class OverallState(InputState, OutputState):
    margin: int  # internal, never returned

  class PrivateState(TypedDict):
    unit_cost: int  # only passed between two nodes

  class ComputeInput(InputState, PrivateState):
    pass

  def look_up_cost(state: InputState) -> PrivateState:
    return {"unit_cost": COST_TO_MAKE[state["item"]]}

  def compute(state: ComputeInput) -> dict:  # its type hint decides which keys it reads, private ones included
    price = PRICES[state["item"]] * state["quantity"]
    return {"price": price, "margin": price - state["unit_cost"] * state["quantity"]}

  graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
  graph.add_node("look_up_cost", look_up_cost)
  graph.add_node("compute", compute)
  graph.add_edge(START, "look_up_cost")
  graph.add_edge("look_up_cost", "compute")
  graph.add_edge("compute", END)
  return graph.compile()


@lab("agents.io-schemas", "Input, output and private state: what callers see vs what nodes see")
def io_schemas():
  """Callers send the order and get only the price; the cost and margin exist only inside the run."""
  graph = three_views_of_one_graph()
  result = graph.invoke({"item": "Oat Latte", "quantity": 2})
  updates = list(graph.stream({"item": "Cappuccino", "quantity": 3}, stream_mode="updates"))
  inputs = list(graph.get_input_jsonschema()["properties"])
  outputs = list(graph.get_output_jsonschema()["properties"])
  return [
    table(
      ["View", "Keys"],
      [
        ["input schema (callers send)", ", ".join(inputs)],
        ["output schema (callers get)", ", ".join(outputs)],
        ["returned", json.dumps(result)],
        ["inside the run", json.dumps(updates)],
      ],
    ),
    callout(
      "input_schema and output_schema give a graph a public contract: callers can't pass internal keys and never "
      "see them in the result (the margin and unit cost here). Private keys pass between nodes that declare them in "
      "their type hints. The support graph v2 returns only triage, reply, needs_review and steps.",
      "decision",
      "Schemas",
    ),
  ] + [
    example("lg.l12.1", "input_schema", three_views_of_one_graph, "", [table(["input keys"], [[", ".join(inputs)]])]),
    example("lg.l12.2", "output_schema", three_views_of_one_graph, "", [table(["returned"], [[json.dumps(result)]])]),
    example(
      "lg.l12.3",
      "Private state between nodes",
      three_views_of_one_graph,
      "",
      [code(json.dumps(updates, indent=2), "json", "updates inside the run")],
    ),
    example(
      "lg.l12.4",
      "A node reads the schema in its type hint",
      three_views_of_one_graph,
      "`compute(state: ComputeInput)` sees `unit_cost`; the caller never does.",
      [
        table(
          ["node", "reads"], [["look_up_cost", "InputState"], ["compute", "ComputeInput = InputState + PrivateState"]]
        )
      ],
    ),
  ]


# ── Lab 7: managed values and runtime info ──────────────────────────────────────
class BrewState(TypedDict):
  cups: int
  target: int
  note: str


def remaining_steps_graph():
  from langgraph.graph import END, START, StateGraph
  from langgraph.managed import RemainingSteps

  class State(BrewState):
    remaining_steps: RemainingSteps  # filled in by LangGraph on every step: you never set it

  def keep_going(state) -> str:
    if state["cups"] >= state["target"]:
      return "done"
    return "wrap_up" if state["remaining_steps"] <= 2 else "brew"  # finish gracefully before the limit

  graph = StateGraph(State)
  graph.add_node("brew", lambda s: {"cups": s["cups"] + 1})
  graph.add_node("wrap_up", lambda s: {"note": f"stopped at {s['cups']} of {s['target']} (step budget)"})
  graph.add_edge(START, "brew")
  graph.add_conditional_edges("brew", keep_going, {"brew": "brew", "wrap_up": "wrap_up", "done": END})
  graph.add_edge("wrap_up", END)
  return graph.compile()


def execution_info_and_helpers():
  import uuid
  from dataclasses import dataclass

  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.config import get_config
  from langgraph.graph import START, StateGraph
  from langgraph.runtime import Runtime, get_runtime
  from langgraph.types import RetryPolicy

  @dataclass
  class Context:
    agent: str

  class State(TypedDict):
    order_id: str
    log: str

  def audit_line(event):  # a helper deep inside a node: no config or runtime passed down, yet it can read them
    return f"{event} by {get_runtime(Context).context.agent} on thread {get_config()['configurable']['thread_id']}"

  def charge(state, runtime: Runtime[Context]):
    info = runtime.execution_info  # thread, run, checkpoint, and which ATTEMPT this is
    if info.node_attempt == 1:
      raise ConnectionError("payment gateway timeout")  # the retry policy runs it again
    return {"log": audit_line(f"charged {state['order_id']} on attempt {info.node_attempt}")}

  graph = StateGraph(State, context_schema=Context)
  graph.add_node("charge", charge, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05, jitter=False))
  graph.add_edge(START, "charge")
  compiled = graph.compile(checkpointer=InMemorySaver())
  config = {"configurable": {"thread_id": "orders-7"}, "run_id": uuid.uuid4()}
  return compiled.invoke({"order_id": "NC-1004", "log": ""}, config, context=Context("support-bot"))


def untracked_value():
  from typing import Annotated

  from langgraph.channels import UntrackedValue
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph

  class State(TypedDict):
    summary: str
    connection: Annotated[object, UntrackedValue(object)]  # exists during the run, never saved

  class FakeConnection:
    def query(self, sql):
      return 128

  graph = StateGraph(State).add_sequence(
    [
      ("connect", lambda s: {"connection": FakeConnection()}),
      ("report", lambda s: {"summary": f"{s['connection'].query('select')} orders"}),
    ]
  )
  graph.add_edge(START, "connect")
  compiled = graph.compile(checkpointer=InMemorySaver())
  config = {"configurable": {"thread_id": "report-1"}}
  result = compiled.invoke({"summary": ""}, config)
  return "connection" in result, "connection" in compiled.get_state(config).values


@lab("agents.runtime-info", "Managed values and runtime info: what LangGraph tells a node about the run")
def runtime_info():
  """RemainingSteps for a graceful stop, execution info with retries, helpers that read config, and untracked state."""
  graph = remaining_steps_graph()
  within = graph.invoke({"cups": 0, "target": 5, "note": ""}, {"recursion_limit": 25})
  graceful = graph.invoke({"cups": 0, "target": 50, "note": ""}, {"recursion_limit": 10})
  charged = execution_info_and_helpers()
  in_result, in_checkpoint = untracked_value()
  return [
    table(
      ["Run", "Result"],
      [
        ["target 5, limit 25", json.dumps(within)],
        ["target 50, limit 10", json.dumps(graceful)],
        ["retry + execution_info + helpers", charged["log"]],
        ["UntrackedValue", f"in the result: {in_result}; in the checkpoint: {in_checkpoint}"],
      ],
    ),
    callout(
      "RemainingSteps lets a graph wrap up before the recursion limit instead of losing the run to an error. "
      "execution_info tells a node which attempt it is (retries), and get_config() / get_runtime() reach the same "
      "facts from helpers deep inside a node. UntrackedValue holds things that can't or mustn't be saved, like a "
      "connection. The support graph v2 uses RemainingSteps to cap its redraft loop.",
      "decision",
      "Runtime info",
    ),
  ] + [
    example(
      "lg.l13.1",
      "RemainingSteps: finish before the limit",
      remaining_steps_graph,
      "",
      [table(["run", "result"], [["limit 10", json.dumps(graceful)]])],
    ),
    example(
      "lg.l13.2",
      "runtime.execution_info and retries",
      execution_info_and_helpers,
      "",
      [table(["log"], [[charged["log"]]])],
    ),
    example(
      "lg.l13.3",
      "get_config() / get_runtime() in helpers",
      execution_info_and_helpers,
      "",
      [table(["log"], [[charged["log"]]])],
    ),
    example(
      "lg.l13.4",
      "UntrackedValue: never checkpointed",
      untracked_value,
      "",
      [table(["in the result", "in the checkpoint"], [[in_result, in_checkpoint]])],
    ),
  ]


# ── Lab 8: Pregel and channels ──────────────────────────────────────────────────
def compiled_is_pregel():
  from langgraph.graph import START, StateGraph
  from langgraph.pregel import Pregel

  class State(TypedDict):
    counter: int
    seen: Annotated[list[str], operator.add]

  graph = StateGraph(State)
  graph.add_node("bump", lambda s: {"counter": s["counter"] + 1, "seen": [f"bump saw {s['counter']}"]})
  graph.add_node("peek", lambda s: {"seen": [f"peek saw {s['counter']}"]})
  graph.add_edge(START, "bump")
  graph.add_edge(START, "peek")
  graph.add_edge("bump", "peek")
  app = graph.compile()
  channels = {k: type(c).__name__ for k, c in app.channels.items() if not k.startswith(("__", "branch"))}
  return app, isinstance(app, Pregel), channels


def writes_visible_next_step(app):
  seen = []
  for event in app.stream({"counter": 0, "seen": []}, stream_mode="debug"):
    if event["type"] == "task_result":
      seen.append((event["step"], event["payload"]["name"], json.dumps(event["payload"]["result"])))
  return seen  # peek in step 1 still sees counter=0 although bump wrote 1 in the same step


def pregel_by_hand():
  from langgraph.channels import EphemeralValue, LastValue, Topic
  from langgraph.pregel import NodeBuilder, Pregel

  shout = NodeBuilder().subscribe_only("text").do(lambda t: t.upper()).write_to("loud", "history")
  excite = NodeBuilder().subscribe_only("loud").do(lambda t: t + "!!!").write_to("history")
  app = Pregel(
    nodes={"shout": shout, "excite": excite},
    channels={"text": EphemeralValue(str), "loud": LastValue(str), "history": Topic(str, accumulate=True)},
    input_channels=["text"],
    output_channels=["loud", "history"],
  )
  return app.invoke({"text": "fresh coffee"})


def channel_types():
  from langgraph.channels import UntrackedValue
  from langgraph.graph import START, StateGraph

  class Kinds(TypedDict):
    plain: str  # no reducer → LastValue
    total: Annotated[int, operator.add]  # a reducer → BinaryOperatorAggregate
    secret: Annotated[object, UntrackedValue(object)]

  app = StateGraph(Kinds).add_node("noop", lambda s: {}).add_edge(START, "noop").compile()
  return {k: type(c).__name__ for k, c in app.channels.items() if not k.startswith(("__", "branch"))}


def delta_channel_sizes(steps, folder):
  from langgraph.channels import DeltaChannel
  from langgraph.checkpoint.sqlite import SqliteSaver
  from langgraph.graph import END, START, StateGraph

  def append_all(current, writes):  # a bulk reducer: every write of a step at once; must be associative
    out = list(current)
    for w in writes:
      out.extend(w)
    return out

  class Plain(TypedDict):
    n: int
    log: Annotated[list[str], operator.add]

  class Delta(TypedDict):
    n: int
    log: Annotated[list[str], DeltaChannel(append_all)]  # beta: stores only what changed each step

  sizes = {}
  for schema in (Plain, Delta):
    path = folder / f"{schema.__name__}.sqlite"
    path.unlink(missing_ok=True)
    builder = StateGraph(schema).add_node("step", lambda s: {"n": s["n"] + 1, "log": ["x" * 200]})
    builder.add_edge(START, "step").add_conditional_edges(
      "step", lambda s, steps=steps: END if s["n"] >= steps else "step"
    )
    with SqliteSaver.from_conn_string(str(path)) as saver:
      graph = builder.compile(checkpointer=saver)
      graph.invoke({"n": 0, "log": []}, {"configurable": {"thread_id": "t"}}, durability="sync")  # sync: see note
    sizes[schema.__name__] = path.stat().st_size
    path.unlink()
  return sizes


@lab(
  "agents.pregel",
  "Under the hood: Pregel, channels and supersteps",
  inputs=[Input("steps", "Loop steps (DeltaChannel test)", "number", default=40, min=10, max=80)],
)
def pregel(steps: int):
  """What compile() builds, writes visible only in the next superstep, a Pregel app by hand, channel types, and
  DeltaChannel's checkpoint sizes."""
  app, is_pregel, channels = compiled_is_pregel()
  seen = writes_visible_next_step(app)
  by_hand = pregel_by_hand()
  kinds = channel_types()
  folder = PROJECT_DIR / ".cache" / "labs" / "agents"
  folder.mkdir(parents=True, exist_ok=True)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # DeltaChannel is beta and says so
    sizes = delta_channel_sizes(steps, folder)
  return [
    table(["superstep", "node", "wrote"], seen, title="peek runs twice: in step 1 it can't see bump's write yet"),
    table(
      ["Thing", "Value"],
      [
        ["compile() returns a Pregel app", is_pregel],
        ["channels", json.dumps(channels)],
        ["a Pregel app by hand", json.dumps(by_hand)],
        ["channel per field kind", json.dumps(kinds)],
      ],
    ),
    scorecard(
      [
        {"variant": "operator.add (LastValue of the whole list)", "kb": round(sizes["Plain"] / 1024)},
        {"variant": "DeltaChannel (only each step's new items)", "kb": round(sizes["Delta"] / 1024)},
      ],
      [Metric("kb", "checkpoint file (KB)", "lower", "int")],
      title=f"{steps} steps, each appending 200 characters, checkpointed to SQLite",
    ),
    callout(
      "StateGraph compiles to Pregel: every state key is a channel, every node an actor that reads and writes "
      "channels, run in supersteps (plan, run the ready nodes in parallel, apply their writes). A write isn't visible "
      "until the next step, which is why parallel nodes never see each other's updates. DeltaChannel (beta) "
      "checkpoints "
      f"only what changed: {sizes['Plain'] // 1024} KB → {sizes['Delta'] // 1024} KB here. Run it with "
      "durability='sync' on SQLite: in the default async mode, long loops deadlocked when this course tested it.",
      "decision",
      "Pregel",
    ),
  ] + [
    example(
      "lg.l14.1",
      "compile() builds a Pregel app of channels",
      compiled_is_pregel,
      "",
      [table(["channel", "type"], list(channels.items()))],
    ),
    example(
      "lg.l14.2",
      "Supersteps: writes visible in the next step",
      writes_visible_next_step,
      "",
      [table(["step", "node", "wrote"], seen)],
    ),
    example(
      "lg.l14.3", "A Pregel app with NodeBuilder", pregel_by_hand, "", [code(json.dumps(by_hand, indent=2), "json")]
    ),
    example(
      "lg.l14.4",
      "Channel types",
      channel_types,
      "EphemeralValue and Topic appear in the hand-built app above.",
      [table(["field", "channel"], list(kinds.items()))],
    ),
    example(
      "lg.l14.5",
      "DeltaChannel: smaller checkpoints",
      delta_channel_sizes,
      "",
      [table(["schema", "KB"], [[k, round(v / 1024)] for k, v in sizes.items()])],
    ),
  ]


# ── Production path: support graph v2 ──────────────────────────────────────────
def evaluate_v1_vs_v2() -> list[dict]:
  import re

  from nimbus.llm.usage import UsageRecorder
  from nimbus.support.evaluation import load_tickets
  from nimbus.support.graph import STATUS_WORDS, build_support_graph, build_support_graph_v2, reply_problems
  from nimbus.support.tools import SupportStore

  model = get_chat_model()
  store = SupportStore()
  graphs = {"v1": build_support_graph(model, store), "v2": build_support_graph_v2(model, store)}
  rows = []
  for labelled in load_tickets(PROJECT_DIR / "data" / "eval" / "tickets.json"):
    ticket = labelled.ticket
    match = re.search(r"NC-\d+", ticket.text)
    order = store.order(match.group(0)) if match else None
    for name, graph in graphs.items():
      rec = UsageRecorder()
      started = time.perf_counter()
      out = graph.invoke(
        {"ticket": ticket.text, "history": [], **({"steps": []} if name == "v1" else {})}, {"callbacks": [rec]}
      )
      reply = out.get("reply", "")
      orders = [{"record": order}] if order else []
      rows.append(
        {
          "system": name,
          "category_ok": out["triage"].category == labelled.expected["category"],
          "has_order": order is not None and ticket.text.isascii() and out["triage"].category != "allergy_safety",
          "status": order is not None
          and any(w in reply.casefold() for w in STATUS_WORDS.get(order["status"], [order["status"]])),
          "problems": len(reply_problems(reply, out["triage"], orders))
          if out["triage"].category != "allergy_safety"
          else 0,
          "redrafts": sum("check_reply:" in s and "ok" not in s and "still" not in s for s in out.get("steps", [])),
          "review": bool(out.get("needs_review")),
          "calls": len(rec.calls),
          "ms": (time.perf_counter() - started) * 1000,
        }
      )
  return rows


@lab(
  "agents.support-v2-eval",
  "Evaluate: support graph v2 vs v1",
  uses_model=True,
  calls="~85 calls to Mistral (20 tickets, two graphs, plus redrafts): about 4 minutes",
)
def support_v2_eval():
  """The 20 labelled tickets through v1 (Module 1) and v2 (routing, Send, a checked loop)."""
  started = time.perf_counter()
  rows = evaluate_v1_vs_v2()
  seconds = time.perf_counter() - started
  out = []
  for name in ("v1", "v2"):
    mine = [r for r in rows if r["system"] == name]
    orders = [r for r in mine if r["has_order"]]
    ms = sorted(r["ms"] for r in mine)
    out.append(
      {
        "variant": "v1 (Module 1)" if name == "v1" else "v2 (production)",
        "category": round(sum(r["category_ok"] for r in mine) / len(mine), 3),
        "status": f"{sum(r['status'] for r in orders)}/{len(orders)}",
        "with_problems": sum(r["problems"] > 0 for r in mine),
        "redrafts": sum(r["redrafts"] for r in mine),
        "review": sum(r["review"] for r in mine),
        "calls": round(sum(r["calls"] for r in mine) / len(mine), 2),
        "p50_ms": round(ms[len(ms) // 2]),
      }
    )
  return [
    scorecard(
      out,
      [
        Metric("category", "category right", "higher", "ratio"),
        Metric("status", "states the real order status", "none", "text"),
        Metric(
          "with_problems",
          "final replies failing the checks",
          "lower",
          "int",
          "Missing order status, claimed actions, too long",
        ),
        Metric("redrafts", "drafts sent back", "none", "int"),
        Metric(
          "review", "flagged for a person", "none", "int", "Safety tickets, and replies still failing after 2 drafts"
        ),
        Metric("calls", "calls / ticket", "lower", "text"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"20 labelled tickets · {seconds:.0f} s",
    ),
    callout(
      f"v2 checks every draft in code (the real order status, no claimed cancellations or refunds, a sane length) "
      f"and sends failing ones back: {out[1]['redrafts']} redraft(s). Final replies failing the checks: "
      f"{out[0]['with_problems']} → {out[1]['with_problems']}; order status stated {out[0]['status']} → "
      f"{out[1]['status']}, for {out[1]['calls'] - out[0]['calls']:.2f} more calls per ticket on average. Replies "
      "that still fail after two drafts, and every safety ticket, are flagged for a person instead of sent as if "
      "they were fine.",
      "decision",
      "Ship support graph v2",
    ),
  ]
