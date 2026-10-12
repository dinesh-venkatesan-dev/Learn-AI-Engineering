"""Agents with LangGraph · Module 4 labs: tool-calling agents built from scratch, ToolNode in depth, an agent inside a
workflow with business state, structured endings, and streaming (agent streams, graph streams, event streaming).
Then: the support desk (agent in a workflow) against support agent v1 (create_agent)."""

import json
import operator
import statistics
import time
import warnings
from typing import Annotated

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model

warnings.filterwarnings("ignore", message=".*v3 streaming protocol.*")  # v3 event streaming is marked beta


def _store():
  from nimbus.support.tools import SupportStore

  return SupportStore()


def _lookup_tools(store):
  from nimbus.support.tools import build_tools

  return [t for t in build_tools(store) if t.name != "escalate_to_human"]  # escalation ends the run: not needed here


def _trace(messages) -> list[list]:
  rows = []
  for m in messages:
    if isinstance(m, AIMessage) and m.tool_calls:
      rows.append(["AI", "calls " + ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)])
    elif isinstance(m, ToolMessage):
      rows.append([f"Tool · {m.name}", str(m.content)[:160]])
    else:
      rows.append([type(m).__name__.replace("Message", ""), m.text[:200]])
  return rows


# ── Lab 1: an agent from scratch ───────────────────────────────────────────────
QUESTION = "Where is my order NC-78123, and how many loyalty points will it earn?"
SYSTEM = "You are Nimbus Coffee's support agent. Use the tools; never guess. Reply in two short sentences."


def messages_state():
  from langgraph.graph import MessagesState

  return MessagesState.__annotations__  # {"messages": Annotated[list[AnyMessage], add_messages]}


def hand_built_agent(model, tools):
  from langgraph.graph import END, START, MessagesState, StateGraph

  by_name = {t.name: t for t in tools}
  bound = model.bind_tools(tools)

  def call_model(state: MessagesState) -> dict:  # the "model" node: system prompt + conversation
    return {"messages": [bound.invoke([SystemMessage(SYSTEM), *state["messages"]])]}

  def run_tools(state: MessagesState) -> dict:  # the "tools" node: every call in the last AIMessage
    calls = state["messages"][-1].tool_calls
    return {"messages": [by_name[c["name"]].invoke(c) for c in calls]}  # invoking with the call → a ToolMessage

  def should_continue(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END

  graph = StateGraph(MessagesState)
  graph.add_node("model", call_model)
  graph.add_node("tools", run_tools)
  graph.add_edge(START, "model")
  graph.add_conditional_edges("model", should_continue, ["tools", END])
  graph.add_edge("tools", "model")  # the loop: model → tools → model … → END
  return graph.compile()


def prebuilt_agent(model, tools):
  from langgraph.graph import START, MessagesState, StateGraph
  from langgraph.prebuilt import ToolNode, tools_condition

  bound = model.bind_tools(tools)
  graph = StateGraph(MessagesState)
  graph.add_node("model", lambda state: {"messages": [bound.invoke([SystemMessage(SYSTEM), *state["messages"]])]})
  graph.add_node("tools", ToolNode(tools))  # runs every call, in parallel, errors handled
  graph.add_edge(START, "model")
  graph.add_conditional_edges("model", tools_condition)  # "tools" if there are tool calls, otherwise END
  graph.add_edge("tools", "model")
  return graph.compile()


@lab(
  "agents.agent-from-scratch", "A tool-calling agent, built from scratch", uses_model=True, calls="~9 calls to Mistral"
)
def agent_from_scratch():
  """The same question through a hand-built agent loop, the prebuilt ToolNode + tools_condition, and create_agent."""
  from langchain.agents import create_agent

  model, store = get_chat_model(), _store()
  tools = _lookup_tools(store)
  question = {"messages": [HumanMessage(QUESTION)]}
  runs = {}
  for name, agent in (
    ("hand-built", hand_built_agent(model, tools)),
    ("ToolNode + tools_condition", prebuilt_agent(model, tools)),
    ("create_agent", create_agent(model, tools, system_prompt=SYSTEM)),
  ):
    started = time.perf_counter()
    result = agent.invoke(question)
    runs[name] = (result, (time.perf_counter() - started) * 1000)
  hand = runs["hand-built"][0]
  made = [
    sorted(
      json.dumps(c, sort_keys=True)
      for m in r["messages"]
      if isinstance(m, AIMessage)
      for c in [{"name": t["name"], "args": t["args"]} for t in m.tool_calls]
    )
    for r, _ in runs.values()
  ]
  same_calls = all(m == made[0] for m in made)

  def row(r, ms):
    return [  # noqa: E731
      sum(isinstance(m, AIMessage) for m in r["messages"]),
      sum(len(m.tool_calls) for m in r["messages"] if isinstance(m, AIMessage)),
      f"{ms / 1000:.1f} s",
      r["messages"][-1].text[:160],
    ]

  return [
    table(["Agent", "model calls", "tool calls", "time", "answer"], [[n, *row(r, ms)] for n, (r, ms) in runs.items()]),
    table(
      ["Message", "Content"], _trace(hand["messages"]), title="The hand-built agent's trace: every step is a message"
    ),
    code(hand_built_agent(model, tools).get_graph().draw_mermaid(), "text", title="The loop"),
    callout(
      "An agent is a loop: the model decides, tools run, the model sees their results, until it answers without "
      "calling a tool. Built by hand it's two nodes and one conditional edge; ToolNode and tools_condition are the "
      "same thing prebuilt (plus parallel calls and error handling); create_agent is that graph with middleware "
      "hooks. "
      + (
        "All three made the same tool calls here. "
        if same_calls
        else "They didn't make the same tool calls here (the model isn't deterministic). "
      )
      + "Build it yourself when the loop needs steps around it (lab 3).",
      "decision",
      "Agent = loop",
    ),
  ] + [
    example("lc.l50.1", "MessagesState", messages_state, "", [code(str(messages_state()), "python")]),
    example(
      "lc.l50.2",
      "A model node and a tools node",
      hand_built_agent,
      "",
      [table(["Message", "Content"], _trace(hand["messages"]))],
    ),
    example(
      "lc.l50.3",
      "The loop",
      hand_built_agent,
      "",
      [code(hand_built_agent(model, tools).get_graph().draw_mermaid(), "text")],
    ),
    example(
      "lc.l50.4",
      "ToolNode and tools_condition",
      prebuilt_agent,
      "",
      [table(["model calls", "tool calls", "time", "answer"], [row(*runs["ToolNode + tools_condition"])])],
    ),
    example(
      "lg.l21.1",
      "An agent is a loop",
      hand_built_agent,
      "",
      [table(["step", "message"], [[i, r[0]] for i, r in enumerate(_trace(hand["messages"]))])],
    ),
    example(
      "lg.l21.2",
      "The model node",
      hand_built_agent,
      "The system prompt is added on every call, not stored in state.",
      [answer(hand["messages"][-1].text, "final answer")],
    ),
    example(
      "lg.l21.3",
      "The tools node: ToolNode",
      prebuilt_agent,
      "",
      [table(["Message", "Content"], _trace(runs["ToolNode + tools_condition"][0]["messages"]))],
    ),
    example(
      "lg.l21.4",
      "tools_condition",
      prebuilt_agent,
      "",
      [code(prebuilt_agent(model, tools).get_graph().draw_mermaid(), "text")],
    ),
    example("lg.l21.5", "Reading the trace", _trace, "", [table(["Message", "Content"], _trace(hand["messages"]))]),
  ]


# ── Lab 2: ToolNode in depth ───────────────────────────────────────────────────
def call(name: str, args: dict, call_id: str) -> dict:
  return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def run_tool_node(node, *calls, **state):
  """A ToolNode needs the graph's runtime, so it runs inside a one-node graph; the AIMessage stands in for a model."""
  from langgraph.graph import START, StateGraph

  schema = state.pop("schema", None)
  if schema is None:
    from langgraph.graph import MessagesState as schema
  graph = StateGraph(schema).add_node("tools", node).add_edge(START, "tools").compile()
  return graph.invoke({"messages": [AIMessage("", tool_calls=list(calls))], **state})


@tool
def warehouse_stock(blend: str) -> str:
  """Bags of a blend in the warehouse (a slow API: 1 second)."""
  time.sleep(1)
  return f"{blend}: 12 bags"


def parallel_calls():
  from langgraph.prebuilt import ToolNode

  started = time.perf_counter()
  result = run_tool_node(
    ToolNode([warehouse_stock]),
    call("warehouse_stock", {"blend": "Nilgiri Sunrise"}, "a"),
    call("warehouse_stock", {"blend": "Chikmagalur Classic"}, "b"),
    call("warehouse_stock", {"blend": "Coorg Dawn"}, "c"),
  )
  return result, time.perf_counter() - started  # three 1-second calls, run at the same time


@tool
def cancel_order(order_id: str) -> str:
  """Cancel an order that hasn't been packed."""
  raise RuntimeError(f"{order_id} is already packed")  # the order system refuses


def default_errors():
  from langgraph.prebuilt import ToolNode

  try:  # any exception other than bad arguments crashes the whole graph
    run_tool_node(ToolNode([cancel_order]), call("cancel_order", {"order_id": "NC-12500"}, "x"))
  except RuntimeError as error:
    crashed = f"RuntimeError: {error}"
  bad_args = run_tool_node(ToolNode([cancel_order]), call("cancel_order", {}, "y"))  # caught: the model can fix it
  return crashed, bad_args["messages"][-1].content


def handled_errors():
  from langgraph.prebuilt import ToolNode

  as_message = ToolNode([cancel_order], handle_tool_errors=True)
  friendly = ToolNode([cancel_order], handle_tool_errors=lambda e: f"Can't cancel: {e}. Offer a return instead.")
  return [
    run_tool_node(node, call("cancel_order", {"order_id": "NC-12500"}, "x"))["messages"][-1].content
    for node in (as_message, friendly)
  ]


def state_access():
  from langchain.tools import ToolRuntime
  from langgraph.graph import MessagesState
  from langgraph.prebuilt import ToolNode
  from langgraph.types import Command

  class DeskState(MessagesState):
    customer_id: str  # set by the app from the login, never by the model
    notes: Annotated[list[str], operator.add]

  store = _store()

  @tool
  def my_orders(runtime: ToolRuntime) -> str:
    """The logged-in customer's orders."""
    return ", ".join(o["order_id"] for o in store.orders_of(runtime.state["customer_id"]))

  @tool
  def add_note(text: str, runtime: ToolRuntime) -> Command:
    """Save a note for the support team."""
    return Command(
      update={"notes": [text], "messages": [ToolMessage("Saved.", tool_call_id=runtime.tool_call_id)]}
    )  # a tool can update state, not just return text

  result = run_tool_node(
    ToolNode([my_orders, add_note]),
    call("my_orders", {}, "o"),
    call("add_note", {"text": "asked about their orders"}, "n"),
    schema=DeskState,
    customer_id="c-3301",
    notes=[],
  )
  return result, my_orders.tool_call_schema.model_json_schema()["properties"]  # the model sees no customer argument


@lab(
  "agents.tool-node",
  "ToolNode in depth: parallel calls, errors, state access, state updates",
  calls="No model calls: the tool calls are written by hand",
)
def tool_node():
  """ToolNode run on hand-written tool calls: timing, crashes and handled errors, ToolRuntime and Command."""
  result, seconds = parallel_calls()
  crashed, bad_args = default_errors()
  handled = handled_errors()
  state, visible = state_access()
  return [
    table(
      ["Call", "Result"],
      [[m.tool_call_id, m.content] for m in result["messages"][1:]],
      title=f"Three 1-second calls in one AIMessage: {seconds:.1f} s",
    ),
    table(
      ["handle_tool_errors", "What happens"],
      [
        ["(default)", crashed + " → the graph crashed"],
        ["(default), bad arguments", bad_args[:160]],
        ["True", handled[0]],
        ["a function", handled[1]],
      ],
      title="A tool that raises",
    ),
    table(
      ["What", "Result"],
      [
        ["my_orders read customer_id from state", state["messages"][1].content],
        ["add_note wrote to state", str(state["notes"])],
        ["arguments the model sees for my_orders", str(visible)],
      ],
    ),
    callout(
      f"ToolNode runs the calls of one AIMessage at the same time (three 1-second calls took {seconds:.1f} s). By "
      "default it only catches bad arguments: any other exception crashes the run, so set handle_tool_errors (True, "
      "a message, or a function) and the model gets the error to work with. Tools read state through ToolRuntime "
      "(the customer comes from the login, so the model can't ask for someone else's orders) and change it by "
      "returning a Command.",
      "decision",
      "ToolNode",
    ),
  ] + [
    example(
      "lg.l22.1", "Parallel tool calls", parallel_calls, "", [table(["calls", "seconds"], [[3, f"{seconds:.1f}"]])]
    ),
    example(
      "lg.l22.2",
      "By default, exceptions crash the graph",
      default_errors,
      "",
      [table(["case", "result"], [["RuntimeError", crashed], ["bad arguments", bad_args[:120]]])],
    ),
    example(
      "lg.l22.3",
      "handle_tool_errors",
      handled_errors,
      "",
      [table(["setting", "message"], [["True", handled[0]], ["function", handled[1]]])],
    ),
    example(
      "lg.l22.4",
      "ToolRuntime reads state",
      state_access,
      "",
      [table(["tool", "result"], [["my_orders", state["messages"][1].content], ["schema", str(visible)]])],
    ),
    example("lg.l22.5", "Command from a tool", state_access, "", [table(["notes"], [[n] for n in state["notes"]])]),
  ]


# ── Lab 3: an agent inside a workflow ──────────────────────────────────────────
REQUESTS = [
  ("c-3301", "Pause my subscription for 3 weeks, and what's the status of NC-55022?"),
  ("c-4120", "There were insects in my coffee bag!"),
  ("c-6060", "Bhai mera subscription 20 hafte ke liye rok do"),
]


def business_state():
  from nimbus.support.desk import DeskState

  return {k: str(v) for k, v in DeskState.__annotations__.items()}


def steps_before_the_agent(store, customer_id, text):
  from nimbus.support.desk import customer_profile, reply_language, system_prompt

  profile = customer_profile(store, customer_id)  # plain code: the order system, no model
  language = reply_language(text)  # decided in code, not guessed by the model from the customer's name
  return profile, language, system_prompt(profile, language)


def steps_after_the_agent(result):
  return result["ticket"]  # file_ticket: who, which orders, what was done, does a person follow up (code)


@lab(
  "agents.agent-in-workflow",
  "An agent inside a bigger workflow, with business state",
  uses_model=True,
  calls="~9 calls to Mistral",
)
def agent_in_workflow():
  """The production support desk: code loads the customer and decides the reply language, the agent loop works, code
  files a ticket for the support desk."""
  from nimbus.support.desk import ask, build_support_desk

  model, store = get_chat_model(), _store()
  desk = build_support_desk(model, store)
  results = [(cid, text, ask(desk, store, cid, text)) for cid, text in REQUESTS]
  profile, language, prompt = steps_before_the_agent(store, *REQUESTS[2])
  tickets = [r["ticket"] for *_, r in results]
  return [
    code(desk.get_graph().draw_mermaid(), "text", title="load_customer → agent ⇄ tools → file_ticket"),
    table(
      ["Customer", "Request", "language (code)", "tools", "actions", "needs a person", "reply"],
      [
        [cid, text, r["language"], ", ".join(t.tools_used), "; ".join(t.actions)[:80], t.needs_human, t.reply]
        for (cid, text, r), t in zip(results, tickets, strict=True)
      ],
      title="Three requests through the desk",
    ),
    table(
      ["Ticket field", *[cid for cid, _ in REQUESTS]],
      [
        [k, *[str(getattr(t, k)) for t in tickets]]
        for k in ("order_ids", "actions", "needs_human", "resolved", "flags")
      ],
    ),
    callout(
      "The agent is one part of the workflow. Plain code runs before it (load the customer's profile, decide the reply "
      "language) and after it (file a ticket the support desk can act on, with needs_human decided by code when the "
      "agent escalated or claimed an action that never ran). State carries business fields (customer, language, "
      "actions, outcome, ticket), not just messages.",
      "decision",
      "Agent in a workflow",
    ),
  ] + [
    example(
      "lg.l23.1",
      "State beyond messages",
      business_state,
      "",
      [table(["field", "type"], list(business_state().items()))],
    ),
    example(
      "lg.l23.2",
      "Code steps before the agent",
      steps_before_the_agent,
      "",
      [
        table(["What", "Result"], [["profile", json.dumps(profile)], ["language", language]]),
        code(prompt, "text", "the system prompt, built from the profile"),
      ],
    ),
    example(
      "lg.l23.3",
      "The agent loop in the middle",
      build_support_desk,
      "",
      [table(["Message", "Content"], _trace(results[0][2]["messages"]))],
    ),
    example(
      "lg.l23.4",
      "Steps after the agent: a ticket",
      steps_after_the_agent,
      "",
      [code(tickets[0].model_dump_json(indent=1), "json")],
    ),
  ]


# ── Lab 4: structured endings ──────────────────────────────────────────────────
ENDING_SCENARIOS = ["s02", "s05", "s06", "h01", "h03", "s09"]


def option_a_respond_node():
  from nimbus.support.desk import RESPOND

  return RESPOND  # after the loop, one more structured call: conversation → SupportOutcome


def option_b_schema_as_tool():
  from nimbus.support.desk import final_answer

  return final_answer.args_schema.model_json_schema()  # the agent "answers" by calling it; tool_choice="any"


def _scenarios(ids=None):
  from nimbus.support.agent_eval import load_scenarios

  scenarios = load_scenarios(get_settings().eval_agent_scenarios) + load_scenarios(
    PROJECT_DIR / "data" / "eval" / "agent_scenarios_hard.json"
  )
  return [s for s in scenarios if ids is None or s["id"] in ids]


@lab(
  "agents.structured-ending",
  "Agents that end with structured output",
  uses_model=True,
  calls="~36 calls to Mistral: 6 scenarios × 2 endings",
)
def structured_ending():
  """The support desk with each ending, on six scenarios where the ending matters: actions to take, an escalation,
  a request the tools must refuse."""
  from nimbus.support.agent_eval import run_desk_scenario, score

  model = get_chat_model()
  rows, out = [], []
  for ending in ("respond", "tool"):
    runs = [(s, run_desk_scenario(model, s, ending)) for s in _scenarios(ENDING_SCENARIOS)]
    scores = [score(s, r) for s, r in runs]
    for (s, r), sc in zip(runs, scores, strict=True):
      rows.append(
        [
          ending,
          s["id"],
          s["request"],
          ", ".join(c["name"] for c in r.tool_calls),
          "✓" if sc["success"] else "✗ " + ", ".join(k for k, v in sc.items() if not v and k != "success"),
          r.outcome.reply if r.outcome else r.error,
        ]
      )
    out.append(
      {
        "variant": "A · respond node" if ending == "respond" else "B · final_answer tool",
        "success": sum(sc["success"] for sc in scores) / len(scores),
        "actions": sum(sc["actions_ok"] for sc in scores) / len(scores),
        "calls": round(sum(r.steps for _, r in runs) / len(runs), 2),
      }
    )
  a, b = out
  return [
    scorecard(
      out,
      [
        Metric("success", "scenarios right"),
        Metric("actions", "right actions taken"),
        Metric("calls", "model calls", "lower", "text"),
      ],
    ),
    table(["Ending", "Scenario", "Request", "tools", "result", "reply"], rows),
    callout(
      f"Option A (a respond node) succeeded on {a['success']:.0%} of these scenarios and option B (the outcome as a "
      f"tool the agent must call) on {b['success']:.0%}, at {a['calls']} vs {b['calls']} model calls. "
      + (
        "With A the agent may stop on plain text, so it can say it will escalate without calling escalate, and the "
        "respond call then writes whatever the conversation suggests. With B, tool_choice='any' keeps it in tool mode "
        "until it calls final_answer. The production desk uses B and keeps A as a fallback."
        if b["success"] > a["success"]
        else "Here A did as well: B's advantage (the agent can't stop on text) shows up on escalations and refusals."
      ),
      "decision",
      "Option B in production",
    ),
  ] + [
    example(
      "lg.l24.1",
      "Option A: a final respond node",
      option_a_respond_node,
      "",
      [table(["variant", "success", "calls"], [[a["variant"], f"{a['success']:.0%}", a["calls"]]])],
    ),
    example(
      "lg.l24.2",
      "Option B: the schema as a tool",
      option_b_schema_as_tool,
      "",
      [code(json.dumps(option_b_schema_as_tool(), indent=1), "json")],
    ),
    example(
      "lg.l24.3",
      "Trade-offs",
      option_a_respond_node,
      "",
      [
        table(
          ["variant", "success", "right actions", "model calls"],
          [[o["variant"], f"{o['success']:.0%}", f"{o['actions']:.0%}", o["calls"]] for o in out],
        )
      ],
    ),
  ]


# ── Lab 5: agent streaming (create_agent) ──────────────────────────────────────
STREAM_QUESTION = {"messages": [HumanMessage("What do a Cappuccino and an Oat Latte cost at the café?")]}


def _menu_agent(model=None, name=None):
  from langchain.agents import create_agent

  store = _store()
  menu = next(t for t in _lookup_tools(store) if t.name == "cafe_menu_item")
  return create_agent(
    model or get_chat_model(), [menu], name=name, system_prompt="Use cafe_menu_item for every item. Be brief."
  )


def stream_updates(agent):
  return [
    (node, [type(m).__name__ for m in out["messages"]])
    for update in agent.stream(STREAM_QUESTION, stream_mode="updates")
    for node, out in update.items()
  ]


def stream_tokens(agent):
  return [
    chunk.text
    for chunk, meta in agent.stream(STREAM_QUESTION, stream_mode="messages")
    if meta["langgraph_node"] == "model" and chunk.text
  ]


def stream_values(agent):
  return [len(state["messages"]) for state in agent.stream(STREAM_QUESTION, stream_mode="values")]


def _warehouse_agent():
  from langchain.agents import create_agent
  from langgraph.config import get_stream_writer

  @tool
  def warehouse(blend: str) -> str:
    """Bags of a blend in stock."""
    write = get_stream_writer()
    write(f"asking the warehouse about {blend}")  # only sent when "custom" is streamed
    return f"{blend}: 12 bags"

  return create_agent(get_chat_model(), [warehouse], system_prompt="Use the tool. Be brief.")


STOCK_QUESTION = {"messages": [HumanMessage("Are Coorg Dawn and Nilgiri Sunrise in stock?")]}


def stream_custom(agent):
  return list(agent.stream(STOCK_QUESTION, stream_mode="custom"))


def several_modes(agent):
  return [
    (mode, next(iter(data)) if mode == "updates" else data)
    for mode, data in agent.stream(STOCK_QUESTION, stream_mode=["updates", "custom"])
  ]


def v2_chunks(agent):
  return [(c["type"], c["ns"]) for c in agent.stream(STREAM_QUESTION, stream_mode=["updates", "values"], version="v2")]


def tool_call_chunks(agent):
  pieces, full, finished = [], None, []
  for chunk in agent.stream(STREAM_QUESTION, stream_mode="messages", version="v2"):
    token, meta = chunk["data"]
    if isinstance(token, AIMessageChunk) and meta["langgraph_node"] == "model":
      pieces += [(c["name"], c["args"]) for c in token.tool_call_chunks]  # raw JSON text, possibly in pieces
      full = token if full is None else full + token  # adding chunks assembles the calls
      if token.chunk_position == "last":
        finished.append([(c["name"], c["args"]) for c in full.tool_calls] or full.text)
        full = None
  return pieces, finished


def reasoning_stream():
  from langchain.agents import create_agent

  thinker = get_chat_model(role="reasoning", model=get_settings().judge_models[0], reasoning_effort="low")
  agent = create_agent(thinker, [], system_prompt="Be brief.")
  parts = {"reasoning": "", "text": ""}
  order = []
  for chunk in agent.stream(
    {"messages": [HumanMessage("Is 12 bags of 250 g more than 3 kg?")]}, stream_mode="messages", version="v2"
  ):
    token, _ = chunk["data"]
    for block in token.content_blocks if isinstance(token, AIMessageChunk) else []:
      if block["type"] in parts:
        parts[block["type"]] += block.get("reasoning") or block.get("text", "")
        if not order or order[-1] != block["type"]:
          order.append(block["type"])
  return parts, order  # the thinking arrives before the answer


def front_desk(barista_model=None):
  from langchain.agents import create_agent

  barista = _menu_agent(barista_model, name="barista")

  @tool
  def ask_barista(question: str) -> str:
    """Ask the barista agent about café menu prices."""
    return barista.invoke({"messages": [HumanMessage(question)]})["messages"][-1].text

  return create_agent(
    get_chat_model(), [ask_barista], name="front_desk", system_prompt="For prices, always ask the barista. Be brief."
  )


def who_is_talking(agent):
  speakers = []
  for chunk in agent.stream(STREAM_QUESTION, stream_mode="messages", subgraphs=True, version="v2"):
    _, meta = chunk["data"]
    name = meta.get("lc_agent_name")  # from create_agent(name=...)
    if not speakers or speakers[-1] != name:
      speakers.append(name)
  return speakers


def event_stream(agent):
  stream = agent.stream_events(STREAM_QUESTION, version="v3")
  seen = []
  for view, item in stream.interleave("messages", "tool_calls", "subagents"):
    if view == "tool_calls":
      seen.append(f"tool {item.tool_name} started")
    elif view == "subagents":
      seen.append(f"sub-agent {item.name} started")
      for message in item.messages:
        seen.append(f"  {item.name}: {''.join(message.text)[:60]}")
    else:
      seen.append(f"{item.node}: {''.join(item.text)[:60]}")
  return seen


def no_token_streaming():
  agent = front_desk(get_chat_model(disable_streaming=True))  # the barista's replies arrive whole
  kinds: dict[str, int] = {}
  for chunk in agent.stream(STREAM_QUESTION, stream_mode="messages", subgraphs=True, version="v2"):
    token, meta = chunk["data"]
    key = f"{meta.get('lc_agent_name')} · {type(token).__name__}"
    kinds[key] = kinds.get(key, 0) + 1
  return kinds


@lab(
  "agents.agent-streaming",
  "Agent streaming: watch create_agent work",
  uses_model=True,
  calls="~30 calls to Mistral and 1 to Groq",
)
def agent_streaming():
  """One menu question streamed every way create_agent offers, then a front desk with a barista sub-agent."""
  agent = _menu_agent()
  updates, tokens, values = stream_updates(agent), stream_tokens(agent), stream_values(agent)
  warehouse = _warehouse_agent()
  custom, modes, v2 = stream_custom(warehouse), several_modes(warehouse), v2_chunks(agent)
  pieces, finished = tool_call_chunks(agent)
  parts, order = reasoning_stream()
  desk = front_desk()
  speakers, events, kinds = who_is_talking(desk), event_stream(desk), no_token_streaming()
  return [
    table(
      ["stream_mode", "what arrives", "count"],
      [
        ["updates", " → ".join(f"{n} {m}" for n, m in updates), len(updates)],
        ["messages", "".join(tokens)[:120], len(tokens)],
        ["values", "messages in state: " + ", ".join(map(str, values)), len(values)],
        ["custom", "; ".join(map(str, custom)), len(custom)],
        ["updates + custom", ", ".join(f"{m}:{d}" for m, d in modes), len(modes)],
      ],
      title="The stream modes, one question",
    ),
    table(
      ["What", "Result"],
      [
        ["tool-call chunks", str(pieces)[:200]],
        ["assembled calls", str(finished)[:200]],
        ["reasoning, then text", " → ".join(order)],
        ["who is talking", " → ".join(map(str, speakers))],
      ],
    ),
    code(
      "\n".join(events), "text", title='stream_events(version="v3") · interleave("messages", "tool_calls", "subagents")'
    ),
    callout(
      "updates for progress (one event per finished step), messages for tokens in a chat UI, values for the whole "
      "state, custom for your own events from inside tools; several at once arrive as (mode, data). version='v2' "
      "gives every chunk one shape {type, ns, data}. Tool calls stream as JSON pieces you add up; sub-agents are "
      "told apart by name=; stream_events v3 gives ready-made views. Turn token streaming off for a model whose "
      "words the user shouldn't watch.",
      "decision",
      "Streaming an agent",
    ),
  ] + [
    example(
      "lc.l22.1",
      'stream_mode="updates"',
      stream_updates,
      "",
      [table(["node", "messages"], [[n, str(m)] for n, m in updates])],
    ),
    example(
      "lc.l22.2",
      'stream_mode="messages"',
      stream_tokens,
      "",
      [table(["tokens", "text"], [[len(tokens), "".join(tokens)]])],
    ),
    example(
      "lc.l22.3",
      'stream_mode="values"',
      stream_values,
      "",
      [table(["messages in state after each step"], [[", ".join(map(str, values))]])],
    ),
    example("lc.l22.4", 'stream_mode="custom"', stream_custom, "", [table(["event"], [[str(c)] for c in custom])]),
    example(
      "lc.l22.5", "Several modes at once", several_modes, "", [table(["mode", "data"], [[m, str(d)] for m, d in modes])]
    ),
    example("lc.l23.1", 'version="v2"', v2_chunks, "", [table(["type", "ns"], [[t, str(ns)] for t, ns in v2])]),
    example(
      "lc.l23.2",
      "Tool calls stream too",
      tool_call_chunks,
      "",
      [table(["What", "Result"], [["chunks", str(pieces)], ["assembled", str(finished)]])],
    ),
    example(
      "lc.l23.3",
      "Reasoning streams before the answer",
      reasoning_stream,
      "",
      [table(["part", "text"], [[k, v[:200]] for k, v in parts.items()]), table(["order"], [[" → ".join(order)]])],
    ),
    example(
      "lc.l23.4", "Sub-agents: who is talking", who_is_talking, "", [table(["speaker"], [[str(s)] for s in speakers])]
    ),
    example("lc.l23.5", "Event streaming (v3)", event_stream, "", [code("\n".join(events), "text")]),
    example(
      "lc.l23.6",
      "Turning token streaming off",
      no_token_streaming,
      "",
      [table(["agent · chunk type", "events"], list(kinds.items()))],
    ),
  ]


# ── Lab 6: graph streaming in depth (the support desk) ─────────────────────────
DESK_REQUEST = ("c-1042", "When is my next coffee delivery, and how many points did NC-12093 earn?")


def _desk_input(store, customer_id, text):
  from nimbus.support.agent import CustomerContext

  context = CustomerContext(customer_id, store.customers.get(customer_id, "the customer"))
  return {"messages": [HumanMessage(text)], "customer_id": customer_id, "actions": []}, context


def tokens_by_node(desk, payload, context):
  counts: dict[str, int] = {}
  for _chunk, meta in desk.stream(payload, stream_mode="messages", context=context):
    counts[meta["langgraph_node"]] = counts.get(meta["langgraph_node"], 0) + 1  # every chunk says which node made it
  return counts


def filter_by_node_and_tag(desk, payload, context):
  by_node, by_tag = [], []
  for chunk, meta in desk.stream(payload, stream_mode="messages", context=context):
    if meta["langgraph_node"] == "agent" and isinstance(chunk, AIMessageChunk):
      by_node.append(chunk)
    if "desk-agent" in meta.get("tags", []):  # survives renaming nodes
      by_tag.append(chunk)
  return by_node, by_tag


def progress_events(desk, payload, context):
  started, events = time.perf_counter(), []
  for mode, data in desk.stream(payload, stream_mode=["custom", "updates"], context=context):
    events.append(
      (round((time.perf_counter() - started) * 1000), mode, data["progress"] if mode == "custom" else next(iter(data)))
    )
  return events


def nested_graph_events(desk, payload, context):
  from langgraph.graph import START, StateGraph

  from nimbus.support.desk import DeskState

  parent = StateGraph(DeskState).add_node("desk", desk).add_edge(START, "desk").compile()  # the desk as one node
  return [
    (" > ".join(p.split(":")[0] for p in ns) or "(parent)", list(update))
    for ns, update in parent.stream(payload, stream_mode="updates", subgraphs=True, context=context)
  ]


@lab(
  "agents.graph-streaming",
  "Streaming a graph: tokens by node, filters, progress, subgraphs",
  uses_model=True,
  calls="~15 calls to Mistral",
)
def graph_streaming():
  """The support desk streamed: which node made each token, the reply's tokens only, progress events with their
  timing, and the desk nested inside a bigger graph. Plus how this differs from create_agent's stream."""
  from nimbus.support.desk import build_support_desk

  store = _store()
  desk = build_support_desk(get_chat_model(), store)
  payload, context = _desk_input(store, *DESK_REQUEST)
  counts = tokens_by_node(desk, payload, context)
  by_node, by_tag = filter_by_node_and_tag(desk, payload, context)
  final = sum(by_node[1:], by_node[0]) if by_node else None
  reply_chunks = [c for c in by_node for tc in c.tool_call_chunks if tc.get("name") == "final_answer" or tc.get("args")]
  events = progress_events(desk, payload, context)
  nested = nested_graph_events(desk, payload, context)
  first_ms = events[0][0] if events else None
  last_ms = events[-1][0] if events else None
  return [
    table(
      ["node", "chunks"], list(counts.items()), title="stream_mode='messages': every chunk with the node that made it"
    ),
    table(
      ["ms", "mode", "event"],
      [list(e) for e in events],
      title=f"Progress: the first event after {first_ms} ms, the run ends at {last_ms} ms",
    ),
    table(["from", "nodes"], [[w, str(n)] for w, n in nested], title="subgraphs=True: the desk inside a parent graph"),
    table(
      ["", "create_agent (lab 5)", "the desk graph"],
      [
        ["node names in metadata", "model, tools", ", ".join(counts)],
        ["where the reply is", "the last AIMessage's text tokens", "the final_answer call's JSON arguments"],
        ["progress events", "whatever tools emit", "every node emits one (load, think, done)"],
      ],
      title="Agent streams vs graph streams",
    ),
    callout(
      f"Streaming a graph is the same API as streaming an agent, with your node names in the metadata. The desk "
      f"answers by calling final_answer, so its reply streams as JSON arguments ({len(reply_chunks)} tool-call "
      "chunks here), not as text: a chat UI has to assemble the call (or parse partial JSON) to show it. Progress "
      f"events are the cheap win: the first arrived after {first_ms} ms, long before the reply.",
      "decision",
      "Graph streaming",
    ),
  ] + [
    example(
      "lg.l25.1",
      "Tokens with the node that made them",
      tokens_by_node,
      "",
      [table(["node", "chunks"], list(counts.items()))],
    ),
    example(
      "lg.l25.2",
      "Filter by node or by tag",
      filter_by_node_and_tag,
      "The desk's agent model is tagged 'desk-agent'.",
      [
        table(["filter", "chunks"], [["node == agent", len(by_node)], ["tag desk-agent", len(by_tag)]]),
        code(str([(c["name"], c["args"]) for c in final.tool_calls]) if final else "", "text", "the chunks, added up"),
      ],
    ),
    example(
      "lg.l25.3",
      'stream_mode="custom": progress',
      progress_events,
      "",
      [table(["ms", "event"], [[e[0], e[2]] for e in events if e[1] == "custom"])],
    ),
    example(
      "lg.l25.4",
      "Several modes at once",
      progress_events,
      "",
      [table(["ms", "mode", "data"], [list(e) for e in events])],
    ),
    example(
      "lg.l25.5",
      "subgraphs=True",
      nested_graph_events,
      "",
      [table(["from", "nodes"], [[w, str(n)] for w, n in nested])],
    ),
  ]


# ── Lab 7: event streaming (v3) ────────────────────────────────────────────────
def message_views(graph, payload, context):
  stream = graph.stream_events(payload, version="v3", context=context)
  calls = [(m.node, "".join(m.text)[:80], [c["name"] for c in m.tool_calls.get()]) for m in stream.messages]
  return calls, list(stream.output)  # one item per model call; then the final state


def subgraph_views(graph, payload, context):
  stream = graph.stream_events(payload, version="v3", context=context)
  return [(sub.graph_name, sub.path, len(list(sub.messages))) for sub in stream.subgraphs]


def interleaved(graph, payload, context):
  stream = graph.stream_events(payload, version="v3", context=context)
  return [
    f"state: {len(item['messages'])} messages" if view == "values" else f"model call in {item.node}"
    for view, item in stream.interleave("values", "messages")
  ]


def raw_events(graph, payload, context):
  counts: dict[tuple, int] = {}
  for event in graph.stream_events(payload, version="v3", context=context):
    ns = event["params"]["namespace"]
    key = (event["method"], ns[0].split(":")[0] if ns else "root")
    counts[key] = counts.get(key, 0) + 1
  return counts


def progress_transformer():
  from langgraph.stream import StreamChannel, StreamTransformer

  class Progress(StreamTransformer):
    """Our own view: the desk's progress messages, as a stream of strings."""

    required_stream_modes = ("custom",)

    def __init__(self, scope=()):
      super().__init__(scope)
      self.progress = StreamChannel()

    def init(self):
      return {"progress": self.progress}

    def process(self, event):
      data = event["params"]["data"]
      if event["method"] == "custom" and isinstance(data, dict) and "progress" in data:
        self.progress.push(data["progress"])
      return True  # keep the event in the main stream too

  return Progress


def tool_call_view(graph, payload, context):
  from langgraph.prebuilt import ToolCallTransformer

  stream = graph.stream_events(payload, version="v3", context=context, transformers=[ToolCallTransformer])
  done = []
  for tool_call in stream.tool_calls:
    for _ in tool_call.output_deltas:  # wait for it to finish
      pass
    done.append((tool_call.tool_name, json.dumps(tool_call.input), str(tool_call.output.content)[:100]))
  return done


@lab(
  "agents.event-streaming", "Event streaming (v3): typed views of a run", uses_model=True, calls="~20 calls to Mistral"
)
def event_streaming():
  """The support desk (nested in a parent graph) through stream_events v3: messages, subgraphs, interleaved views,
  the raw events, a custom transformer for progress, and the built-in tool-call view."""
  from langgraph.graph import START, StateGraph

  from nimbus.support.desk import DeskState, build_support_desk

  store = _store()
  desk = build_support_desk(get_chat_model(), store)
  payload, context = _desk_input(store, *DESK_REQUEST)
  parent = StateGraph(DeskState).add_node("desk", desk).add_edge(START, "desk").compile()
  calls, output_keys = message_views(desk, payload, context)
  subs = subgraph_views(parent, payload, context)
  order = interleaved(desk, payload, context)
  counts = raw_events(desk, payload, context)
  stream = desk.stream_events(payload, version="v3", context=context, transformers=[progress_transformer()])
  progress = list(stream.extensions["progress"])  # read while the run goes: views don't replay
  tools_run = tool_call_view(desk, payload, context)
  return [
    table(
      ["node", "text", "tool calls"],
      [[n, t, str(c)] for n, t, c in calls],
      title="stream.messages: one item per model call",
    ),
    table(
      ["method", "where", "events"],
      [[m, w, n] for (m, w), n in sorted(counts.items())],
      title="The raw protocol events underneath",
    ),
    table(
      ["tool", "input", "output"], [list(t) for t in tools_run], title="ToolCallTransformer: tool executions as a view"
    ),
    callout(
      "stream_events(version='v3') turns the raw events into views: messages (one per model call, with text and "
      "tool calls), values, subgraphs by name, the final output, and your own views through a StreamTransformer. "
      "Views are read while the run happens and don't replay, so read every view you need in one pass (interleave) "
      "or start another run. It's still marked beta.",
      "decision",
      "Event streaming",
    ),
  ] + [
    example(
      "lg.l26.1",
      ".messages and .output",
      message_views,
      "",
      [
        table(["node", "tool calls"], [[n, str(c)] for n, _, c in calls]),
        table(["final state keys"], [[", ".join(output_keys)]]),
      ],
    ),
    example(
      "lg.l26.2",
      ".subgraphs",
      subgraph_views,
      "",
      [table(["graph", "path", "model calls"], [[g, str(p), n] for g, p, n in subs])],
    ),
    example("lg.l26.3", "interleave()", interleaved, "", [table(["in order"], [[o] for o in order])]),
    example(
      "lg.l26.4",
      "The raw events",
      raw_events,
      "",
      [table(["method", "where", "events"], [[m, w, n] for (m, w), n in sorted(counts.items())])],
    ),
    example(
      "lg.l26.5",
      "Your own view: a StreamTransformer",
      progress_transformer,
      "",
      [table(["progress"], [[p] for p in progress])],
    ),
    example(
      "lg.l26.6",
      "ToolCallTransformer",
      tool_call_view,
      "",
      [table(["tool", "input", "output"], [list(t) for t in tools_run])],
    ),
  ]


# ── Production path: the support desk vs support agent v1 ──────────────────────
def evaluate_desk() -> list[dict]:
  from nimbus.support.agent_eval import run_desk_scenario, run_scenario, score

  model, rows = get_chat_model(), []
  for scenario in _scenarios():
    for name, run in (("v1", lambda s: run_scenario(model, s)), ("desk", lambda s: run_desk_scenario(model, s))):
      result = run(scenario)
      rows.append({"system": name, "scenario": scenario, "run": result, "score": score(scenario, result)})
  return rows


@lab(
  "agents.support-desk-eval",
  "Evaluate: the support desk vs support agent v1",
  uses_model=True,
  calls="~100 calls to Mistral: 15 scenarios × 2 agents, about 5 minutes",
)
def support_desk_eval():
  """Support agent v1 (create_agent, LangChain Module 4) and the support desk graph on the 10 agent scenarios plus 5
  harder ones (two requests at once, an order that doesn't exist, a pause longer than allowed, in Hinglish, points
  for an amount, a refund for a cancelled order)."""
  started = time.perf_counter()
  rows = evaluate_desk()
  seconds = time.perf_counter() - started
  out = []
  for name in ("v1", "desk"):
    mine = [r for r in rows if r["system"] == name]
    rate = lambda key, m=mine: round(sum(r["score"][key] for r in m) / len(m), 3)  # noqa: E731
    out.append(
      {
        "variant": "v1 · create_agent" if name == "v1" else "desk · agent in a workflow",
        "success": rate("success"),
        "tools_ok": rate("tools_ok"),
        "actions_ok": rate("actions_ok"),
        "outcome_ok": rate("outcome_ok"),
        "answer_ok": rate("answer_ok"),
        "steps": round(sum(r["run"].steps for r in mine) / len(mine), 2),
        "tokens": round(sum(r["run"].usage.input_tokens + r["run"].usage.output_tokens for r in mine) / len(mine)),
        "p50_ms": round(statistics.median(r["run"].ms for r in mine)),
      }
    )
  failures = [
    [
      r["system"],
      r["scenario"]["id"],
      r["scenario"]["request"],
      ", ".join(k for k, v in r["score"].items() if not v and k != "success"),
      (r["run"].outcome.reply if r["run"].outcome else r["run"].error) or "",
    ]
    for r in rows
    if not r["score"]["success"]
  ]
  v1, desk = out
  return [
    scorecard(
      out,
      [
        Metric("success", "success"),
        Metric("tools_ok", "right tools"),
        Metric("actions_ok", "right actions"),
        Metric("outcome_ok", "right outcome"),
        Metric("answer_ok", "right facts in the reply"),
        Metric("steps", "model calls", "lower", "text"),
        Metric("tokens", "tokens / scenario", "lower", "int"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"15 scenarios (10 from LangChain Module 4 + 5 harder) · {seconds:.0f} s",
    ),
    table(["Agent", "Scenario", "Request", "failed on", "reply"], failures, title="Every failure"),
    callout(
      f"The desk scored {desk['success']:.0%} against v1's {v1['success']:.0%}, at {desk['steps']} vs {v1['steps']} "
      f"model calls and {desk['tokens']:,} vs {v1['tokens']:,} tokens per scenario. What the workflow adds is "
      "around the agent: the reply language decided in code (while this module was built, ministral answered "
      "English questions from customers with Indian names in Hindi), a ticket for the support desk, needs_human "
      "decided by code when the agent "
      "escalated or claimed an action no tool ran, and progress events for the UI."
      + (
        " It ships as the production support agent."
        if desk["success"] >= v1["success"]
        else " On success alone it doesn't beat v1 in this run."
      ),
      "decision" if desk["success"] >= v1["success"] else "warning",
      "Ship the support desk" if desk["success"] >= v1["success"] else "No gain on success",
    ),
  ]
