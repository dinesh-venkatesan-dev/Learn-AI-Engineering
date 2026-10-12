"""LangChain · Module 4 labs: agents with create_agent. The agent and its graph, structured responses, long
conversations, runtime context and custom state. Each learning point is a small snippet function that the lab runs;
the Covers tab shows its code next to what it produced."""

import json
import time
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from academy.labs.runtime import Input, Metric, answer, callout, code, columns, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model
from nimbus.support.agent_eval import evaluate_agent, load_scenarios, run_scenario, score
from nimbus.support.assistant import answer_with_tools
from nimbus.support.tools import SupportStore, build_tools


def _ms(started: float) -> int:
  return round((time.perf_counter() - started) * 1000)


@lru_cache
def _tools():
  return build_tools(SupportStore())


def _transcript(messages) -> list[list]:
  rows = []
  for m in messages:
    if isinstance(m, AIMessage) and m.tool_calls:
      rows.append(
        [
          "AIMessage",
          "asks for " + ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in m.tool_calls),
        ]
      )
    elif isinstance(m, ToolMessage):
      rows.append([f"ToolMessage ({m.name})", str(m.content)[:150]])
    elif not isinstance(m, SystemMessage):
      rows.append([type(m).__name__, m.text[:150]])
  return rows


# ── Lab 1: create_agent ─────────────────────────────────────────────────────────
def create_an_agent(model, tools):
  from langchain.agents import create_agent

  agent = create_agent(model, tools=tools, system_prompt="You are Nimbus Coffee's support agent. Use the tools.")
  return agent  # a compiled LangGraph graph, ready to run


def look_at_the_graph(agent):
  graph = agent.get_graph()
  nodes = list(graph.nodes)  # a "model" node and a "tools" node, looping
  edges = [(e.source, e.target) for e in graph.edges]
  return nodes, edges


def invoke_with_state(agent):
  from langchain_core.messages import HumanMessage

  result = agent.invoke({"messages": [HumanMessage("Where is my order NC-78123, and is the oat latte vegan?")]})
  return result  # the final state: {"messages": [...every step...]}


def read_every_step(result):
  steps = [(type(m).__name__, getattr(m, "tool_calls", None) or m.text[:60]) for m in result["messages"]]
  answer = result["messages"][-1].text  # the last AIMessage is the answer
  return steps, answer


@lab(
  "lc.create-agent",
  "create_agent: the tool loop, built for you",
  uses_model=True,
  calls="2–3 calls for the agent + 2–3 for the hand-written loop (Mistral)",
)
def create_agent_lab():
  model, tools = get_chat_model(), _tools()
  agent = create_an_agent(model, tools)
  nodes, edges = look_at_the_graph(agent)
  started = time.perf_counter()
  result = invoke_with_state(agent)
  agent_ms = _ms(started)
  steps, final = read_every_step(result)
  started = time.perf_counter()
  manual = answer_with_tools(model, tools, "Where is my order NC-78123, and is the oat latte vegan?")
  manual_ms = _ms(started)
  ai = [m for m in result["messages"] if isinstance(m, AIMessage)]
  return [
    table(
      ["Message", "Content"], _transcript(result["messages"]), title="agent.invoke(...): every step is in the result"
    ),
    scorecard(
      [
        {
          "variant": "create_agent",
          "model_calls": len(ai),
          "tool_calls": sum(len(m.tool_calls) for m in ai),
          "ms": agent_ms,
        },
        {
          "variant": "the hand-written loop (Module 3)",
          "model_calls": manual.rounds,
          "tool_calls": len(manual.tool_calls),
          "ms": manual_ms,
        },
      ],
      [
        Metric("model_calls", "model calls", "lower", "int"),
        Metric("tool_calls", "tool calls", "none", "int"),
        Metric("ms", "time", "lower", "ms"),
      ],
      title="The same question: create_agent vs the loop you wrote",
    ),
    columns([answer(final, "create_agent"), answer(manual.answer, "hand-written loop")]),
    callout(
      f"`create_agent` builds the loop from Module 3 as a LangGraph graph (nodes: {', '.join(nodes)}): the model node "
      "calls the model, the tools node runs the requested tools, and it loops until the model answers without tool "
      "calls. Same calls, same answer; what you gain is everything built on the graph: memory, streaming, "
      "middleware, human approval and structured responses (the rest of this module and the next).",
      "decision",
      "What to take away",
    ),
    example(
      "lc.l21.1",
      "create_agent(model, tools, system_prompt)",
      create_an_agent,
      "It returns a compiled LangGraph graph.",
      [table(["Expression", "Value"], [["type(agent).__name__", type(agent).__name__]])],
    ),
    example(
      "lc.l21.2",
      "The graph: a model node and a tools node, looping",
      look_at_the_graph,
      "The model node decides; the tools node runs what it asked for; back to the model until no tool calls.",
      [table(["Variable", "Value"], [["nodes", ", ".join(nodes)], ["edges", ", ".join(f"{a}→{b}" for a, b in edges)]])],
    ),
    example(
      "lc.l21.3",
      "invoke() takes state and returns state",
      invoke_with_state,
      'In: `{"messages": [...]}`. Out: the same keys, with every message the run added.',
      [
        table(
          ["Expression", "Value"],
          [["list(result)", json.dumps(list(result))], ["len(result['messages'])", len(result["messages"])]],
        )
      ],
    ),
    example(
      "lc.l21.4",
      "Read the result: every step of the loop",
      read_every_step,
      "Human, AI asking for tools, tool results, and the final AI answer.",
      [
        table(
          ["Message", "Calls or text"],
          [[a, json.dumps(b, ensure_ascii=False)[:120] if not isinstance(b, str) else b] for a, b in steps],
        )
      ],
    ),
  ]


# ── Lab 2: structured responses ─────────────────────────────────────────────────
def answer_with_an_object(model, get_order):
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy
  from pydantic import BaseModel, Field

  class OrderStatus(BaseModel):
    order_id: str
    status: str
    on_its_way: bool = Field(description="true if shipped but not yet delivered")

  agent = create_agent(model, tools=[get_order], response_format=ToolStrategy(OrderStatus))
  result = agent.invoke({"messages": [HumanMessage("Is order NC-78123 on its way?")]})
  status = result["structured_response"]  # a validated OrderStatus
  return result, status


def let_it_fix_bad_output(model):
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy
  from pydantic import BaseModel, Field

  class Rating(BaseModel):
    stars: int = Field(ge=1, le=5, description="1 to 5")
    comment: str

  agent = create_agent(
    model,
    tools=[],
    system_prompt="Copy the rating exactly as the customer wrote it.",
    response_format=ToolStrategy(Rating),
  )  # handle_errors=True by default
  result = agent.invoke({"messages": [HumanMessage("Parse this review: Amazing beans, 10/10!")]})
  feedback = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]  # the error, then success
  return feedback, result["structured_response"]


def your_own_retry_message(model):
  from langchain.agents import create_agent
  from langchain.agents.structured_output import StructuredOutputValidationError, ToolStrategy
  from pydantic import BaseModel, Field

  class Rating(BaseModel):
    stars: int = Field(ge=1, le=5)
    comment: str

  def explain(error: Exception) -> str:  # what the model reads after an invalid answer
    return "stars must be 1 to 5. Convert other scales: 10/10 → 5, 4/10 → 2."

  review = {"messages": [HumanMessage("Parse this review: Amazing beans, 10/10!")]}
  guided = create_agent(
    model,
    tools=[],
    system_prompt="Copy the rating exactly as written.",
    response_format=ToolStrategy(Rating, handle_errors=explain),
  ).invoke(review)
  strict = create_agent(
    model,
    tools=[],
    system_prompt="Copy the rating exactly as written.",
    response_format=ToolStrategy(Rating, handle_errors=False),
  )
  try:
    strict.invoke(review)
  except StructuredOutputValidationError as error:
    raised = error
  return guided["structured_response"], raised


@lab(
  "lc.agent-structured",
  "Agent structured output: an agent that ends with typed data",
  uses_model=True,
  calls="about 8 calls to Mistral",
)
def agent_structured():
  model = get_chat_model()
  result, status = answer_with_an_object(model, _tools()[0])
  feedback, fixed = let_it_fix_bad_output(model)
  guided, raised = your_own_retry_message(model)
  outcome_calls = [c["name"] for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
  return [
    table(
      ["Message", "Content"],
      _transcript(result["messages"]),
      title="The agent used get_order, then 'called' OrderStatus",
    ),
    code(repr(status), "python", 'result["structured_response"]'),
    table(
      ["Case", "What happened"],
      [
        ["the model wrote stars=10 (rule: 1 to 5)", " → ".join(f[:90] for f in feedback)],
        ["…after the automatic retry", repr(fixed)],
        ["handle_errors=<your function>", repr(guided)],
        ["handle_errors=False", f"{type(raised).__name__}: {str(raised).splitlines()[0][:120]}"],
      ],
      title="When the model's output breaks the schema",
    ),
    callout(
      '`response_format` makes the agent finish with a validated object in `result["structured_response"]`. With '
      "ToolStrategy, the schema is offered as one more tool (here: " + " → ".join(outcome_calls) + "); calling it ends "
      "the run. Invalid output goes back to the model with the error so it can fix it, with LangChain's message, your "
      "own, or not at all (`handle_errors=False` raises). The production agent's scorecard compares this with "
      "ProviderStrategy (the provider's JSON-schema mode).",
      "decision",
      "What to take away",
    ),
    example(
      "lc.l24.1",
      "response_format: tools first, then a validated object",
      answer_with_an_object,
      "The agent looks things up with its tools, then answers in the schema.",
      [code(repr(status), "python")],
    ),
    example(
      "lc.l24.2",
      'The result is in result["structured_response"]',
      answer_with_an_object,
      "Next to the messages, in the final state.",
      [
        table(
          ["Expression", "Value"],
          [["list(result)", json.dumps(list(result))], ["status.on_its_way", str(status.on_its_way)]],
        )
      ],
    ),
    example(
      "lc.l24.3",
      "How it works: the schema is one more tool",
      answer_with_an_object,
      "The model ends the run by calling a tool named after the schema.",
      [table(["Tool calls, in order", ""], [[" → ".join(outcome_calls), ""]])],
    ),
    example(
      "lc.l24.4",
      "Invalid output goes back to the model, which fixes it",
      let_it_fix_bad_output,
      "The validation error arrives as a ToolMessage; the model tries again.",
      [
        table(["ToolMessage", "Content"], [[i + 1, f[:200]] for i, f in enumerate(feedback)]),
        code(repr(fixed), "python"),
      ],
    ),
    example(
      "lc.l24.5",
      "handle_errors: your own retry message, or raise",
      your_own_retry_message,
      "Tell the model exactly how to fix it, or fail loudly when a retry would hide a problem.",
      [table(["Variable", "Value"], [["guided", repr(guided)], ["raised", type(raised).__name__]])],
    ),
  ]


# ── Lab 3: long conversations ───────────────────────────────────────────────────
FACTS = ["My name is Ravi.", "My order is NC-78123.", "I'm vegan.", "I live in Chennai.", "My cat is called Pixel."]
QUESTION = "What's my name, my order number and my cat's name? One short sentence."


def history_grows(model):
  from langchain.agents import create_agent
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(model, system_prompt="Reply in one short sentence.", checkpointer=InMemorySaver())
  config = {"configurable": {"thread_id": "ravi"}}  # one conversation = one thread
  input_tokens = []
  for fact in FACTS:
    reply = agent.invoke({"messages": [HumanMessage(fact)]}, config)["messages"][-1]
    input_tokens.append(reply.usage_metadata["input_tokens"])  # every turn resends the whole history
  return agent, config, input_tokens


def keep_the_last_messages(history):
  from langchain_core.messages import trim_messages

  trimmed = trim_messages(
    history,
    strategy="last",
    token_counter=len,
    max_tokens=4,  # keep the last 4 messages
    include_system=True,
    start_on="human",
  )
  return trimmed


def summarise_old_messages(model):
  from langchain.agents import create_agent
  from langchain.agents.middleware import SummarizationMiddleware
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
    model,
    system_prompt="Reply in one short sentence.",
    checkpointer=InMemorySaver(),
    middleware=[SummarizationMiddleware(model=model, trigger=("messages", 6), keep=("messages", 2))],
  )
  config = {"configurable": {"thread_id": "ravi-summary"}}
  for fact in FACTS:
    agent.invoke({"messages": [HumanMessage(fact)]}, config)
  kept = agent.get_state(config).values["messages"]  # the first message is now a summary
  return agent, config, kept


def forget_one_thing(agent, config):
  from langchain_core.messages import RemoveMessage

  messages = agent.get_state(config).values["messages"]
  said = next(i for i, m in enumerate(messages) if "vegan" in m.text.lower())  # the customer's message...
  pair = messages[said : said + 2]  # ...and the reply right after it, which may repeat the fact
  agent.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in pair]})  # gone from the saved history
  left = agent.get_state(config).values["messages"]
  return len(messages), len(left)


def clear_everything(agent, config):
  from langchain_core.messages import RemoveMessage
  from langgraph.graph.message import REMOVE_ALL_MESSAGES

  agent.update_state(config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})
  left = agent.get_state(config).values["messages"]
  return len(left)


def _remembers(text: str) -> int:
  low = text.casefold()
  return sum(fact in low for fact in ("ravi", "78123", "pixel"))


@lab(
  "lc.agent-long-chats",
  "Managing context: long conversations inside the model's limits",
  uses_model=True,
  calls="about 20 calls to Mistral (three 6-turn conversations, plus summaries)",
)
def agent_long_chats():
  model = get_chat_model()
  agent, config, growth = history_grows(model)
  history = agent.get_state(config).values["messages"]
  full = agent.invoke({"messages": [HumanMessage(QUESTION)]}, config)["messages"][-1]
  trimmed = keep_the_last_messages([SystemMessage("Reply in one short sentence."), *history])
  trim_reply = model.invoke([*trimmed, HumanMessage(QUESTION)])
  s_agent, s_config, kept = summarise_old_messages(model)
  summary_reply = s_agent.invoke({"messages": [HumanMessage(QUESTION)]}, s_config)["messages"][-1]
  before, after = forget_one_thing(agent, config)
  vegan_reply = agent.invoke({"messages": [HumanMessage("Am I vegan? Answer yes, no, or I don't know.")]}, config)[
    "messages"
  ][-1]
  cleared = clear_everything(agent, config)
  rows = [
    {
      "variant": "full history",
      "input_tokens": full.usage_metadata["input_tokens"],
      "remembered": _remembers(full.text),
    },
    {
      "variant": "trim_messages (last 4)",
      "input_tokens": trim_reply.usage_metadata["input_tokens"],
      "remembered": _remembers(trim_reply.text),
    },
    {
      "variant": "SummarizationMiddleware",
      "input_tokens": summary_reply.usage_metadata["input_tokens"],
      "remembered": _remembers(summary_reply.text),
    },
  ]
  return [
    table(
      ["Turn", "Message", "Input tokens for that turn"],
      [[i + 1, f, t] for i, (f, t) in enumerate(zip(FACTS, growth, strict=True))],
      title="Every turn resends the whole history",
    ),
    scorecard(
      rows,
      [
        Metric("input_tokens", "input tokens (last turn)", "lower", "int"),
        Metric("remembered", "facts remembered (of 3)", "higher", "int"),
      ],
      title=f'Then: "{QUESTION}"',
    ),
    columns(
      [answer(full.text, "full history"), answer(trim_reply.text, "trimmed"), answer(summary_reply.text, "summarised")]
    ),
    table(
      ["Action", "Result"],
      [
        [
          "RemoveMessage on the 'vegan' turn",
          f"{before} → {after} messages; then 'Am I vegan?' → {vegan_reply.text.strip()[:80]}",
        ],
        ["RemoveMessage(id=REMOVE_ALL_MESSAGES)", f"{cleared} messages left"],
      ],
      title="Deleting from the saved history",
    ),
    callout(
      _long_chat_verdict(rows),
      "decision",
      "Which to use",
    ),
    example(
      "lc.l26.1",
      "Why it matters: history grows every turn",
      history_grows,
      "With a checkpointer the agent remembers, by resending everything.",
      [table(["Turn", "input tokens"], [[i + 1, t] for i, t in enumerate(growth)])],
    ),
    example(
      "lc.l26.2",
      "trim_messages: keep only the most recent",
      keep_the_last_messages,
      "Cheap and simple; whatever falls off is forgotten.",
      [
        table(["#", "Kept message"], [[i + 1, f"{type(m).__name__}: {m.text[:80]}"] for i, m in enumerate(trimmed)]),
        answer(trim_reply.text, "Asked afterwards"),
      ],
    ),
    example(
      "lc.l26.3",
      "SummarizationMiddleware: replace old messages with a summary",
      summarise_old_messages,
      "Once the history reaches the trigger, older messages become one summary message.",
      [
        table(["#", "Message in state"], [[i + 1, f"{type(m).__name__}: {m.text[:120]}"] for i, m in enumerate(kept)]),
        answer(summary_reply.text, "Asked afterwards"),
      ],
    ),
    example(
      "lc.l26.4",
      "RemoveMessage: delete specific messages for good",
      forget_one_thing,
      'For "forget that" requests and personal data: removed from the saved state, not just hidden. Remove the '
      "reply too: the model's answer to 'I'm vegan' can repeat the fact.",
      [
        table(
          ["Variable", "Value"],
          [["before", before], ["after", after], ["asked 'Am I vegan?'", vegan_reply.text.strip()[:120]]],
        )
      ],
    ),
    example(
      "lc.l26.5",
      "REMOVE_ALL_MESSAGES: clear a whole conversation",
      clear_everything,
      "One special id removes every message in the thread.",
      [table(["Variable", "Value"], [["messages left", cleared]])],
    ),
  ]


def _long_chat_verdict(rows: list[dict]) -> str:
  full, trim, summary = rows
  cheaper = summary["input_tokens"] < full["input_tokens"]
  return (
    f"History grows every turn, and every turn pays for all of it. Trimming cost {trim['input_tokens']} tokens but "
    f"remembered {trim['remembered']} of 3 facts. Summarising remembered {summary['remembered']} of 3 at "
    f"{summary['input_tokens']} tokens vs {full['input_tokens']} for the full history: "
    + (
      "already cheaper. "
      if cheaper
      else "MORE here, because a summary has its own overhead and six short turns "
      "are still small; it pays off on long conversations. "
    )
    + 'Deleting messages is how you honour "forget that": delete the customer\'s message and the reply that '
    "repeats it, or the fact survives. Nimbus's agent summarises once a conversation passes 4,000 tokens and keeps "
    "the last 6 messages as they are."
  )


# ── Lab 4: runtime context and custom state ─────────────────────────────────────
def same_agent_different_customers(model, store):
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain.tools import ToolRuntime, tool

  @dataclass
  class Customer:
    customer_id: str

  @tool
  def my_orders(runtime: ToolRuntime[Customer]) -> str:
    """The current customer's orders."""
    return ", ".join(o["order_id"] for o in store.orders_of(runtime.context.customer_id)) or "No orders."

  agent = create_agent(model, tools=[my_orders], context_schema=Customer)
  question = {"messages": [HumanMessage("Which orders do I have? Just the numbers.")]}
  priya = agent.invoke(question, context=Customer("c-1042"))["messages"][-1].text  # read-only, per call
  meena = agent.invoke(question, context=Customer("c-3301"))["messages"][-1].text
  return priya, meena


def prompt_from_context(model):
  from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain.agents.middleware import ModelRequest, dynamic_prompt

  @dataclass
  class Customer:
    name: str
    language: str

  @dynamic_prompt
  def greet(request: ModelRequest) -> str:  # middleware reads the context through request.runtime
    c = request.runtime.context
    return f"You are Nimbus Coffee's assistant. Greet {c.name} by name and reply in {c.language}. One sentence."

  agent = create_agent(model, tools=[], middleware=[greet], context_schema=Customer)
  hello = {"messages": [HumanMessage("Hi! Do you sell decaf?")]}
  english = agent.invoke(hello, context=Customer("Priya", "English"))["messages"][-1].text
  tamil = agent.invoke(hello, context=Customer("Kavya", "Tamil"))["messages"][-1].text
  return english, tamil


def keep_extra_state(model):
  import operator
  from typing import Annotated

  from langchain.agents import AgentState, create_agent
  from langchain.tools import ToolRuntime, tool
  from langchain_core.messages import ToolMessage
  from langgraph.types import Command

  class NotesState(AgentState):
    notes: Annotated[list[str], operator.add]  # a reducer: parallel updates are concatenated, not overwritten

  @tool
  def add_note(note: str, runtime: ToolRuntime) -> Command:
    """Add a note to the support ticket."""
    return Command(
      update={"notes": [note], "messages": [ToolMessage(f"Noted: {note}", tool_call_id=runtime.tool_call_id)]}
    )

  agent = create_agent(model, tools=[add_note], state_schema=NotesState)
  result = agent.invoke(
    {"messages": [HumanMessage("Add two notes: 'prefers email' and 'allergic to nuts'.")], "notes": []}
  )
  return result["notes"], list(result)


@lab(
  "lc.agent-context-state",
  "Custom state and runtime context: giving the agent extra data",
  uses_model=True,
  calls="about 8 calls to Mistral",
)
def agent_context_state():
  model, store = get_chat_model(), SupportStore()
  priya, meena = same_agent_different_customers(model, store)
  english, tamil = prompt_from_context(model)
  notes, keys = keep_extra_state(model)
  return [
    table(
      ["Customer (context)", "Same question, answer"],
      [["c-1042 Priya", priya], ["c-3301 Meena", meena]],
      title="One agent, different customers: context decides whose orders a tool reads",
    ),
    table(
      ["Context", "Reply"],
      [["Priya, English", english], ["Kavya, Tamil", tamil]],
      title="A prompt built from the context",
    ),
    table(
      ["Expression", "Value"],
      [["result['notes']", json.dumps(notes, ensure_ascii=False)], ["list(result)", json.dumps(keys)]],
      title="Custom state, updated by a tool",
    ),
    callout(
      "**Context** is read-only data per call (who is asking, their plan, settings): tools read it through "
      "`ToolRuntime`, middleware through `request.runtime`, and the model can't see or change it, so it can't ask for "
      "another customer's orders. **State** is what the agent keeps and updates during a run: tools write to it by "
      "returning `Command(update=...)`, and a reducer decides how parallel updates combine.",
      "decision",
      "Context or state",
    ),
    example(
      "lc.l29.1",
      "Runtime context: read-only data per call",
      same_agent_different_customers,
      "Pass `context=` to invoke; the same agent serves every customer.",
      [table(["Variable", "Value"], [["priya", priya], ["meena", meena]])],
    ),
    example(
      "lc.l29.2",
      "Tools read it via ToolRuntime, middleware via request.runtime",
      prompt_from_context,
      "Here a dynamic prompt reads the customer's name and language.",
      [table(["Variable", "Value"], [["english", english], ["tamil", tamil]])],
    ),
    example(
      "lc.l29.3",
      "Custom state: extra fields the agent keeps",
      keep_extra_state,
      "Extend AgentState; the field comes back in the result.",
      [table(["Variable", "Value"], [["keys", json.dumps(keys)]])],
    ),
    example(
      "lc.l29.4",
      "A tool updates state with Command; a reducer merges parallel updates",
      keep_extra_state,
      "Two add_note calls in one turn: `operator.add` concatenates both instead of keeping the last.",
      [table(["Variable", "Value"], [["notes", json.dumps(notes, ensure_ascii=False)]])],
    ),
  ]


# ── Production path labs: the support agent ─────────────────────────────────────
HISTORY = PROJECT_DIR / ".state" / "agent_scorecards.jsonl"
AGENT_METRICS = [
  Metric("success", "success", "higher", "ratio", "Right tools, right actions, right outcome fields, right reply"),
  Metric("tools_ok", "tools", "higher", "ratio"),
  Metric("actions_ok", "actions", "higher", "ratio"),
  Metric("outcome_ok", "outcome", "higher", "ratio"),
  Metric("answer_ok", "reply", "higher", "ratio"),
  Metric("steps", "model calls", "lower", "ratio"),
  Metric("tokens", "tokens", "lower", "int"),
  Metric("cost_per_1k_usd", "$ / 1k", "lower", "usd"),
]


def _scenario_options() -> list[str]:
  return [
    f"{s['id']} · {s['customer_id']} · {s['request']}" for s in load_scenarios(get_settings().eval_agent_scenarios)
  ]


@lab(
  "lc.agent-one",
  "Run one support conversation through the agent",
  uses_model=True,
  calls="2–5 calls to Mistral",
  inputs=[
    Input("scenario", "Labelled scenario", "select", default=_scenario_options()[4], options=_scenario_options())
  ],
)
def agent_one(scenario: str):
  sid = scenario.split(" · ", 1)[0]
  labelled = {s["id"]: s for s in load_scenarios(get_settings().eval_agent_scenarios)}[sid]
  run = run_scenario(get_chat_model(), labelled, get_settings().agent_output)
  if run.outcome is None:
    return [callout(run.error or "no outcome", "warning", "The agent failed")]
  s = score(labelled, run)
  return [
    table(
      ["Message", "Content"], _transcript(run.messages), title=f"Customer {labelled['customer_id']}: the conversation"
    ),
    code(run.outcome.model_dump_json(indent=1), "json", "SupportOutcome (structured_response)"),
    table(
      ["State", "Value"],
      [
        ["actions", json.dumps(run.actions, ensure_ascii=False)],
        ["model calls", run.steps],
        ["tokens", run.usage.input_tokens + run.usage.output_tokens],
        ["ms", round(run.ms)],
      ],
      title="What the agent did",
    ),
    table(["Check", "Result"], [[k, "✓" if v else "✗"] for k, v in s.items()], title="Against the label"),
  ]


@lab(
  "lc.agent-eval",
  "Score the support agent on 10 labelled scenarios",
  uses_model=True,
  calls="about 30 calls to Mistral per strategy (about 3 minutes for both)",
  inputs=[Input("compare", "Compare ToolStrategy with ProviderStrategy", "bool", default=True)],
)
def agent_eval(compare: bool):
  from datetime import UTC, datetime

  model, scenarios = get_chat_model(), load_scenarios(get_settings().eval_agent_scenarios)
  variants = (
    [("ToolStrategy", "tool"), ("ProviderStrategy", "provider")]
    if compare
    else [(get_settings().agent_output, get_settings().agent_output)]
  )
  reports = [evaluate_agent(name, model, scenarios, output) for name, output in variants]
  at = datetime.now(UTC).isoformat(timespec="seconds")
  HISTORY.parent.mkdir(exist_ok=True)
  with HISTORY.open("a") as f:
    for r in reports:
      f.write(json.dumps({"at": at, **r.summary()}) + "\n")
  misses = [
    [
      r.variant,
      sc["id"],
      sc["request"][:50],
      ", ".join(k for k, v in s.items() if not v and k != "success"),
      ", ".join(c["name"] for c in run.tool_calls),
      json.dumps(run.actions)[:60],
    ]
    for r in reports
    for sc, run, s in r.rows
    if not s["success"]
  ]
  history = [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]
  views = [scorecard([r.summary() for r in reports], AGENT_METRICS, title=f"{len(scenarios)} labelled scenarios")]
  if misses:
    views.append(
      table(
        ["Strategy", "Scenario", "Request", "Failed checks", "Tools", "Actions"],
        misses,
        title=f"Not handled ({len(misses)})",
      )
    )
  views.append(
    scorecard(
      [{**h, "variant": f"#{i + 1} · {h['variant']}"} for i, h in enumerate(history)], AGENT_METRICS, title="History"
    )
  )
  return views
