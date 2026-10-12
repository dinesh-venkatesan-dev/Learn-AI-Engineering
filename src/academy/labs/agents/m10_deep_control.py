"""Agents with LangGraph · Module 10 labs: deep agents' memory (AGENTS.md), skills, context management, human in the
loop, typed results and runtime context, customising the harness, and serving. Then the capstone: Nimbus's weekly ops
review by one deep agent with the whole harness, against the same agent with everything pasted into its prompt."""

import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from academy.labs.runtime import Metric, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR
from nimbus.llm import get_chat_model

LIMITS = {"recursion_limit": 80}


def _thread(name: str) -> dict:
  return {"configurable": {"thread_id": name}, **LIMITS}


def _calls(messages) -> list[str]:
  out = []
  for m in messages:
    for c in getattr(m, "tool_calls", None) or []:
      target = c["args"].get("file_path") or c["args"].get("subagent_type") or c["args"].get("to") or ""
      out.append(f"{c['name']}({target})" if target else c["name"])
  return out


# ── Lab 1: memory (AGENTS.md) ──────────────────────────────────────────────────
NAMESPACE = ("lab-ops-memory",)
RULES = (
  "# Nimbus Ops: house rules\n- Address café notes to the manager by name: Bandra is Priya, Colaba is Rahul.\n- "
  "Notes are plain text: no markdown, no bold, no bullet symbols.\n"
)


def memory_agent(model, store):
  from deepagents import create_deep_agent
  from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
  from langgraph.checkpoint.memory import InMemorySaver

  from nimbus.ops.agents import weekly_numbers

  backend = CompositeBackend(
    default=StateBackend(), routes={"/memories/": StoreBackend(namespace=lambda rt: NAMESPACE)}
  )
  return create_deep_agent(
    model=model,
    tools=[weekly_numbers],
    backend=backend,
    store=store,
    memory=["/memories/AGENTS.md"],
    checkpointer=InMemorySaver(),
    system_prompt="You are Nimbus's operations assistant. Use the tools for every number. Be concise.",
  )


def seed(store):
  from deepagents.backends.utils import create_file_data

  store.put(NAMESPACE, "/AGENTS.md", create_file_data(RULES))  # under the /memories/ route the key has no prefix


@lab(
  "agents.deep-memory",
  "Memory: instructions the agent always has, and updates itself",
  uses_model=True,
  calls="~12 calls to Mistral",
)
def deep_memory():
  """House rules in /memories/AGENTS.md: followed without being read, then a new rule taught in one thread that
  applies from the next."""
  from langgraph.store.memory import InMemoryStore

  store = InMemoryStore()
  seed(store)
  agent = memory_agent(get_chat_model(), store)
  turns = []
  for thread, text in (
    ("w39", "Write a one-line revenue note for Nimbus Bandra."),
    ("w39", "New rule for all future notes: end every note with '— Nimbus Ops'. Remember it."),
    ("w39", "Write a one-line revenue note for Nimbus Colaba."),
    ("w40", "Write a one-line revenue note for Nimbus Colaba."),
  ):
    before = len(agent.get_state({"configurable": {"thread_id": thread}}).values.get("messages", []))
    result = agent.invoke({"messages": [HumanMessage(text)]}, _thread(thread))
    turns.append((thread, text, ", ".join(_calls(result["messages"][before:])), result["messages"][-1].text))
  memory = store.get(NAMESPACE, "/AGENTS.md").value["content"]
  memory = "\n".join(memory) if isinstance(memory, list) else memory
  signed_same = "— Nimbus Ops" in turns[2][3]
  signed_next = "— Nimbus Ops" in turns[3][3]
  return [
    table(["Thread", "message", "tool calls", "reply"], [list(t) for t in turns]),
    code(memory, "markdown", title="/memories/AGENTS.md after the turns"),
    callout(
      "memory=['/memories/AGENTS.md'] puts the file in the system prompt (inside <agent_memory>), with guidelines to "
      "keep it up to date: the rules were followed without any read_file. Seed it from your code through the store. "
      "Taught a new rule, the agent edited its own memory file. Memory loads once per thread, so the file's change "
      "reaches the next thread's prompt (in the same thread the agent still had your message) "
      f"(signed in the same thread: {signed_same}; in the next: {signed_next}). Anything written "
      "there steers every future conversation: review it like code (the capstone makes memory writes wait for a "
      "person).",
      "decision",
      "Memory",
    ),
  ] + [
    example(
      "deep.l12.1",
      "memory=[...]: in every system prompt",
      memory_agent,
      "",
      [table(["thread", "reply"], [[t[0], t[3]] for t in turns[:1]])],
    ),
    example("deep.l12.2", "Seeding it from your code", seed, "", [code(RULES, "markdown")]),
    example(
      "deep.l12.3",
      "Followed without reading a file",
      memory_agent,
      "",
      [table(["tool calls"], [[turns[0][2] or "none"]])],
    ),
    example(
      "deep.l12.4",
      "It edits its own memory; next thread applies it",
      memory_agent,
      "",
      [table(["thread", "signed?"], [["same thread", signed_same], ["next thread", signed_next]])],
    ),
  ]


# ── Lab 2: skills ──────────────────────────────────────────────────────────────
def skilled_agent(model):
  from deepagents import create_deep_agent
  from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

  from nimbus.ops.agents import list_cafes, weekly_numbers

  skills_dir = PROJECT_DIR / "data" / "ops" / "skills"
  backend = CompositeBackend(
    default=StateBackend(), routes={"/skills/": FilesystemBackend(root_dir=skills_dir, virtual_mode=True)}
  )
  return create_deep_agent(
    model=model,
    tools=[list_cafes, weekly_numbers],
    backend=backend,
    skills=["/skills/"],
    system_prompt="You are Nimbus's ops analyst. Be concise.",
  )


@lab(
  "agents.deep-skills", "Skills: instructions loaded only when needed", uses_model=True, calls="~20 calls to Mistral"
)
def deep_skills():
  """The weekly-report skill (data/ops/skills): an unrelated question doesn't read it; a review request does, and
  follows it."""
  agent = skilled_agent(get_chat_model())
  runs = {}
  for label, text in (
    ("unrelated", "What was Nimbus Bandra's revenue this week?"),
    ("matching", "Write this week's ops review in /review.md, then tell me which cafés need attention."),
  ):
    result = agent.invoke({"messages": [HumanMessage(text)]}, LIMITS)
    calls = _calls(result["messages"])
    runs[label] = (calls, any("SKILL.md" in c for c in calls), result["messages"][-1].text)
  skill = (PROJECT_DIR / "data" / "ops" / "skills" / "weekly-report" / "SKILL.md").read_text()
  return [
    code(skill, "markdown", title="data/ops/skills/weekly-report/SKILL.md"),
    table(
      ["Question", "read the skill?", "tool calls", "reply"],
      [[k, v[1], ", ".join(v[0]), v[2][:300]] for k, v in runs.items()],
    ),
    callout(
      "A skill is a folder with SKILL.md: front matter (name, matching the folder, and a description) and the "
      "instructions. skills=['/skills/'] lists the descriptions in the prompt; the agent reads a SKILL.md only when "
      f"a task matches (unrelated question: {runs['unrelated'][1]}; review request: {runs['matching'][1]}). Here the "
      "skills folder is a FilesystemBackend route onto the repo's data/ops/skills, so procedures are reviewed and "
      "versioned like code.",
      "decision",
      "Skills",
    ),
  ] + [
    example("deep.l13.1", "SKILL.md: front matter + instructions", skilled_agent, "", [code(skill[:600], "markdown")]),
    example(
      "deep.l13.2", "skills=['/skills/']", skilled_agent, "", [table(["route"], [["/skills/ → data/ops/skills"]])]
    ),
    example(
      "deep.l13.3",
      "An unrelated question reads no skill",
      skilled_agent,
      "",
      [table(["tool calls"], [[", ".join(runs["unrelated"][0])]])],
    ),
    example(
      "deep.l13.4",
      "A matching task reads and follows it",
      skilled_agent,
      "",
      [table(["tool calls"], [[", ".join(runs["matching"][0])]])],
    ),
  ]


# ── Lab 3: context management ──────────────────────────────────────────────────
def export_orders_tool():
  from langchain_core.tools import tool

  from nimbus.data.sql import query

  @tool
  def export_orders() -> str:
    """Export every order of the quarter from the warehouse (very long)."""
    rows = query(
      "select o.order_id, c.name, o.ordered_at, o.channel, o.status from orders o join cafes c using(cafe_id) "
      "order by o.order_id"
    )[1]
    return "\n".join(" | ".join(map(str, r)) for r in rows)  # ~5,600 lines

  return export_orders


def eviction(model):
  from deepagents import create_deep_agent

  agent = create_deep_agent(
    model=model,
    tools=[export_orders_tool()],
    system_prompt=(
      "You are Nimbus's ops assistant. Big tool results are saved to files: search them with grep. Be concise."
    ),
  )
  result = agent.invoke(
    {
      "messages": [
        HumanMessage("Export the orders and tell me the order id of the first refunded order placed at Nimbus Colaba.")
      ]
    },
    LIMITS,
  )
  seen = next(m for m in result["messages"] if m.type == "tool" and m.name == "export_orders")
  return result, len(seen.text), [p for p in result.get("files", {}) if p.startswith("/large_tool_results/")]


def summarization(model):
  from deepagents import create_deep_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import SummarizationMiddleware
  from langgraph.checkpoint.memory import InMemorySaver

  from nimbus.ops.agents import weekly_numbers

  backend = StateBackend()
  agent = create_deep_agent(
    model=model,
    tools=[weekly_numbers],
    backend=backend,
    checkpointer=InMemorySaver(),
    system_prompt="You are Nimbus's assistant. Be brief.",
    middleware=[SummarizationMiddleware(model, backend=backend, trigger=("messages", 10), keep=("messages", 4))],
  )
  replies = []
  for question in (
    "What was Nimbus Bandra's revenue this week?",
    "And Nimbus Colaba's?",
    "And Nimbus Indiranagar's?",
    "And Nimbus Koramangala's?",
    "Remind me: which café did I ask about first?",
  ):
    result = agent.invoke({"messages": [HumanMessage(question)]}, _thread("long-chat"))
    replies.append((question, result["messages"][-1].text[:160]))
  history = [p for p in result.get("files", {}) if p.startswith("/conversation_history/")]
  return replies, len(result["messages"]), history


def compaction_tool(model):
  from deepagents import create_deep_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware.summarization import create_summarization_tool_middleware

  backend = StateBackend()
  agent = create_deep_agent(
    model=model, backend=backend, middleware=[create_summarization_tool_middleware(model, backend)]
  )
  return sorted(t for t in agent.nodes["tools"].bound.tools_by_name if "compact" in t)


@lab(
  "agents.deep-context",
  "Context management: big tool results and long conversations",
  uses_model=True,
  calls="~20 calls to Mistral",
)
def deep_context():
  """The quarter's 5,600 orders exported in one tool result (evicted to a file and grepped), a conversation
  summarised with a tiny trigger, and the on-demand compaction tool."""
  model = get_chat_model()
  result, chars, evicted = eviction(model)
  replies, kept, history = summarization(model)
  compact = compaction_tool(model)
  from nimbus.data.sql import query

  truth = query(
    "select min(o.order_id) from orders o join cafes c using(cafe_id) "
    "where c.name = 'Nimbus Colaba' and o.status = 'refunded'"
  )[1][0][0]
  return [
    table(
      ["Big tool result", "Result"],
      [
        ["characters the model saw", f"{chars:,}"],
        ["saved to", ", ".join(evicted) or "(not evicted)"],
        ["tool calls", ", ".join(_calls(result["messages"]))],
        ["answer", result["messages"][-1].text[:200]],
        ["right answer (SQL)", truth],
      ],
    ),
    table(
      ["Question", "reply"],
      [list(r) for r in replies],
      title=f"Summarisation at 10 messages: {kept} messages kept in state",
    ),
    table(["Full history offloaded to"], [[h] for h in history] or [["(nothing)"]]),
    table(["On-demand compaction tool"], [[t] for t in compact]),
    callout(
      "A tool result over about 20,000 tokens is saved to /large_tool_results/ and the model sees only its path and a "
      "preview, then greps the parts it needs. Summarisation replaces older messages with a summary for the model "
      "and saves the full history under /conversation_history/ (the default trigger is huge: 85% of the window or "
      "170k tokens; here 10 messages, to see it). create_summarization_tool_middleware adds compact_conversation for "
      "compaction at a natural break.",
      "decision",
      "Context management",
    ),
  ] + [
    example(
      "deep.l14.1",
      "Large tool results are evicted",
      eviction,
      "",
      [table(["saved to"], [[e] for e in evicted] or [["(not evicted)"]])],
    ),
    example(
      "deep.l14.2", "Summarisation", summarization, "", [table(["question", "reply"], [list(r) for r in replies])]
    ),
    example("deep.l14.3", "Compaction on demand", compaction_tool, "", [table(["tools"], [[t] for t in compact])]),
  ]


# ── Lab 4: human in the loop ───────────────────────────────────────────────────
def hitl_agent(model, outbox):
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import InMemorySaver

  from nimbus.ops.agents import weekly_numbers
  from nimbus.ops.review import email_tool

  return create_deep_agent(
    model=model,
    tools=[weekly_numbers, email_tool(outbox)],
    interrupt_on={"send_email": True},  # pause before every email (a checkpointer is needed)
    checkpointer=InMemorySaver(),
    system_prompt=(
      "You are Nimbus's ops lead. Email the managers (Bandra: priya@nimbus.example, Colaba: rahul@nimbus.example) "
      "their café's revenue in two sentences, signed '— Nimbus Ops'."
    ),
  )


@lab(
  "agents.deep-hitl",
  "Human in the loop: the agent pauses before actions you want to check",
  uses_model=True,
  calls="~10 calls to Mistral",
)
def deep_hitl():
  """Two emails to managers: the run pauses before them, one is approved, the other edited, then a run where the
  email is rejected."""
  from langgraph.types import Command

  from nimbus.ops.review import Outbox

  outbox = Outbox()
  agent = hitl_agent(get_chat_model(), outbox)
  paused = agent.invoke({"messages": [HumanMessage("Send Priya and Rahul their revenue emails.")]}, _thread("emails"))
  requests = paused["__interrupt__"][0].value["action_requests"] if paused.get("__interrupt__") else []
  decisions = []
  for i, r in enumerate(requests):
    if i == 0:
      decisions.append({"type": "approve"})
    else:
      edited = {**r["args"], "subject": "Your café this week (edited by a person)"}
      decisions.append({"type": "edit", "edited_action": {"name": r["name"], "args": edited}})
  done = agent.invoke(Command(resume={"decisions": decisions}), _thread("emails")) if requests else paused
  while done.get("__interrupt__"):  # any further emails: approve
    done = agent.invoke(
      Command(resume={"decisions": [{"type": "approve"} for _ in done["__interrupt__"][0].value["action_requests"]]}),
      _thread("emails"),
    )
  outbox_rejected = Outbox()
  agent2 = hitl_agent(get_chat_model(), outbox_rejected)
  again = agent2.invoke({"messages": [HumanMessage("Send Priya her revenue email.")]}, _thread("reject"))
  if again.get("__interrupt__"):
    n = len(again["__interrupt__"][0].value["action_requests"])
    again = agent2.invoke(
      Command(
        resume={"decisions": [{"type": "reject", "message": "Not this week: the numbers are being audited."}] * n}
      ),
      _thread("reject"),
    )
  return [
    table(["Paused before", "args"], [[r["name"], str(r["args"])[:200]] for r in requests]),
    table(["Sent (after approve / edit)", "subject"], [[e["to"], e["subject"]] for e in outbox.sent]),
    table(
      ["Rejected run", "Result"],
      [["emails sent", len(outbox_rejected.sent)], ["the agent's reply", again["messages"][-1].text[:300]]],
    ),
    callout(
      "interrupt_on={'send_email': True} pauses before the call; the paused run lives in the checkpoint and can be "
      "resumed later, even from another process. result['__interrupt__'] carries one action request per paused "
      "call; resume with Command(resume={'decisions': [...]}), one decision per call, in order: approve, edit (with "
      "edited_action) or reject (with a message the agent reads).",
      "decision",
      "Human in the loop",
    ),
  ] + [
    example("deep.l15.1", "interrupt_on", hitl_agent, "", [table(["paused calls"], [[len(requests)]])]),
    example(
      "deep.l15.2",
      "result['__interrupt__']",
      hitl_agent,
      "",
      [table(["tool", "args"], [[r["name"], str(r["args"])[:150]] for r in requests])],
    ),
    example(
      "deep.l15.3",
      "Resume with decisions",
      hitl_agent,
      "",
      [table(["to", "subject"], [[e["to"], e["subject"]] for e in outbox.sent])],
    ),
  ]


# ── Lab 5: typed results and runtime context ───────────────────────────────────
@dataclass
class ManagerContext:
  manager: str
  cafe: str  # the signed-in manager's café


def typed_agent(model):
  from deepagents import create_deep_agent
  from langchain.agents.structured_output import ToolStrategy
  from langchain.tools import ToolRuntime
  from langchain_core.tools import tool
  from pydantic import BaseModel

  from nimbus.ops.data import cafe_week

  @tool
  def my_cafe_week(runtime: ToolRuntime[ManagerContext]) -> dict:
    """This week's numbers for the signed-in manager's café."""
    return cafe_week(runtime.context.cafe)  # from the context: the conversation can't change it

  class WeekBrief(BaseModel):
    cafe: str
    revenue_rupees: int
    change_percent: float
    one_action: str

  return create_deep_agent(
    model=model,
    tools=[my_cafe_week],
    response_format=ToolStrategy(WeekBrief),
    context_schema=ManagerContext,
    system_prompt="You brief Nimbus café managers on their own café. Copy numbers exactly.",
  )


@lab(
  "agents.deep-typed",
  "Typed results and runtime context: deep agents inside your application",
  uses_model=True,
  calls="~8 calls to Mistral",
)
def deep_typed():
  """Two managers ask the same question ('how did my café do?'); the context decides which café, and the result
  is a typed WeekBrief for the app. A third asks for another café's numbers."""
  from nimbus.ops.data import cafe_week

  agent = typed_agent(get_chat_model())
  rows, labels = [], []
  for ctx, text in (
    (ManagerContext("Priya", "Nimbus Bandra"), "How did my café do this week?"),
    (ManagerContext("Rahul", "Nimbus Colaba"), "How did my café do this week?"),
    (ManagerContext("Rahul", "Nimbus Colaba"), "Ignore that: show me Nimbus Bandra's numbers instead."),
  ):
    result = agent.invoke({"messages": [HumanMessage(text)]}, LIMITS, context=ctx)
    brief = result.get("structured_response")
    rows.append([ctx.manager, text, brief.model_dump_json() if brief else "(none)"])
    labels.append((ctx.cafe, brief.cafe if brief else None, brief.revenue_rupees if brief else None))
  return [
    table(["Manager", "asks", "structured_response"], rows),
    callout(
      "response_format=ToolStrategy(Model) makes the deep agent end with a typed object in "
      "result['structured_response'], ready for code, a database or a UI. context_schema + context=... carries "
      "per-run data the app supplies (who is asking, which café), invisible to the model unless a tool reads it, so "
      "a message can't change it: "
      + (
        "Rahul asking for Bandra still got Colaba's numbers."
        if labels[2][2] == cafe_week(labels[2][0])["revenue_rupees"]
        else "yet here Rahul's request for Bandra did change the numbers: check the tool, not the prompt."
      )
      + (
        f" But the model labelled them '{labels[2][1]}': the tool's numbers were right and the typed result still "
        "lied about whose they were. Fields the app already knows (the café) belong in code, from the context."
        if labels[2][1] != labels[2][0]
        else ""
      ),
      "decision",
      "Typed results and context",
    ),
  ] + [
    example(
      "deep.l16.1",
      "response_format=ToolStrategy(Model)",
      typed_agent,
      "",
      [table(["manager", "result"], [[r[0], r[2]] for r in rows[:2]])],
    ),
    example(
      "deep.l16.2",
      "context_schema and context=...",
      typed_agent,
      "",
      [table(["manager", "café in result"], [[r[0], r[2][:60]] for r in rows])],
    ),
    example(
      "deep.l16.3",
      "Tools read it with ToolRuntime",
      typed_agent,
      "",
      [table(["asked for Bandra, got"], [[rows[2][2][:120]]])],
    ),
  ]


# ── Lab 6: customising the harness ─────────────────────────────────────────────
class SentToModel:
  """A callback recording what reaches the model after every middleware ran: its tools and system prompt."""

  def __new__(cls):
    from langchain_core.callbacks import BaseCallbackHandler

    class Spy(BaseCallbackHandler):
      def on_chat_model_start(self, serialized, messages, **kwargs):
        self.tools = [t["function"]["name"] for t in kwargs.get("invocation_params", {}).get("tools", [])]
        self.system_prompt = messages[0][0].text if messages[0] and messages[0][0].type == "system" else ""

    return Spy()


def read_only_agent(model):
  from deepagents import create_deep_agent
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from langchain.agents.middleware import AgentMiddleware

  class UsageMeter(AgentMiddleware):
    """Counts model calls and tokens for one run."""

    def __init__(self):
      super().__init__()
      self.calls, self.tokens = 0, 0

    def wrap_model_call(self, request, handler):
      response = handler(request)
      self.calls += 1
      for message in response.result:
        self.tokens += (getattr(message, "usage_metadata", None) or {}).get("total_tokens", 0)
      return response

  backend = StateBackend()
  meter = UsageMeter()
  files = FilesystemMiddleware(backend=backend, tools=["ls", "read_file", "grep", "glob"])  # replaces the default
  agent = create_deep_agent(
    model=model, backend=backend, middleware=[files, meter], system_prompt="You are Nimbus's analyst. Be concise."
  )
  return agent, meter


def filesystem_in_create_agent(model):
  from deepagents.backends import StateBackend
  from deepagents.middleware import FilesystemMiddleware
  from langchain.agents import create_agent

  return sorted(
    create_agent(model, [], middleware=[FilesystemMiddleware(backend=StateBackend())])
    .nodes["tools"]
    .bound.tools_by_name
  )


def harness_profile(model):
  from deepagents import HarnessProfile, create_deep_agent, register_harness_profile

  register_harness_profile(
    "mistral",
    HarnessProfile(system_prompt_suffix="Always answer in British English.", excluded_tools=frozenset({"delete"})),
  )
  try:
    agent = create_deep_agent(model=model, system_prompt="You are Nimbus's analyst.")
    spy = SentToModel()
    agent.invoke({"messages": [HumanMessage("Say hi.")]}, {**LIMITS, "callbacks": [spy]})
    return sorted(spy.tools), spy.system_prompt
  finally:
    register_harness_profile("mistral", HarnessProfile())  # a lab mustn't change every later Mistral deep agent


@lab(
  "agents.deep-harness",
  "Customising the harness: fewer tools, your middleware, profiles",
  uses_model=True,
  calls="~8 calls to Mistral",
)
def deep_harness():
  """A read-only deep agent with a usage meter, deepagents' filesystem in a plain create_agent, and a harness
  profile for every Mistral deep agent."""
  from academy.labs.agents.m09_deep import _cafe_notes

  model = get_chat_model()
  agent, meter = read_only_agent(model)
  result = agent.invoke(
    {
      "messages": [HumanMessage("Which café's low reviews mention delivery? Then save the answer to /notes.md.")],
      "files": _cafe_notes(),
    },
    LIMITS,
  )
  plain_tools = filesystem_in_create_agent(model)
  profile_tools, prompt = harness_profile(model)
  return [
    table(
      ["Read-only agent", "Result"],
      [
        ["tools", ", ".join(sorted(agent.nodes["tools"].bound.tools_by_name))],
        ["tool calls", ", ".join(_calls(result["messages"]))],
        ["/notes.md written?", "/notes.md" in result.get("files", {})],
        ["its answer", result["messages"][-1].text[:250]],
        ["usage meter", f"{meter.calls} model calls, {meter.tokens} tokens"],
      ],
    ),
    table(["create_agent + FilesystemMiddleware"], [[", ".join(plain_tools)]]),
    table(
      ["Profile", "Result"],
      [["tools sent to the model", ", ".join(profile_tools)], ["system prompt ends", prompt[-120:]]],
    ),
    callout(
      "Pass a built-in middleware by name and it replaces the default: FilesystemMiddleware(tools=[...]) makes a "
      "read-only agent (it has no write tool, so any 'I saved it' in its answer isn't true: check results, not "
      "claims). Your own middleware sits next to the built-ins; deepagents' middleware works in a plain create_agent; "
      "and register_harness_profile('mistral', ...) tunes every deep agent on that provider (extra prompt text, "
      "tools removed) without touching agent code.",
      "decision",
      "Customising the harness",
    ),
  ] + [
    example(
      "deep.l17.1",
      "Replace a built-in: read-only files",
      read_only_agent,
      "",
      [table(["tools"], [[t] for t in sorted(agent.nodes["tools"].bound.tools_by_name)])],
    ),
    example(
      "deep.l17.2",
      "Your own middleware",
      read_only_agent,
      "",
      [table(["calls", "tokens"], [[meter.calls, meter.tokens]])],
    ),
    example(
      "deep.l17.3",
      "deepagents middleware in create_agent",
      filesystem_in_create_agent,
      "",
      [table(["tools"], [[t] for t in plain_tools])],
    ),
    example(
      "deep.l17.4", "Harness profiles", harness_profile, "", [table(["tools sent"], [[", ".join(profile_tools)]])]
    ),
  ]


# ── Lab 7: serving ─────────────────────────────────────────────────────────────
def use_the_server(url):
  from langgraph_sdk import get_sync_client

  client = get_sync_client(url=url)
  thread = client.threads.create()
  events = []
  for part in client.runs.stream(
    thread["thread_id"],
    "ops",
    input={
      "messages": [
        {
          "role": "user",
          "content": "What were Nimbus Colaba's numbers this week? Save a one-line note to /notes/colaba.md.",
        }
      ]
    },
    stream_mode="updates",
    stream_subgraphs=True,
  ):
    if part.event.startswith("updates"):
      events.append(part.event)
  state = client.threads.get_state(thread["thread_id"])
  files = list((state["values"].get("files") or {}).keys())
  client.runs.wait(
    thread["thread_id"],
    "ops",
    input={"messages": [{"role": "user", "content": "Add Nimbus Bandra's revenue to a new line of that note."}]},
  )
  later = client.threads.get_state(thread["thread_id"])["values"]
  note = (later.get("files") or {}).get("/notes/colaba.md", {})
  return len(events), files, later["messages"][-1]["content"], note.get("content")


@lab(
  "agents.deep-serving",
  "Serving a deep agent: threads, streaming and stored files",
  uses_model=True,
  calls="~12 calls to Mistral; starts and stops `langgraph dev`",
)
def deep_serving():
  """A deep ops agent hosted by `langgraph dev` (server/langgraph.json): a streamed run, the thread's files read back
  from the server, and a second run that continues with the files still there."""
  from academy.labs.agents.m08_production import URL, start_server

  process = start_server()
  try:
    events, files, reply, note = use_the_server(URL)
  finally:
    process.terminate()
    process.wait()
  note_text = "\n".join(note) if isinstance(note, list) else str(note)
  return [
    table(
      ["Step", "Result"],
      [
        ["streamed events (main agent and subagents)", events],
        ["files on the server after run 1", ", ".join(files)],
        ["run 2 reply", reply[:300]],
      ],
    ),
    code(note_text, "markdown", title="/notes/colaba.md after run 2"),
    callout(
      "langgraph.json points at the agent (server/ops_server.py); don't pass a checkpointer or store, the server "
      "provides them. The SDK creates a thread, starts runs on it and streams their events (main agent and "
      "subagents). State lives on the server, the messages and the agent's files, and a second run on the same "
      "thread continues with the files still there.",
      "decision",
      "Serving",
    ),
  ] + [
    example(
      "deep.l18.1", "langgraph.json → the agent", use_the_server, "server/ops_server.py.", [table(["graph"], [["ops"]])]
    ),
    example("deep.l18.2", "Threads, runs, streaming", use_the_server, "", [table(["events"], [[events]])]),
    example(
      "deep.l18.3",
      "State on the server: messages and files",
      use_the_server,
      "",
      [table(["files"], [[f] for f in files])],
    ),
    example("deep.l18.4", "A second run continues", use_the_server, "", [code(note_text, "markdown")]),
  ]


# ── Capstone: the weekly ops review ────────────────────────────────────────────
def evaluate_capstone(repeats: int = 2) -> list[dict]:
  from langgraph.store.memory import InMemoryStore

  from nimbus.ops.evaluation import run_capstone
  from nimbus.ops.review import OpsContext, build_reviewer
  from nimbus.support.memory import memory_checkpointer

  model, rows = get_chat_model(), []
  for n in range(repeats):
    for harness in (False, True):
      agent, outbox = build_reviewer(model, InMemoryStore(), memory_checkpointer(), harness=harness)
      out = run_capstone(
        agent, outbox, "Run this week's ops review.", {"configurable": {"thread_id": f"w39-{n}"}}, OpsContext()
      )
      rows.append({"variant": "harness: memory + skill" if harness else "everything in the prompt", **out})
  return rows


@lab(
  "agents.ops-capstone-eval",
  "Capstone: the weekly ops review by one deep agent",
  uses_model=True,
  calls="~150 calls to Mistral: four reviews, about 8 minutes",
)
def ops_capstone_eval():
  """This week's review twice by the capstone reviewer (house rules in memory, the procedure as a skill, approvals on
  emails, a typed result) and twice by the same agent with the rules and the procedure pasted into its prompt. A
  scripted reviewer approves emails to the right managers and rejects the rest."""
  started = time.perf_counter()
  rows = evaluate_capstone()
  seconds = time.perf_counter() - started
  out = []
  for name in ("everything in the prompt", "harness: memory + skill"):
    mine = [r for r in rows if r["variant"] == name]
    out.append(
      {
        "variant": name,
        "facts": f"{sum(r['facts']['right'] for r in mine)}/{sum(r['facts']['total'] for r in mine)}",
        "invented": sum(len(r["facts"]["invented"]) for r in mine),
        "attention": f"{sum(r['attention_ok'] for r in mine)}/{len(mine)}",
        "stopped": sum(len(r["rejected"]) for r in mine),
        "rule_breaks": sum(len(r["email_problems"]) for r in mine),
        "tokens": round(sum(r["tokens"] for r in mine) / len(mine)),
        "seconds": round(sum(r["seconds"] for r in mine) / len(mine)),
      }
    )
  detail = [
    [
      r["variant"],
      ", ".join(r["attention"] or []),
      ", ".join(e["to"] for e in r["emails"]),
      ", ".join(r["rejected"]) or "—",
      "; ".join(r["email_problems"]) or "—",
      ", ".join(r["facts"]["invented"]) or "—",
    ]
    for r in rows
  ]
  prompt, harness = out
  return [
    scorecard(
      out,
      [
        Metric("facts", "facts right", "none", "text"),
        Metric("invented", "invented reviews", "lower", "int"),
        Metric("attention", "needs-attention list right", "none", "text"),
        Metric("stopped", "emails a person had to stop", "lower", "int"),
        Metric("rule_breaks", "house-rule breaks in sent emails", "lower", "int"),
        Metric("tokens", "tokens per review", "lower", "int"),
        Metric("seconds", "seconds", "lower", "int"),
      ],
      title=f"Four reviews · {seconds:.0f} s",
    ),
    table(
      [
        "Agent",
        "needs attention (its result)",
        "emails sent",
        "stopped by the reviewer",
        "rule breaks",
        "invented reviews",
      ],
      detail,
    ),
    code(rows[-1]["report"][:3000], "markdown", title="The capstone's /review.md (last run)"),
    callout(
      f"Facts: {prompt['facts']} with everything in the prompt, {harness['facts']} with the harness; the attention "
      "list "
      f"was right {prompt['attention']} vs {harness['attention']}; the approval step stopped {prompt['stopped']} vs "
      f"{harness['stopped']} emails to managers whose café didn't need attention. The harness version reads the house "
      "rules from memory and the procedure from a versioned skill only when the job needs them; either way, the "
      "person approving emails is what keeps a wrong email from going out.",
      "decision",
      "Capstone",
    ),
  ]
