"""Agents with LangGraph · Module 3 labs: designing a graph from a business process, and the workflow patterns:
prompt chaining with a gate, routing, parallelisation, orchestrator-workers and evaluator-optimizer.
Then: support graph v3 (specialised handlers) against v2."""

import json
import operator
import re
import time
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from academy.labs import cache
from academy.labs.runtime import Metric, answer, callout, code, columns, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model


@lru_cache
def _retriever():
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.rag.retriever import Retriever

  s = get_settings()
  return Retriever(s, get_embeddings(s), get_chat_model())


# ── Lab 1: thinking in LangGraph ────────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.

DESIGN = [  # node, kind, what it does, error strategy
  ("triage", "LLM", "category, urgency, language of the ticket", "structured output, retried once on a bad parse"),
  ("look_up_order", "data", "the order record for each order id", "no model; a missing order is a fact, not an error"),
  ("refund_facts", "data", "the refund decision, by the policy, in code", "deterministic: tested, never guessed"),
  ("knowledge_facts", "data", "public documents that answer a question", "the RAG retriever (hybrid, reranked)"),
  ("draft_reply", "LLM", "a reply from the facts in state", "loop back with feedback when the check fails"),
  ("check_reply", "code", "order status stated, no claimed actions, no links", "after 2 drafts: flag for a person"),
  ("safety_reply", "person", "the approved message; a person calls", "never drafted by a model"),
]


def map_the_process():
  steps = [node for node, *_ in DESIGN]  # 1. small steps, each a node
  kinds = {node: kind for node, kind, *_ in DESIGN}  # 2. LLM, data, code, or a person
  return steps, kinds


def design_the_state():
  from nimbus.support.graph import SupportStateV3

  return list(
    SupportStateV3.__annotations__
  )  # 3. raw data only (ticket, triage, orders, facts): prompts are built inside nodes


def error_strategy_per_kind():
  return {node: strategy for node, _, _, strategy in DESIGN}  # 4. retry, loop back, ask a person, or let it crash


def wire_draw_and_run(graph, tickets):
  runs = {t: graph.invoke({"ticket": t, "history": []}) for t in tickets}  # 5. real cases through the drawn graph
  return runs


@lab(
  "agents.thinking",
  "Thinking in LangGraph: from a business process to a graph",
  uses_model=True,
  calls="~8 calls to Mistral",
)
def thinking():
  """The production support graph designed the five-step way, then run on three real tickets."""
  from nimbus.support.graph import support_graph

  steps, kinds = map_the_process()
  state_keys = design_the_state()
  strategies = error_strategy_per_kind()
  graph = support_graph()
  tickets = [
    "Please cancel order NC-55021, I placed it twice by mistake!",
    "Is Chikmagalur Classic good for cold brew, and do you sell 1 kg bags?",
    "Found a small piece of plastic in my cold brew bottle from the Bandra café.",
  ]
  runs = wire_draw_and_run(graph, tickets)
  return [
    table(
      ["Node", "Kind", "Does", "When it fails"], [list(d) for d in DESIGN], title="The design, written before the code"
    ),
    code(graph.get_graph().draw_mermaid(), "text", title="The production support graph (v3)"),
    table(
      ["Ticket", "Path", "Reply", "for a person?"],
      [
        [t, " → ".join(s.split(":")[0] for s in r["steps"]), r["reply"], "yes" if r.get("needs_review") else "no"]
        for t, r in runs.items()
      ],
    ),
    callout(
      "Design before code: split the process into steps, name each step's kind (a model, data, code, or a person), "
      "keep only raw data in the state (prompts are built inside nodes, so they can change without migrating "
      "state), and decide per kind what happens when it fails. Then wire it, draw it, and push real cases through. "
      "Every other pattern in this module is a shape this design can take.",
      "decision",
      "Design first",
    ),
  ] + [
    example(
      "lg.l15.1", "Map the process into small steps", map_the_process, "", [table(["step"], [[s] for s in steps])]
    ),
    example("lg.l15.2", "Name each step's kind", map_the_process, "", [table(["step", "kind"], list(kinds.items()))]),
    example(
      "lg.l15.3", "State: raw data only", design_the_state, "", [table(["state keys"], [[", ".join(state_keys)]])]
    ),
    example(
      "lg.l15.4",
      "An error strategy per kind",
      error_strategy_per_kind,
      "",
      [table(["step", "strategy"], list(strategies.items()))],
    ),
    example(
      "lg.l15.5",
      "Wire, draw, run real cases",
      wire_draw_and_run,
      "",
      [table(["ticket", "steps"], [[t, " | ".join(r["steps"])] for t, r in runs.items()])],
    ),
  ]


# ── Lab 2: prompt chaining with a gate ──────────────────────────────────────────
BLEND = "Coorg Dawn: a medium roast from Coorg with notes of jaggery and orange peel, 250 g for 640 rupees"


class Outline(TypedDict, total=False):
  brief: str
  outline: str
  gate: str
  draft: str
  final: str


def chain_with_a_gate(model):
  from langgraph.graph import END, START, StateGraph

  def outline(state):  # step 1: a small, focused prompt
    return {
      "outline": cache.text(
        model,
        f"Outline a launch email about this coffee in exactly 3 lines, each starting with '- '. No headings, no "
        f"sub-bullets.\n{state['brief']}",
      )
    }

  def gate(state):  # plain code between steps: stop instead of polishing a bad outline
    bullets = [line for line in state["outline"].splitlines() if line.startswith("- ")]
    ok = len(bullets) == 3 and "640" in state["outline"]
    return {
      "gate": "pass"
      if ok
      else f"fail: {len(bullets)} bullets, price {'in' if '640' in state['outline'] else 'missing'}"
    }

  def draft(state):  # step 2 builds on step 1
    return {"draft": cache.text(model, f"Write the email (under 120 words) from this outline:\n{state['outline']}")}

  def polish(state):  # step 3
    return {
      "final": cache.text(model, f"Make this email warmer and add a one-line subject at the top:\n{state['draft']}")
    }

  graph = StateGraph(Outline)
  graph.add_node("outline", outline)
  graph.add_node("gate", gate)
  graph.add_node("draft", draft)
  graph.add_node("polish", polish)
  graph.add_edge(START, "outline")
  graph.add_edge("outline", "gate")
  graph.add_conditional_edges("gate", lambda s: "draft" if s["gate"] == "pass" else END, ["draft", END])
  graph.add_edge("draft", "polish")
  graph.add_edge("polish", END)
  return graph.compile()


def one_big_prompt(model, brief):
  return cache.text(
    model,
    "Write a warm launch email (under 120 words, one-line subject at the top) about this coffee, mentioning its "
    f"price:\n{brief}",
  )


@lab(
  "agents.prompt-chaining",
  "Prompt chaining with a gate",
  uses_model=True,
  calls="~5 calls to Mistral (cached after the first run)",
)
def prompt_chaining():
  """A launch email as three focused steps with a code gate after the first, against one big prompt."""
  model = get_chat_model()
  graph = chain_with_a_gate(model)
  result = graph.invoke({"brief": BLEND})
  bad = graph.invoke({"brief": "a new coffee (no details yet)"})
  single = one_big_prompt(model, BLEND)

  def checks(text):
    return {
      "price stated": "640" in text,
      "under 140 words": len(text.split()) < 140,
      "has a subject": text.lower().lstrip("*# ").startswith("subject"),
      "no [placeholders]": "[" not in text,
    }

  final = result.get("final", "")
  return [
    columns(
      [
        answer(final or "(stopped at the gate)", "chained: outline → gate → draft → polish"),
        answer(single, "one big prompt"),
      ]
    ),
    table(
      ["Check", "chained", "one prompt"],
      [[k, "✓" if v else "✗", "✓" if checks(single)[k] else "✗"] for k, v in checks(final).items()],
    ),
    table(
      ["Brief", "gate", "steps run"],
      [
        [BLEND[:40] + "…", result["gate"], ", ".join(k for k in ("outline", "draft", "final") if result.get(k))],
        [
          "a new coffee (no details yet)",
          bad["gate"],
          ", ".join(k for k in ("outline", "draft", "final") if bad.get(k)),
        ],
      ],
    ),
    callout(
      "Chaining splits a task into fixed steps, each a small prompt you can inspect and test. The gate is the "
      "point: plain code between steps stops the run when the outline has no price, instead of polishing a draft "
      "that was wrong from the start (the second brief stopped there, saving two calls). One big prompt is cheaper "
      "when it works and harder to debug when it doesn't.",
      "decision",
      "Prompt chaining",
    ),
  ] + [
    example(
      "lg.l16.1",
      "Fixed steps, each building on the last",
      chain_with_a_gate,
      "",
      [code(result.get("outline", ""), "markdown", "step 1: outline")],
    ),
    example(
      "lg.l16.2",
      "A gate in plain code",
      chain_with_a_gate,
      "",
      [table(["brief", "gate"], [["Coorg Dawn", result["gate"]], ["no details", bad["gate"]]])],
    ),
    example(
      "lg.l16.3",
      "Why chaining helps",
      one_big_prompt,
      "Every step's output is in the state: you can see where it went wrong.",
      [table(["step", "characters"], [[k, len(result.get(k, ""))] for k in ("outline", "draft", "final")])],
    ),
  ]


# ── Lab 3: routing to specialised handlers ──────────────────────────────────────
TICKETS = {
  "I bought an unopened bag last week and want to return it (NC-4410).": "refund_return",
  "Which blend is best for a moka pot? I like chocolatey coffee.": "product_question",
  "Can I still cancel NC-55021? I ordered it twice.": "cancel_order",
}


class RouteState(TypedDict, total=False):
  ticket: str
  route: str
  reply: str


def a_router(model):
  from langgraph.graph import END, START, StateGraph

  class Route(BaseModel):
    route: Literal["refunds", "products", "orders"]

  def classify(state):
    return {
      "route": cache.structured(
        model, Route, f"Route this support ticket: refunds, products or orders.\n{state['ticket']}"
      ).route
    }

  def handler(name, facts):  # each route: its own focused prompt AND only the facts it needs
    def run(state):
      return {
        "reply": cache.text(
          model, f"You handle {name} for Nimbus Coffee. Facts: {facts}\nReply in two sentences to: {state['ticket']}"
        )
      }

    return run

  graph = StateGraph(RouteState)
  graph.add_node("classify", classify)
  graph.add_node(
    "refunds", handler("refunds", "unopened bags can be returned within 14 days; a teammate confirms refunds")
  )
  graph.add_node(
    "products",
    handler(
      "product questions",
      "Monsoon Malabar Reserve: dark roast, chocolate and spice, best for espresso; Chikmagalur Classic: medium, "
      "caramel",
    ),
  )
  graph.add_node("orders", handler("orders", "orders can be cancelled free until packed"))
  graph.add_edge(START, "classify")
  graph.add_conditional_edges("classify", lambda s: s["route"], ["refunds", "products", "orders"])
  for name in ("refunds", "products", "orders"):
    graph.add_edge(name, END)
  return graph.compile()


@lab(
  "agents.routing",
  "Routing to specialised handlers",
  uses_model=True,
  calls="~6 calls to Mistral (cached after the first run)",
)
def routing():
  """Three tickets classified into a fixed set of routes, each answered by a handler with its own prompt and facts."""
  model = get_chat_model()
  graph = a_router(model)
  runs = {t: graph.invoke({"ticket": t}) for t in TICKETS}
  # the orders handler's facts don't include the order's status: does its reply claim one anyway?
  guessed = [
    r for r in runs.values() if r["route"] == "orders" and re.search(r"\b(packed|shipped|dispatched)\b", r["reply"])
  ]
  note = (
    " Fixed facts aren't enough when the answer depends on the ticket: the orders handler was never told this "
    "order's status, yet its reply states one. v3 below looks the order up first."
    if guessed
    else ""
  )
  return [
    table(["Ticket", "route", "reply"], [[t, r["route"], r["reply"]] for t, r in runs.items()]),
    code(graph.get_graph().draw_mermaid(), "text", title="Every possible path; each run takes one"),
    callout(
      "One prompt for everything has to carry every rule and every fact, and each extra rule dilutes the others. "
      "Routing classifies first (structured output, a fixed set of routes), then each handler has a short prompt "
      "and only the facts its route needs. The production support graph v3 is this pattern, with the facts "
      "gathered by code and the RAG retriever instead of hard-coded." + note,
      "decision",
      "Routing",
    ),
  ] + [
    example(
      "lg.l17.1",
      "An LLM classifies into fixed routes",
      a_router,
      "",
      [table(["ticket", "route"], [[t, r["route"]] for t, r in runs.items()])],
    ),
    example(
      "lg.l17.2",
      "Each route: its own prompt and facts",
      a_router,
      "",
      [table(["route", "reply"], [[r["route"], r["reply"]] for r in runs.values()])],
    ),
    example(
      "lg.l17.3",
      "The graph shows every path; the run shows one",
      a_router,
      "",
      [code(graph.get_graph().draw_mermaid(), "text")],
    ),
  ]


# ── Lab 4: parallelisation ──────────────────────────────────────────────────────
REVIEW_TEXT = (
  "Nimbus Bandra: the oat latte was perfect and the staff were lovely, but we waited 25 minutes and the music was "
  "too loud to talk."
)


class SectionState(TypedDict, total=False):
  text: str
  sections: Annotated[list[str], operator.add]
  summary: str


def sectioning(model):
  from langgraph.graph import END, START, StateGraph

  def aspect(name):
    def run(state):
      return {
        "sections": [
          f"{name}: "
          + cache.text(model, f"In one short sentence, what does this review say about {name} ONLY?\n{state['text']}")
        ]
      }

    return run

  graph = StateGraph(SectionState)
  for name in ("drinks", "service", "atmosphere"):
    graph.add_node(name, aspect(name))
    graph.add_edge(START, name)  # independent parts, at the same time
  graph.add_node("combine", lambda s: {"summary": " | ".join(sorted(s["sections"]))})
  graph.add_edge(["drinks", "service", "atmosphere"], "combine")
  graph.add_edge("combine", END)
  return graph.compile()


class Vote(BaseModel):
  send: bool = Field(description="Is the reply safe to send to the customer as it is?")
  reason: str


class VoteState(TypedDict, total=False):
  reply: str
  votes: Annotated[list[tuple[str, bool, str]], operator.add]  # (judge, send?, reason)
  verdict: str


def voting(model):
  from langgraph.graph import END, START, StateGraph

  rules = {
    "policy": "it promises no refund, compensation or action that a teammate hasn't confirmed",
    "tone": "it is polite and calm",
    "facts": "it states no specific facts beyond what the customer wrote",
  }

  def judge(name, rule):
    def run(state):
      vote = cache.structured(model, Vote, f"Judge this support reply on ONE rule: {rule}.\nReply:\n{state['reply']}")
      return {"votes": [(name, vote.send, vote.reason)]}

    return run

  graph = StateGraph(VoteState)
  for name, rule in rules.items():
    graph.add_node(name, judge(name, rule))
    graph.add_edge(START, name)

  def count(state):
    yes = sum(send for _, send, _ in state["votes"])
    return {"verdict": f"{'send' if yes >= 2 else 'hold'} ({yes}/{len(state['votes'])} yes)"}

  graph.add_node("count", count)
  graph.add_edge(list(rules), "count")
  graph.add_edge("count", END)
  return graph.compile()


@lab(
  "agents.parallelization",
  "Parallelisation: sectioning and voting",
  uses_model=True,
  calls="~9 calls to Mistral (cached after the first run)",
)
def parallelization():
  """A café review split into three aspects at once, and three judges voting on whether two replies can be sent."""
  model = get_chat_model()
  sections = sectioning(model).invoke({"text": REVIEW_TEXT, "sections": []})
  graph = voting(model)
  replies = {
    "careful": "Thank you for letting us know. A teammate will look into your order and reply within one business day.",
    "risky": "So sorry! I've refunded your order and added 500 bonus points to your account. Enjoy a free drink on us!",
  }
  verdicts = {k: graph.invoke({"reply": v, "votes": []}) for k, v in replies.items()}
  dissent = [
    (k, judge) for k, v in verdicts.items() for judge, send, _ in v["votes"] if send != v["verdict"].startswith("send")
  ]
  overruled = (
    f" Here {len(dissent)} judge(s) disagreed with the majority ("
    + "; ".join(f"{j} on the {k} reply" for k, j in dissent)
    + ") and were outvoted."
    if dissent
    else " Here all three judges agreed on both replies."
  )
  return [
    table(
      ["Aspect summary"],
      [[s] for s in sorted(sections["sections"])],
      title="Sectioning: three aspects, three calls at the same time",
    ),
    table(
      ["Reply", "votes", "verdict"],
      [
        [replies[k], ", ".join(f"{j}: {'yes' if s else 'no'}" for j, s, _ in sorted(v["votes"])), v["verdict"]]
        for k, v in verdicts.items()
      ],
      title="Voting: three judges, majority wins",
    ),
    callout(
      "Sectioning splits a task into independent parts that run at once and are combined; voting asks the same "
      "question of several judges (or several times) and takes the majority, so one judge's mistake doesn't decide. "
      "Both are fan-out / fan-in graphs with a reducer collecting the parts or the votes." + overruled,
      "decision",
      "Parallelisation",
    ),
  ] + [
    example(
      "lg.l18.1", "Sectioning", sectioning, "", [table(["sections"], [[s] for s in sorted(sections["sections"])])]
    ),
    example(
      "lg.l18.2", "Voting", voting, "", [table(["reply", "verdict"], [[k, v["verdict"]] for k, v in verdicts.items()])]
    ),
    example(
      "lg.l18.3",
      "Fan-out / fan-in with a reducer",
      voting,
      "`votes` has operator.add: each judge appends its vote.",
      [code(graph.get_graph().draw_mermaid(), "text")],
    ),
  ]


# ── Lab 5: orchestrator and workers ─────────────────────────────────────────────
class Section(BaseModel):
  title: str
  question: str = Field(description="What this section must answer, from the documents")


class Plan(BaseModel):
  sections: list[Section] = Field(description="2 to 4 sections")


class ReportState(TypedDict, total=False):
  topic: str
  plan: list[dict]
  written: Annotated[list[str], operator.add]
  report: str


class WorkerInput(TypedDict):
  title: str
  question: str


def orchestrator_and_workers(model, retriever):
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Send

  def orchestrate(state):  # an LLM plans: how many sections isn't known in advance
    plan = cache.structured(
      model,
      Plan,
      f"Plan a short internal briefing on: {state['topic']}. 2-4 sections, each answerable from company documents.",
    )
    return {"plan": [s.model_dump() for s in plan.sections[:4]]}  # a cap: a model can't start fifty workers

  def assign(state):
    return [Send("write_section", s) for s in state["plan"]]  # one worker per section, in parallel

  def write_section(state: WorkerInput):
    _, hits = retriever.search(state["question"], k=2)
    facts = "\n".join(d.page_content for d, _ in hits)
    return {
      "written": [
        f"## {state['title']}\n"
        + cache.text(model, f"Using only these facts, answer in two sentences: {state['question']}\n\n{facts}")
      ]
    }

  def synthesise(state):
    return {"report": "\n\n".join(sorted(state["written"]))}

  graph = StateGraph(ReportState)
  graph.add_node("orchestrate", orchestrate)
  graph.add_node("write_section", write_section)
  graph.add_node("synthesise", synthesise)
  graph.add_edge(START, "orchestrate")
  graph.add_conditional_edges("orchestrate", assign, ["write_section"])
  graph.add_edge("write_section", "synthesise")
  graph.add_edge("synthesise", END)
  return graph.compile()


@lab(
  "agents.orchestrator",
  "Orchestrator and workers",
  uses_model=True,
  calls="~5–9 calls to Mistral (cached after the first run)",
)
def orchestrator():
  """A planner decides the sections of a briefing; one worker per section searches and writes; a synthesiser
  joins them."""
  model = get_chat_model()
  graph = orchestrator_and_workers(model, _retriever())
  started = time.perf_counter()
  result = graph.invoke(
    {"topic": "what new employees should know about leave, remote work and travel expenses", "written": []}
  )
  seconds = time.perf_counter() - started
  return [
    table(
      ["Section", "Question"],
      [[s["title"], s["question"]] for s in result["plan"]],
      title=f"The plan ({len(result['plan'])} sections, decided by the model)",
    ),
    code(result["report"], "markdown", title=f"The briefing ({seconds:.1f} s)"),
    callout(
      "When the subtasks depend on the input (how many sections, which questions), an orchestrator plans them, "
      "Send starts one worker per subtask in parallel, and a synthesiser combines the results. It's routing plus "
      "map-reduce, with the map decided by a model; cap the plan's size so a model can't start fifty workers.",
      "decision",
      "Orchestrator-workers",
    ),
  ] + [
    example(
      "lg.l19.1",
      "The orchestrator plans the subtasks",
      orchestrator_and_workers,
      "",
      [table(["section"], [[s["title"]] for s in result["plan"]])],
    ),
    example(
      "lg.l19.2",
      "Send: one worker per subtask",
      orchestrator_and_workers,
      "",
      [code(graph.get_graph().draw_mermaid(), "text")],
    ),
    example(
      "lg.l19.3",
      "A synthesiser combines them",
      orchestrator_and_workers,
      "",
      [code(result["report"][:800], "markdown")],
    ),
  ]


# ── Lab 6: evaluator and optimizer ──────────────────────────────────────────────
class Grade(BaseModel):
  verdict: Literal["accept", "revise"]
  feedback: str = Field(description="What to change, if revise")


class EvalState(TypedDict, total=False):
  task: str
  draft: str
  attempts: int
  verdict: str
  feedback: str
  history: Annotated[list[str], operator.add]


def evaluator_optimizer(model, max_attempts):
  from langgraph.graph import END, START, StateGraph

  def write(state):
    # the previous draft and the feedback go back to the writer (and make each attempt's prompt different)
    hint = (
      f"\n\nYour previous draft:\n{state['draft']}\n\nFix this: {state['feedback']}" if state.get("feedback") else ""
    )
    return {"draft": cache.text(model, f"{state['task']}{hint}"), "attempts": state.get("attempts", 0) + 1}

  def evaluate(state):
    words = len(state["draft"].split())
    code_problems = [
      p
      for p, bad in (
        (f"it has {words} words: cut it to under 60", words > 60),
        ("remove the [placeholder]", "[" in state["draft"]),
      )
      if bad
    ]
    if code_problems:  # code checks what code can measure
      return {
        "verdict": "revise",
        "feedback": "; ".join(code_problems),
        "history": [f"attempt {state['attempts']} ({words} words): code: {'; '.join(code_problems)}"],
      }
    grade = cache.structured(
      model,
      Grade,
      # a model for what needs understanding: does the blurb claim anything the brief doesn't say?
      f"Brief: {BLEND}\n\nBlurb:\n{state['draft']}\n\nAccept if the blurb states the price and at least one "
      "flavour note from the brief, and claims nothing the brief doesn't support (no awards, sourcing, ethics, "
      "health or origin details beyond the brief). Otherwise revise, naming each unsupported claim.",
    )
    return {
      "verdict": grade.verdict,
      "feedback": grade.feedback,
      "history": [
        f"attempt {state['attempts']} ({words} words): judge: {grade.verdict} {grade.feedback[:200]}".strip()
      ],
    }

  def next_step(state) -> str:
    if state["verdict"] == "accept":
      return END
    return END if state["attempts"] >= max_attempts else "write"  # always cap the loop

  graph = StateGraph(EvalState)
  graph.add_node("write", write)
  graph.add_node("evaluate", evaluate)
  graph.add_edge(START, "write")
  graph.add_edge("write", "evaluate")
  graph.add_conditional_edges("evaluate", next_step, ["write", END])
  return graph.compile()


@lab(
  "agents.evaluator-optimizer",
  "Evaluator and optimizer: write, grade, revise, with a cap",
  uses_model=True,
  calls="~2–6 calls to Mistral (cached after the first run)",
)
def evaluator_optimizer_lab():
  """A product blurb written, checked by code and graded by a model, and revised until it passes or the cap is hit."""
  model = get_chat_model()
  # the brief asks for a claim the facts don't support: the judge should send it back, the writer fix it
  task = f"Write a product blurb (about 50 words) for the website: {BLEND}. Call it award-winning!"
  result = evaluator_optimizer(model, 3).invoke({"task": task, "history": []})
  # no length given: the code check keeps sending it back; the cap stops the loop
  wordy = f"Write a product blurb for the website: {BLEND}. Be enthusiastic!"
  capped = evaluator_optimizer(model, 3).invoke({"task": wordy, "history": []})
  runs = {"asks for an unsupported claim": result, "no length given": capped}
  return [
    table(
      ["Task", "Step"],
      [[name, h] for name, r in runs.items() for h in r["history"]],
      title="Each attempt: what the checks said",
    ),
    table(["Task", "attempts", "final verdict"], [[name, r["attempts"], r["verdict"]] for name, r in runs.items()]),
    columns([answer(r["draft"], f"final draft: {name}") for name, r in runs.items()]),
    callout(
      "One model writes, checks grade it, and the feedback goes back to the writer with its previous draft. Use code "
      "for what code can measure (length, placeholders: free and exact) and a model only for what needs "
      "understanding (claims the brief doesn't support). Always cap the loop: a draft that never passes comes back "
      "marked 'revise' so a person knows. "
      + " ".join(f"'{name}': {r['verdict']} after {r['attempts']} attempt(s)." for name, r in runs.items()),
      "decision",
      "Evaluator-optimizer",
    ),
  ] + [
    example(
      "lg.l20.1",
      "Checks grade the draft and give feedback",
      evaluator_optimizer,
      "",
      [table(["step"], [[h] for h in result["history"]])],
    ),
    example(
      "lg.l20.2", "The writer revises with the feedback", evaluator_optimizer, "", [answer(result["draft"], "final")]
    ),
    example(
      "lg.l20.3",
      "Always cap the loop",
      evaluator_optimizer,
      "",
      [table(["task", "attempts", "verdict"], [[n, r["attempts"], r["verdict"]] for n, r in runs.items()])],
    ),
  ]


# ── Production path: support graph v3 ──────────────────────────────────────────
RUBRIC = (
  "This is a customer-support reply. Courtesy (thanks, apologies, offers to help) and restating what the "
  "customer wrote are not claims. Count as claims: facts about orders, products, policies, and promised actions."
)


def _cached_judge(models, ticket, reply, sources, n):
  """The judge's verdict, cached on disk by its inputs (an unchanged reply isn't judged twice), or None when the
  judge itself fails (counted, not hidden)."""
  import hashlib

  from nimbus.rag.quality import Verdict, judge

  key = hashlib.sha256(json.dumps([ticket, reply, sources, RUBRIC]).encode()).hexdigest()[:24]
  path = PROJECT_DIR / ".cache" / "labs" / "judge" / f"{key}.json"
  if path.exists():
    return Verdict.model_validate_json(path.read_text())
  try:
    verdict = judge(models, ticket, reply, sources, n, None, RUBRIC)
  except Exception:  # noqa: BLE001 - a judge failure is counted and shown, not hidden
    return None
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(verdict.model_dump_json())
  return verdict


def evaluate_v2_vs_v3() -> list[dict]:
  from nimbus.llm.usage import UsageRecorder
  from nimbus.rag.quality import judge_models
  from nimbus.support.evaluation import load_tickets
  from nimbus.support.graph import (
    build_support_graph_v2,
    build_support_graph_v3,
    order_note,
    refund_decision,
    reply_problems,
  )
  from nimbus.support.prompts import NEXT_STEPS
  from nimbus.support.tools import SupportStore

  model, judge_model = get_chat_model(), judge_models()
  store = SupportStore()
  graphs = {"v2": build_support_graph_v2(model, store), "v3": build_support_graph_v3(model, store, _retriever())}
  rows = []
  for labelled in load_tickets(PROJECT_DIR / "data" / "eval" / "tickets.json"):
    ticket = labelled.ticket
    ids = list(dict.fromkeys(m.upper() for m in re.findall(r"NC-\d+", ticket.text, re.IGNORECASE)))
    records = {i: store.order(i) for i in ids}
    # the judge's facts are the same for both systems: order notes, refund decisions, the top public documents
    _, hits = _retriever().search(ticket.text, k=3, audience="public")
    facts = [f"The customer wrote: {ticket.text}", f"Policy: {NEXT_STEPS[labelled.expected['category']]}"]
    facts += [order_note(i, r) for i, r in records.items()] + [refund_decision(store, r, i) for i, r in records.items()]
    facts += [d.page_content for d, _ in hits]
    sources = "\n".join(f"[{n}] {f}" for n, f in enumerate(facts, 1))
    orders = [{"record": r} for r in records.values() if r]
    for name, graph in graphs.items():
      rec = UsageRecorder()
      started = time.perf_counter()
      out = graph.invoke({"ticket": ticket.text, "history": []}, {"callbacks": [rec]})
      ms = (time.perf_counter() - started) * 1000
      safety = out["triage"].category == "allergy_safety"
      verdict = None
      if not safety:
        verdict = _cached_judge(judge_model, ticket.text, out["reply"], sources, len(facts))
      rows.append(
        {
          "system": name,
          "ticket": ticket.text,
          "reply": out["reply"],
          "safety": safety,
          "faithfulness": verdict.faithfulness if verdict else None,
          "unsupported": verdict.unsupported if verdict else None,
          "relevant": verdict.relevant if verdict else None,
          "problems": 0 if safety else len(reply_problems(out["reply"], out["triage"], orders)),
          "calls": len(rec.calls),
          "ms": ms,
        }
      )
  return rows


@lab(
  "agents.support-v3-eval",
  "Evaluate: support graph v3 (specialised handlers) vs v2",
  uses_model=True,
  calls="~100 calls: two graphs on 20 tickets (Mistral) + a judgement per reply (Groq): about 5 minutes",
)
def support_v3_eval():
  """Both graphs on the 20 labelled tickets; a Groq judge checks every reply against the same facts (order records,
  the refund policy, the top public documents for the ticket)."""
  started = time.perf_counter()
  rows = evaluate_v2_vs_v3()
  seconds = time.perf_counter() - started
  out = []
  for name in ("v2", "v3"):
    judged = [r for r in rows if r["system"] == name and r["faithfulness"] is not None]
    mine = [r for r in rows if r["system"] == name]
    ms = sorted(r["ms"] for r in mine)
    out.append(
      {
        "variant": "v2 (Module 2)" if name == "v2" else "v3 (production)",
        "faithfulness": round(sum(r["faithfulness"] for r in judged) / max(len(judged), 1), 3),
        "unsupported": sum(len(r["unsupported"]) for r in judged),
        "relevant": round(sum(r["relevant"] for r in judged) / max(len(judged), 1), 3),
        "with_problems": sum(r["problems"] > 0 for r in mine),
        "unjudged": sum(1 for r in mine if not r["safety"] and r["faithfulness"] is None),
        "calls": round(sum(r["calls"] for r in mine) / len(mine), 2),
        "p50_ms": round(ms[len(ms) // 2]),
      }
    )
  # a real improvement: at least a fifth fewer unsupported claims, not a difference of one or two
  better = out[1]["unsupported"] <= 0.8 * out[0]["unsupported"] and out[1]["faithfulness"] > out[0]["faithfulness"]
  claims = [[r["system"], r["ticket"], c] for r in rows if r["unsupported"] for c in r["unsupported"]]
  return [
    scorecard(
      out,
      [
        Metric(
          "faithfulness", "faithfulness", "higher", "ratio", "Share of a reply's claims the facts support (judged)"
        ),
        Metric("unsupported", "unsupported claims (all replies)", "lower", "int"),
        Metric("relevant", "addresses the ticket", "higher", "ratio"),
        Metric("with_problems", "replies failing the code checks", "lower", "int"),
        Metric("unjudged", "judge failed", "lower", "int"),
        Metric("calls", "calls / ticket", "lower", "text"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"20 labelled tickets, judged by Groq's gpt-oss (safety tickets excluded) · {seconds:.0f} s",
    ),
    table(
      ["System", "Ticket", "Claim the facts don't support"],
      claims,
      title="Every unsupported claim: spot-check the judge",
    ),
    callout(
      f"With a handler per kind of ticket gathering its own facts, unsupported claims went {out[0]['unsupported']} → "
      f"{out[1]['unsupported']} and faithfulness {out[0]['faithfulness']:.2f} → {out[1]['faithfulness']:.2f}, for "
      f"{out[1]['calls']} vs {out[0]['calls']} calls per ticket: product and café answers now come from public "
      "documents, and refund answers from a decision made in code. The judge is a different provider's model: "
      "read the claims above before trusting the totals."
      if better
      else f"v3 didn't reduce unsupported claims here ({out[0]['unsupported']} → {out[1]['unsupported']}, "
      f"faithfulness {out[0]['faithfulness']:.2f} → {out[1]['faithfulness']:.2f}). Read the claims above: if the "
      "judge is right, the handlers' facts aren't reaching the reply.",
      "decision" if better else "warning",
      "Ship support graph v3" if better else "No improvement measured",
    ),
  ]
