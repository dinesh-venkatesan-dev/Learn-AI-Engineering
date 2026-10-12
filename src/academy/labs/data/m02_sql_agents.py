"""Data agents · Module 2 labs: SQL agents and workflows. An agent that explores the warehouse, the same job as a
fixed LangGraph workflow, a data steward approving sensitive queries, and execution accuracy across systems."""

import json
import time

from langgraph.types import Command

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.llm import get_chat_model
from nimbus.llm.usage import UsageRecorder


def _calls(messages) -> list[str]:
  out = []
  for m in messages:
    for c in getattr(m, "tool_calls", None) or []:
      detail = c["args"].get("sql") or c["args"].get("table") or ""
      out.append(f"{c['name']}({' '.join(str(detail).split())[:160]})")
  return out


# ── Lab 1: a SQL agent ─────────────────────────────────────────────────────────
def build_agent(model):
  from nimbus.data.agent import build_sql_agent

  # Three tools: list_tables, describe_table, run_query (every query through the guard); the glossary in the prompt
  return build_sql_agent(model)


def run_agent(agent, question: dict) -> dict:
  from langchain_core.messages import HumanMessage

  from nimbus.data.evaluation import right_row
  from nimbus.data.sql import matches

  recorder = UsageRecorder()
  result = agent.invoke(
    {"messages": [HumanMessage(question["question"])]}, {"recursion_limit": 30, "callbacks": [recorder]}
  )
  rows = []
  for m in result["messages"]:
    if m.type == "tool" and m.name == "run_query" and m.text.startswith("{"):
      rows += [tuple(r) for r in json.loads(m.text)["rows"]]
  right = right_row(question["sql"])
  return {
    "question": question["question"],
    "tools": _calls(result["messages"]),
    "answer": result["messages"][-1].text,
    "right": ", ".join(map(str, right)),
    "correct": all(matches(v, rows) for v in right),
    "calls": recorder.usage().calls,
  }


@lab("data.sql-agent", "A SQL agent: explore, query, check, answer", uses_model=True, calls="~20 calls to Mistral")
def sql_agent():
  """Three hard questions (a comparison over a definition, a growth over months, a fact to look up first) through a
  create_agent with the SQL tools and the glossary."""
  from nimbus.data.agent import SYSTEM
  from nimbus.data.evaluation import load_questions

  agent = build_agent(get_chat_model())
  runs = [run_agent(agent, q) for q in load_questions("hard")]
  work = table(
    ["question", "tool calls", "answer", "right answer", "✓"],
    [[r["question"], "\n".join(r["tools"]), r["answer"], r["right"], "✓" if r["correct"] else "✗"] for r in runs],
  )
  return [
    work,
    callout(
      f"{sum(r['correct'] for r in runs)}/{len(runs)} right, with {sum(r['calls'] for r in runs) / len(runs):.1f} "
      "model calls per question: the agent lists and describes tables, then queries as often as the question needs. "
      "Every query and its result is on the page, and the answer says what it counted.",
      "info",
      "Agents explore",
    ),
    example("data.l06.1", "Three tools through the guard", build_agent, "src/nimbus/data/tools.py", [work]),
    example("data.l06.2", "The glossary in the system prompt", build_agent, "", [code(SYSTEM, "text")]),
    example("data.l06.3", "Multi-step questions", run_agent, "", [work]),
    example(
      "data.l06.4",
      "The work is visible",
      run_agent,
      "",
      [table(["question", "tool calls"], [[r["question"], "\n".join(r["tools"])] for r in runs])],
    ),
    example(
      "data.l06.5",
      "Assumptions stated",
      build_agent,
      "SYSTEM asks the agent to say what it counted.",
      [answer(r["answer"], r["question"]) for r in runs],
    ),
  ]


# ── Lab 2: a SQL workflow ──────────────────────────────────────────────────────
def build_workflow(model, reference_rows=True):
  from nimbus.data.workflow import build_sql_workflow

  # pick_tables → write_sql → run ─(error, tries left)→ fix_sql → run … → answer
  return build_sql_workflow(model, reference_rows=reference_rows)


def pick_tables_only(model, question):
  from nimbus.data.tools import table_names
  from nimbus.data.workflow import RELATIONS, Tables

  # The model names the tables; write_sql then sees only THEIR schemas
  pick = model.with_structured_output(Tables, method="json_schema")
  prompt = f"Tables: {', '.join(table_names())}. Relations: {RELATIONS}.\nWhich tables are needed to answer: {question}"
  return pick.invoke(prompt).tables


def run_workflow(workflow, question: dict) -> dict:
  from nimbus.data.evaluation import right_row
  from nimbus.data.sql import matches

  recorder = UsageRecorder()
  state = workflow.invoke({"question": question["question"]}, {"callbacks": [recorder]})
  right = right_row(question["sql"])
  return {
    "question": question["question"],
    "tables": state.get("tables", []),
    "sql": state.get("sql", ""),
    "tries": len(state.get("attempts", [])),
    "answer": state.get("answer", ""),
    "right": ", ".join(map(str, right)),
    "correct": not state.get("error") and all(matches(v, state.get("rows") or []) for v in right),
    "calls": recorder.usage().calls,
  }


@lab(
  "data.sql-workflow",
  "A SQL workflow in LangGraph: fixed, predictable steps",
  uses_model=True,
  calls="~35 calls to Mistral",
)
def sql_workflow():
  """The same three hard questions through the workflow (with and without small reference tables in full) and the
  agent: right answers and model calls per question."""
  from nimbus.data.evaluation import load_questions

  model = get_chat_model()
  hard = load_questions("hard")
  workflow = build_workflow(model)
  plain = build_workflow(model, reference_rows=False)
  agent = build_agent(model)
  with_rows = [run_workflow(workflow, q) for q in hard]
  without = [run_workflow(plain, q) for q in hard]
  agent_runs = [run_agent(agent, q) for q in hard]
  rows = []
  for name, runs in (("workflow", with_rows), ("workflow, no reference rows", without), ("agent", agent_runs)):
    rows.append(
      {
        "variant": name,
        "correct": round(sum(r["correct"] for r in runs) / len(runs), 3),
        "calls": round(sum(r["calls"] for r in runs) / len(runs), 2),
      }
    )
  detail = table(
    ["question", "tables picked", "SQL", "answer", "right", "✓"],
    [
      [r["question"], ", ".join(r["tables"]), r["sql"], r["answer"], r["right"], "✓" if r["correct"] else "✗"]
      for r in with_rows
    ],
    title="The workflow",
  )
  discovery = table(
    ["", "SQL", "answer", "✓"],
    [
      [name, r[2]["sql"], r[2]["answer"], "✓" if r[2]["correct"] else "✗"]
      for name, r in (("workflow", with_rows), ("no reference rows", without))
    ],
    title=hard[2]["question"],
  )
  flow, agent_row = rows[0], rows[2]
  return [
    scorecard(rows, [Metric("correct", "right answers"), Metric("calls", "model calls per question", "lower", "text")]),
    code(workflow.get_graph().draw_mermaid(), "text", title="The graph (Mermaid)"),
    detail,
    discovery,
    callout(
      f"The workflow made {flow['calls']} model calls per question against the agent's {agent_row['calls']}, and "
      f"got {flow['correct']:.0%} against {agent_row['correct']:.0%}. Its steps are fixed, so it can't look a fact up "
      "before writing the query. On the café's opening month, "
      + (
        "the version without reference rows still got it, by joining on cafes.opened_on inside the query; "
        if without[2]["correct"]
        else "only the version with the cafés table in full got it right; "
      )
      + "with reference tables in the prompt the fact doesn't have to be guessed or derived. Empty or all-zero results "
      "are reported as 'no matching data' in code.",
      "decision",
      "Workflow for one-query questions",
    ),
    example("data.l07.1", "The graph", build_workflow, "", [code(workflow.get_graph().draw_mermaid(), "text")]),
    example(
      "data.l07.2",
      "pick_tables",
      pick_tables_only,
      "",
      [table(["question", "tables"], [[r["question"], ", ".join(r["tables"])] for r in with_rows])],
    ),
    example(
      "data.l07.3",
      "A conditional edge does the retrying",
      build_workflow,
      "after_run: back to fix_sql while there's an error and tries left (src/nimbus/data/workflow.py).",
      [table(["question", "attempts"], [[r["question"], str(r["tries"])] for r in with_rows])],
    ),
    example(
      "data.l07.4",
      "The trade-off, measured",
      run_workflow,
      "",
      [scorecard(rows, [Metric("calls", "model calls", "lower", "text")])],
    ),
    example("data.l07.5", "Where a workflow fails: discovery", run_workflow, "", [discovery]),
  ]


# ── Lab 3: a data steward ──────────────────────────────────────────────────────
STEWARD = {
  "How many orders were cancelled in July 2026?": {"type": "approve"},
  "How many customers are in each loyalty tier?": {"type": "approve"},
  "When did our Mumbai gold-tier customers sign up?": {
    "type": "edit",
    "sql": "SELECT strftime('%Y', signup_date) AS year, COUNT(*) AS customers FROM customers WHERE city = 'Mumbai' "
    "AND loyalty_tier = 'gold' GROUP BY year ORDER BY year",
  },
  "Which cities do our silver customers live in, with their signup dates?": {
    "type": "reject",
    "reason": "row-level customer data needs a ticket; ask for counts instead",
  },
}


def ask_with_steward(workflow, question, decide, thread):
  config = {"configurable": {"thread_id": thread}}
  state = workflow.invoke({"question": question}, config)
  reviews = []
  while "__interrupt__" in state:  # paused in the run node; the checkpointer keeps the run
    request = state["__interrupt__"][0].value
    decision = decide(request)
    reviews.append((request["sql"], decision))
    state = workflow.invoke(Command(resume=decision), config)
  return state, reviews


@lab(
  "data.steward", "Human approval: a data steward OKs sensitive queries", uses_model=True, calls="~12 calls to Mistral"
)
def steward():
  """Four questions through the reviewed workflow: one runs straight through, three read customer data and are
  approved, edited and rejected by a scripted steward."""
  from langgraph.checkpoint.memory import InMemorySaver

  from nimbus.data.workflow import build_sql_workflow, needs_review

  workflow = build_sql_workflow(get_chat_model(), review=True, checkpointer=InMemorySaver())
  rows = []
  for n, (question, decision) in enumerate(STEWARD.items()):
    state, reviews = ask_with_steward(workflow, question, lambda request, d=decision: d, f"q{n}")
    reviewed = "; ".join(
      f"{d['type']}{': ' + d.get('reason', '') if d['type'] == 'reject' else ''}" for _, d in reviews
    )
    rows.append([question, reviewed or "ran straight through", state.get("sql", ""), state.get("answer", "")])
  view = table(["question", "steward", "SQL that ran", "answer"], rows)
  return [
    view,
    callout(
      f"{sum(r[1] != 'ran straight through' for r in rows)} of {len(rows)} questions paused for the steward; the "
      "rest never waited. The policy is code (needs_review), the pause is interrupt() inside the run node, and the "
      "checkpointer keeps a paused run for as long as the steward takes.",
      "decision",
      "Customer data goes through a steward",
    ),
    example("data.l08.1", "A policy in code", needs_review, "", [view]),
    example(
      "data.l08.2", "interrupt() in the run node", build_sql_workflow, "The run node of build_sql_workflow.", [view]
    ),
    example("data.l08.3", "Approve, edit, reject", ask_with_steward, "", [view]),
    example(
      "data.l08.4",
      "Everything else runs straight through",
      ask_with_steward,
      "",
      [table(["question", "steward"], [r[:2] for r in rows])],
    ),
  ]


# ── Lab 4 (and the production path): evaluate three systems ───────────────────
def systems(model) -> dict:
  from nimbus.data.agent import ask_agent, build_sql_agent
  from nimbus.data.text_to_sql import TextToSQL
  from nimbus.data.workflow import ask_workflow, build_sql_workflow

  chain, workflow, agent = TextToSQL(model), build_sql_workflow(model), build_sql_agent(model)
  return {
    "TextToSQL (Module 1)": lambda q, config: chain.ask(q, config=config),
    "Workflow": lambda q, config: ask_workflow(workflow, q, config),
    "Agent": lambda q, config: ask_agent(agent, q, config),
  }


def evaluate_systems():
  from nimbus.data.evaluation import evaluate, load_questions

  questions = load_questions() + load_questions("hard")
  return {name: evaluate(ask, questions) for name, ask in systems(get_chat_model()).items()}


@lab(
  "data.sql-systems-eval",
  "Evaluate: execution accuracy across three systems",
  uses_model=True,
  calls="~170 calls to Mistral: 15 questions × 3 systems, about 8 minutes",
)
def sql_systems_eval():
  """The 12 evaluation questions and 3 hard ones through TextToSQL, the workflow and the agent: right answers with
  a normalized and a strict scorer, model calls, tokens and time."""
  from nimbus.data.evaluation import load_questions, right_row, summary

  started = time.perf_counter()
  results = evaluate_systems()
  seconds = time.perf_counter() - started
  rows = []
  for name, graded in results.items():
    hard = [g for g in graded if g.question["id"].startswith("h")]
    rows.append({**summary(name, graded), "hard": round(sum(g.correct for g in hard) / max(len(hard), 1), 3)})
  misses = [
    [name, g.question["question"], ", ".join(map(str, right_row(g.question["sql"]))), g.sql or "(none)", g.answer[:160]]
    for name, graded in results.items()
    for g in graded
    if not g.correct
  ]
  scorer = [
    [name, g.question["question"], ", ".join(map(str, g.rows[:3]))]
    for name, graded in results.items()
    for g in graded
    if g.correct and not g.strict
  ]
  questions = load_questions() + load_questions("hard")
  best = max(rows, key=lambda r: (r["correct"], -r["calls"]))
  card = scorecard(
    rows,
    [
      Metric("correct", "right answers"),
      Metric("strict", "strict scorer"),
      Metric("hard", "hard questions right"),
      Metric("calls", "model calls per question", "lower", "text"),
      Metric("tokens", "tokens per question", "lower", "int"),
      Metric("latency_p50_ms", "latency p50", "lower", "ms"),
    ],
    title=f"{len(questions)} questions · {seconds:.0f} s",
  )
  failures = table(
    ["System", "question", "right", "SQL", "answer"], misses or [["—", "no misses", "", "", ""]], title="Every miss"
  )
  return [
    card,
    failures,
    table(["System", "question", "rows"], scorer or [["—", "none", ""]], title="Right only after normalizing"),
    callout(
      f"Most right answers: {best['variant']} ({best['correct']:.0%}, {best['calls']} model calls per question). "
      + "; ".join(f"{r['variant']} {r['correct']:.0%} at {r['calls']} calls" for r in rows if r is not best)
      + ". Read the misses: each shows its SQL, so a fix is a glossary line or an example.",
      "decision",
      f"Ship {best['variant']}",
    ),
    example(
      "data.l09.1",
      "An eval set",
      load_questions,
      "data/eval/sql_questions.json",
      [table(["id", "tags", "question"], [[q["id"], ", ".join(q["tags"]), q["question"]] for q in questions])],
    ),
    example(
      "data.l09.2", "Execution accuracy, strict and normalized", right_row, "nimbus.data.evaluation.grade", [card]
    ),
    example("data.l09.3", "Three systems, one set", systems, "", [card]),
    example("data.l09.4", "Read the failures", evaluate_systems, "", [failures]),
  ]
