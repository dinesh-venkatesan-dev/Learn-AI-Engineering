"""Three ways to put the specialists behind one assistant.

  single      ONE agent with every tool (account lookups, a public-document search, the café menu and skills)
  router      ONE structured call plans which specialists to ask and what; they run in parallel (Send); a final
              call combines their answers (skipped when only one answered); small talk is answered at once
  supervisor  a loop: a model picks the next specialist and its task, reads the answer, and decides again, until
              it writes the reply; code caps the rounds and never sends the same task twice

All three return an AssistantReply, so the evaluation compares them on the same questions.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from nimbus.assistant.skills import SkillState, load_skills, skills_prompt, unlock_tools
from nimbus.assistant.specialists import CAFE_PROMPT, DESCRIPTIONS, Name, Specialists, cafe_tools
from nimbus.support.agent import SYSTEM as ACCOUNT_SYSTEM
from nimbus.support.agent import CustomerContext, context_tools
from nimbus.support.tools import build_tools

MAX_TASKS = 3  # a router plan or a supervisor run never starts more specialists than this
SPECIALISTS = "\n".join(f"- {name}: {text}" for name, text in DESCRIPTIONS.items())


@dataclass
class AssistantReply:
  reply: str
  specialists: list[str] = field(default_factory=list)  # which specialists were asked, in order


# ── single agent ───────────────────────────────────────────────────────────────
class SingleState(SkillState):
  pass


def build_single(model: BaseChatModel, specialists: Specialists, retriever):
  store = specialists.store
  lookups = [t for t in build_tools(store) if t.name in ("get_order", "check_refund", "loyalty_points")]
  account = [t for t in context_tools(store) if t.name in ("my_orders", "my_subscription")]  # read-only

  @tool
  def search_public_documents(question: str) -> str:
    """Search Nimbus's public documents (shipping, returns, subscriptions, loyalty rules, cafés, products)."""
    _, hits = retriever.search(question, k=3, audience="public")
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d, _ in hits) or "Nothing found."

  prompt = (
    ACCOUNT_SYSTEM.format(name="the customer", customer_id="(from the login)")
    + " Use search_public_documents for policies, cafés and products, and only state what it returns.\n\n"
    + CAFE_PROMPT
    + "\n\n"
    + skills_prompt(load_skills())
  )
  agent = create_agent(
    model,
    [*account, *lookups, search_public_documents, *cafe_tools(store)],
    system_prompt=prompt,
    state_schema=SingleState,
    context_schema=CustomerContext,
    middleware=[unlock_tools],
  )

  def ask(customer_id: str, question: str, config: dict | None = None) -> AssistantReply:
    context = CustomerContext(customer_id, store.customers.get(customer_id, "the customer"))
    result = agent.invoke({"messages": [HumanMessage(question)], "skills_loaded": []}, config or {}, context=context)
    used = [c["name"] for m in result["messages"] for c in getattr(m, "tool_calls", [])]
    return AssistantReply(result["messages"][-1].text, used)

  return agent, ask


# ── router ─────────────────────────────────────────────────────────────────────
class Task(BaseModel):
  specialist: Name
  # measured: rephrased tasks ("free shipping policies and conditions") lost answers the customer's words found
  question: str = Field(description="The customer's own words for this specialist's part, not rephrased")


class Plan(BaseModel):
  tasks: list[Task] = Field(description="One task per specialist needed; empty for small talk or greetings")
  reply: str = Field("", description="Only when tasks is empty: a short, friendly reply")


class RouterState(TypedDict, total=False):
  question: str
  customer_id: str
  plan: list[dict]
  answers: Annotated[list[dict], operator.add]
  reply: str


class WorkerInput(TypedDict):
  customer_id: str
  specialist: str
  question: str


PLAN_PROMPT = (
  "You route questions for Nimbus Coffee's assistant. The specialists:\n{specialists}\n\nSplit the customer's "
  "message into the parts each specialist must answer (one task per specialist, at most {max}), keeping the "
  "customer's own words for each part. Greetings and thanks need no specialist.\n\nMessage: {question}"
)
COMBINE_PROMPT = (
  "Write one reply to the customer from the specialists' answers. Keep every fact and number as given; add "
  "nothing. Two to four short sentences.\n\nCustomer: {question}\n\n{answers}"
)


def build_router(model: BaseChatModel, specialists: Specialists):
  planner = model.with_structured_output(Plan, method="json_schema")

  def plan(state: RouterState) -> dict:
    result = planner.invoke(PLAN_PROMPT.format(specialists=SPECIALISTS, max=MAX_TASKS, question=state["question"]))
    tasks = [t.model_dump() for t in result.tasks[:MAX_TASKS]]
    return {"plan": tasks, "reply": "" if tasks else result.reply}

  def dispatch(state: RouterState):
    if not state["plan"]:
      return END  # small talk: answered by the planner, no specialist runs
    return [Send("work", {"customer_id": state["customer_id"], **task}) for task in state["plan"]]

  def work(task: WorkerInput) -> dict:
    answer = specialists.ask(task["specialist"], task["customer_id"], task["question"])
    return {"answers": [{"specialist": task["specialist"], "answer": answer}]}

  def combine(state: RouterState) -> dict:
    if len(state["answers"]) == 1:
      return {"reply": state["answers"][0]["answer"]}  # one specialist: its answer IS the reply
    answers = "\n".join(f"{a['specialist']}: {a['answer']}" for a in state["answers"])
    return {"reply": model.invoke(COMBINE_PROMPT.format(question=state["question"], answers=answers)).text}

  graph = StateGraph(RouterState)
  graph.add_node("plan", plan)
  graph.add_node("work", work)
  graph.add_node("combine", combine)
  graph.add_edge(START, "plan")
  graph.add_conditional_edges("plan", dispatch, ["work", END])
  graph.add_edge("work", "combine")
  graph.add_edge("combine", END)
  compiled = graph.compile()

  def ask(customer_id: str, question: str, config: dict | None = None) -> AssistantReply:
    result = compiled.invoke({"question": question, "customer_id": customer_id, "answers": []}, config or {})
    return AssistantReply(result["reply"], [a["specialist"] for a in result.get("answers", [])])

  return compiled, ask


# ── supervisor ─────────────────────────────────────────────────────────────────
class Decision(BaseModel):
  next: Literal["account", "knowledge", "cafe", "FINISH"]
  task: str = Field("", description="The customer's own words for the next specialist's part (empty when FINISH)")
  reply: str = Field("", description="When FINISH: the reply to the customer, from the specialists' answers only")


class SupervisorState(TypedDict, total=False):
  question: str
  customer_id: str
  notes: Annotated[list[str], operator.add]  # every specialist's answer, labelled with its name
  asked: Annotated[list[str], operator.add]
  decision: dict
  reply: str


SUPERVISOR_PROMPT = (
  "You supervise Nimbus Coffee's specialists:\n{specialists}\n\nCustomer: {question}\n\nAnswers so far:\n{notes}\n\n"
  "Choose the next specialist and what to ask it, or FINISH with the reply when every part is answered (greetings "
  "need no specialist). Never ask the same specialist the same thing twice."
)


def build_supervisor(model: BaseChatModel, specialists: Specialists):
  decide = model.with_structured_output(Decision, method="json_schema")

  def supervisor(state: SupervisorState) -> dict:
    notes = "\n".join(state.get("notes", [])) or "(none yet)"
    decision = decide.invoke(SUPERVISOR_PROMPT.format(specialists=SPECIALISTS, question=state["question"], notes=notes))
    repeat = f"{decision.next}: {decision.task}" in state.get("asked", [])
    if decision.next != "FINISH" and (len(state.get("asked", [])) >= MAX_TASKS or repeat):
      # code guards around model routing: a cap, and no repeats
      final = decide.invoke(
        SUPERVISOR_PROMPT.format(specialists=SPECIALISTS, question=state["question"], notes=notes)
        + "\nYou must FINISH now."
      )
      return {"decision": {"next": "FINISH"}, "reply": final.reply or notes}
    if decision.next == "FINISH":
      return {"decision": {"next": "FINISH"}, "reply": decision.reply}
    return {"decision": {"next": decision.next, "task": decision.task}}

  def worker(name: str):
    def run(state: SupervisorState) -> dict:
      task = state["decision"]["task"]
      answer = specialists.ask(name, state["customer_id"], task)
      return {"notes": [f"{name}: {answer}"], "asked": [f"{name}: {task}"]}

    return run

  graph = StateGraph(SupervisorState)
  graph.add_node("supervisor", supervisor)
  for name in DESCRIPTIONS:
    graph.add_node(name, worker(name))
    graph.add_edge(name, "supervisor")
  graph.add_edge(START, "supervisor")
  graph.add_conditional_edges(
    "supervisor", lambda s: END if s["decision"]["next"] == "FINISH" else s["decision"]["next"], [*DESCRIPTIONS, END]
  )
  compiled = graph.compile()

  def ask(customer_id: str, question: str, config: dict | None = None) -> AssistantReply:
    result = compiled.invoke({"question": question, "customer_id": customer_id, "notes": [], "asked": []}, config or {})
    return AssistantReply(result["reply"], [a.split(":")[0] for a in result.get("asked", [])])

  return compiled, ask
