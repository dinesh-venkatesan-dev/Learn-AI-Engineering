"""Support desk: the support agent built as a LangGraph workflow, with business state around the agent loop.

  load_customer  code      the customer's profile from the order system (raw data in state; the prompt is built later)
  agent          model     system prompt from the profile + the conversation → the model with the tools bound; it must
                           call a tool every turn (tool_choice="any"), and it finishes by calling `final_answer`
  tools          ToolNode  runs every tool call (in parallel), errors become messages, tools read the customer from
                           the runtime context and record actions with Command
  respond        model     a fallback only: if the agent stops without final_answer (or runs out of steps), one more
                           structured call turns the conversation into the outcome
  file_ticket    code      a record for the support desk: who, which orders, what was done, does a person follow up;
                           a reply that claims an action no tool performed is flagged for a person

    START → load_customer → agent ⇄ tools ─(final_answer called)→ file_ticket → END
                              └→ respond ──────────────────────────↗

Every node emits a progress event (stream_mode="custom"), so a UI can show what's happening while the agent works.
Measured in Agents Module 4: ending with a separate respond call let the agent stop on plain text ("I'll escalate
this", without escalating) and the respond call then claimed actions nobody took ("paused for 20 weeks").
"""

import logging
import operator
import re
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.errors import NodeError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel

from nimbus.llm.errors import classify
from nimbus.support.agent import SYSTEM, CustomerContext, SupportOutcome, agent_tools
from nimbus.support.approvals import REVIEWERS, Review, refund_tool, review_calls, skip_rejected
from nimbus.support.memory import recall, remember
from nimbus.support.tools import SupportStore

UNAVAILABLE = (
  "Sorry, our assistant can't answer right now. A teammate has your message and will reply within one business day."
)
log = logging.getLogger(__name__)
RESPOND = (
  "Write the final outcome of this conversation for the customer and the support desk. The reply must only state "
  "what the tool results above say; if a person must follow up (escalated, or not resolved), set needs_human."
)
FINISH = (
  " If a lookup finds nothing, say so plainly. When you have what you need, call final_answer with the reply; never "
  "answer in plain text."
)
# Measured: told to reply "in the customer's language", ministral answered English questions from customers with
# Indian names in Hindi, and every rewording that mentioned language made it worse. Code decides; the prompt names it.
# fmt: off
ENGLISH = {
  "the", "a", "an", "is", "are", "was", "my", "me", "i", "you", "your", "what", "where", "when", "how", "why", "can",
  "could", "please", "and", "or", "for", "to", "of", "in", "on", "it", "this", "that", "do", "does", "did", "have",
  "has", "will", "would", "should", "with", "from", "at", "be", "not", "no",
}
# fmt: on
# a reply that says it did something needs the tool that does it in the run's actions
CLAIMS = {
  r"\b(has been|have|I've|we've|is now)\s+paused\b|\bpaused (it|your)": "pause_subscription",
  r"\b(escalated|passed (this|it) (on|to))\b": "escalate",
}


class Ticket(BaseModel):
  """What the support desk sees for every conversation, filed in code from the run (no model)."""

  customer_id: str
  order_ids: list[str]
  actions: list[str]
  tools_used: list[str]
  needs_human: bool
  resolved: bool
  reply: str
  flags: list[str] = []  # why a person should look, found by code


class DeskState(MessagesState):
  customer_id: str
  customer: dict  # the profile, loaded in code before the agent starts
  language: str  # decided in code from the customer's message
  actions: Annotated[list[str], operator.add]  # recorded by the tools themselves (Command)
  outcome: SupportOutcome | None
  ticket: Ticket | None
  memories: list[str]  # what the customer told us in earlier conversations, recalled for this message
  reviews: dict  # this turn's approvals: tool call id → decision (approvals.py)
  flags: list[str]  # problems found before the ticket (e.g. the model was unavailable)
  on_backup: bool  # this turn switched to the fallback model
  remaining_steps: RemainingSteps


@tool(args_schema=SupportOutcome)
def final_answer(reply: str, resolved: bool, needs_human: bool, order_ids: list[str] | None = None) -> str:
  """Send the final answer to the customer. Call it last, alone, once the other tools have given you what you need."""
  return "Sent to the customer."


def customer_profile(store: SupportStore, customer_id: str) -> dict:
  orders = store.orders_of(customer_id)
  return {
    "name": store.customers.get(customer_id, "the customer"),
    "customer_id": customer_id,
    "orders": len(orders),
    "has_subscription": customer_id in store.subscriptions,
    "today": store.today.strftime("%d %B %Y"),  # measured: without it, replies turned 2026 dates into 2024
  }


def reply_language(text: str) -> str:
  """The language to reply in, decided in code from the message (not from the customer's name)."""
  words = re.findall(r"[a-z']+", text.casefold())
  if text.isascii() and words and sum(w in ENGLISH for w in words) / len(words) >= 0.2:
    return "English"
  return "the language the customer wrote in"


def system_prompt(customer: dict, language: str = "English", memories: list[str] | None = None) -> str:
  subscription = "has a bean subscription" if customer["has_subscription"] else "has no bean subscription"
  prompt = SYSTEM.format(name=customer["name"], customer_id=customer["customer_id"])
  prompt = prompt.replace("in the customer's language", f"in {language}") + f" They {subscription}."
  if customer.get("today"):
    prompt += f" Today is {customer['today']}."
  if memories:
    # measured: "you know: ..." let a remembered preference leak into account facts ("Chikmagalur Classic decaf")
    prompt += (
      " The customer's preferences, from earlier conversations: " + " ".join(memories) + " These are preferences, "
      "not facts about their account: orders and subscriptions come only from the tools."
    )
  return prompt


def unbacked_claims(reply: str, actions: list[str]) -> list[str]:
  """Actions the reply says happened that no tool performed."""
  text = reply.replace("\u2019", "'")
  done = " ".join(actions)
  return [
    f"reply claims {tool_name} but it never ran"
    for pattern, tool_name in CLAIMS.items()
    if re.search(pattern, text, re.IGNORECASE) and tool_name not in done
  ]


def build_support_desk(
  model: BaseChatModel,
  store: SupportStore | None = None,
  checkpointer=None,
  max_steps: int = 12,
  ending: Literal["tool", "respond"] = "tool",
  memory: BaseStore | None = None,
  history_tokens: int | None = 3000,
  refunds: bool = False,
  review: bool = False,
  read_only: bool = False,
  fallback: BaseChatModel | None = None,
  retries: int = 0,
):
  """`ending`: how the outcome is produced. "tool" (production): the agent calls final_answer, a tool every turn.
  "respond": the agent may stop with text, and a final structured call writes the outcome (kept for comparison).
  `memory`: a store for customer memories (recalled before the agent, saved by its remember tool).
  `history_tokens`: the most recent conversation the model sees each call (the checkpointer keeps it all).
  `refunds`: the issue_refund tool. `review`: sensitive calls pause for a person first (needs a checkpointer).
  `read_only`: no tools that change an account (the assistant answers questions; changes go through the desk).
  `retries`: extra attempts of the agent step on temporary errors; from the second attempt it uses `fallback`
  (another provider) when given. When every attempt fails, the ticket goes to a person with an apology."""
  store = store or SupportStore()
  extra = [remember] if memory is not None else []
  if refunds:
    extra.append(refund_tool(store))
  refund_note = (
    " To refund an order check_refund says is eligible, call issue_refund with the amount; a teammate approves it, so "
    "say the refund is done only if issue_refund says so."
    if refunds
    else ""
  )
  base = [t for t in agent_tools(store) if not (read_only and t.name in REVIEWERS)]
  if ending == "tool":
    tools = [*base, *extra, final_answer]
    agent_model = model.bind_tools(tools, tool_choice="any")  # a tool every turn: the answer is a tool too
    agent_model = agent_model.with_config(tags=["desk-agent"])  # a UI can stream this model's chunks by tag
    finish = FINISH
  else:
    tools = [*base, *extra]
    agent_model = model.bind_tools(tools).with_config(tags=["desk-agent"])
    finish = " If a lookup finds nothing, say so plainly."
  outcome_model = model.with_structured_output(SupportOutcome, method="json_schema")
  backup_model = None
  if fallback is not None:
    # tool_choice="auto": Groq's gpt-oss answered a forced tool call with "Parsing failed" (400). If the backup
    # answers in text, the respond node writes the outcome, and that call falls back to the backup too.
    backup_model = fallback.bind_tools(tools, tool_choice="auto").with_config(tags=["desk-agent", "fallback"])
    outcome_model = outcome_model.with_fallbacks(
      [fallback.with_structured_output(SupportOutcome, method="json_schema")]
    )

  def load_customer(state: DeskState, runtime: Runtime[CustomerContext]) -> dict:
    get_stream_writer()({"progress": "loading your account"})
    latest = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))  # runs every turn
    known = recall(runtime.store, state["customer_id"], latest.text) if runtime.store is not None else []
    return {
      "customer": customer_profile(store, state["customer_id"]),
      "language": reply_language(latest.text),
      "memories": known,
      "outcome": None,  # this turn's answer: a checkpointed conversation still holds the previous turn's
      "ticket": None,
      "reviews": {},
      "flags": [],
      "on_backup": False,
    }

  def agent(state: DeskState, runtime: Runtime[CustomerContext]) -> dict:
    attempt = runtime.execution_info.node_attempt if runtime.execution_info else 1
    # the backup from the second attempt, and for the rest of the turn once it was needed (no retrying a dead provider)
    use_backup = backup_model is not None and (attempt > 1 or state.get("on_backup", False))
    chosen = backup_model if use_backup else agent_model
    get_stream_writer()({"progress": "thinking" if attempt == 1 else f"thinking (attempt {attempt})"})
    system = SystemMessage(
      system_prompt(state["customer"], state["language"], state.get("memories")) + refund_note + finish
    )
    history = state["messages"]
    if history_tokens:  # the model sees the recent conversation; the checkpoint keeps all of it
      history = trim_messages(
        history, max_tokens=history_tokens, strategy="last", token_counter=count_tokens_approximately, start_on="human"
      )
    return {"messages": [chosen.invoke([system, *history])], "on_backup": use_backup}

  def after_agent(state: DeskState) -> Literal["review", "tools", "respond"]:
    last = state["messages"][-1]
    # tool calls → run them; but leave room for respond + file_ticket before the recursion limit
    if isinstance(last, AIMessage) and last.tool_calls and state["remaining_steps"] > 4:
      return "review" if review else "tools"
    return "respond"

  tools_by_name = {t.name: t for t in tools}

  def review_node(state: DeskState) -> dict:
    """Sensitive calls pause for the customer or a teammate; edits replace the call in place (same message id)."""
    last = state["messages"][-1]
    if not any(c["name"] in REVIEWERS for c in last.tool_calls):
      return {}
    get_stream_writer()({"progress": "waiting for approval"})

    def ask(payload: dict) -> Review:
      return Review.model_validate(interrupt(payload, response_schema=Review))

    edited, decisions = review_calls(last, ask, tools_by_name, store)
    return {"messages": [edited], "reviews": {**state.get("reviews", {}), **decisions}}

  def after_tools(state: DeskState) -> Literal["agent", "file_ticket"]:
    finished = any(isinstance(m, ToolMessage) and m.name == "final_answer" for m in _latest_tool_results(state))
    return "file_ticket" if finished else "agent"

  def respond(state: DeskState) -> dict:
    get_stream_writer()({"progress": "writing the reply"})
    system = SystemMessage(system_prompt(state["customer"], state["language"], state.get("memories")))
    messages = [system, *state["messages"], HumanMessage(RESPOND)]
    return {"outcome": outcome_model.invoke(messages)}

  def file_ticket(state: DeskState) -> dict:
    outcome = state.get("outcome") or _final_answer(state)
    calls = [c for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
    mentioned = re.findall(r"NC-\d+", " ".join(str(c["args"]) for c in calls) + " " + outcome.reply)
    escalated = any(a.startswith("escalate") for a in state.get("actions", []))
    flags = [*state.get("flags", []), *unbacked_claims(outcome.reply, state.get("actions", []))]
    ticket = Ticket(
      customer_id=state["customer_id"],
      order_ids=sorted(set(outcome.order_ids) | set(mentioned)),
      actions=state.get("actions", []),
      tools_used=[c["name"] for c in calls if c["name"] != "final_answer"],
      needs_human=outcome.needs_human or escalated or bool(flags),  # code has the last word on routing
      resolved=outcome.resolved and not escalated and not flags,
      reply=outcome.reply,
      flags=flags,
    )
    get_stream_writer()({"progress": "done"})
    # the reply as an AI message: the conversation reads naturally next turn, and never ends on a tool result
    return {"outcome": outcome, "ticket": ticket, "messages": [AIMessage(outcome.reply)]}

  graph = StateGraph(DeskState, context_schema=CustomerContext)
  graph.add_node("load_customer", load_customer)
  retry = RetryPolicy(max_attempts=1 + retries, initial_interval=0.5, retry_on=lambda e: classify(e).action == "retry")

  def give_up(state: DeskState, error: NodeError) -> Command:
    """Every attempt failed: no model reply at all, so a person takes the conversation (the error is logged)."""
    failure = classify(error.error)
    log.warning("desk agent failed after %s attempt(s): %s", 1 + retries, failure)
    apology = SupportOutcome(reply=UNAVAILABLE, resolved=False, needs_human=True)
    return Command(update={"outcome": apology, "flags": [f"agent unavailable: {failure.reason}"]}, goto="file_ticket")

  graph.add_node("agent", agent, retry_policy=retry, error_handler=give_up)
  graph.add_node("review", review_node)
  graph.add_node("tools", ToolNode(tools, handle_tool_errors=True, wrap_tool_call=skip_rejected))
  graph.add_node("respond", respond)
  graph.add_node("file_ticket", file_ticket)
  graph.add_edge(START, "load_customer")
  graph.add_edge("load_customer", "agent")
  graph.add_conditional_edges("agent", after_agent, ["review", "tools", "respond"])
  graph.add_edge("review", "tools")
  graph.add_conditional_edges("tools", after_tools, ["agent", "file_ticket"])
  graph.add_edge("respond", "file_ticket")
  graph.add_edge("file_ticket", END)
  return graph.compile(checkpointer=checkpointer, store=memory).with_config(recursion_limit=max_steps * 2)


def _latest_tool_results(state: DeskState) -> list:
  results = []
  for m in reversed(state["messages"]):
    if not isinstance(m, ToolMessage):
      break
    results.append(m)
  return results


def _final_answer(state: DeskState) -> SupportOutcome:
  for m in reversed(state["messages"]):
    if isinstance(m, AIMessage):
      for call in m.tool_calls:
        if call["name"] == "final_answer":
          return SupportOutcome(**call["args"])
  raise ValueError("no final_answer call in the conversation")


def ask(desk, store: SupportStore, customer_id: str, text: str) -> dict:
  """One request through the desk; the customer comes from the login (context), never from the model."""
  context = CustomerContext(customer_id, store.customers.get(customer_id, "the customer"))
  return desk.invoke({"messages": [HumanMessage(text)], "customer_id": customer_id, "actions": []}, context=context)
