"""Support agent v1: create_agent with the support tools, the customer as runtime context, an audit trail in state,
a validated outcome, and memory per conversation.

  context  CustomerContext  who is asking: set by the app per request, read by tools through ToolRuntime. The model
                            never sees or chooses it, so it can't look up someone else's orders.
  state    SupportState     messages + `actions`: what the agent did (paused a subscription, escalated), appended by
                            the tools themselves with Command(update=...).
  output   SupportOutcome   the reply plus routing fields (resolved, needs_human), validated.
  memory   a checkpointer per conversation (thread_id), summarised when it gets long.
"""

import operator
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from nimbus.support.tools import SupportStore, build_tools

SYSTEM = (
  "You are Nimbus Coffee's support agent. You are talking to {name} (customer {customer_id}). Use the tools to look "
  "things up and to act; never guess order details, prices or refund eligibility, and never promise a refund the "
  "check_refund tool didn't confirm. Escalate safety issues, legal threats and anything you can't resolve. "
  "Reply in two or three short sentences, in the customer's language."
)


@dataclass
class CustomerContext:
  customer_id: str
  name: str


class SupportState(AgentState):
  actions: Annotated[list[str], operator.add]  # a reducer: updates from several tool calls are concatenated


class SupportOutcome(BaseModel):
  """The final answer to the customer, with routing fields for the support desk."""

  reply: str = Field(description="The message to send to the customer")
  resolved: bool = Field(description="true if the request is fully handled with no follow-up needed")
  needs_human: bool = Field(description="true if a person must follow up (escalated, or couldn't be resolved)")
  order_ids: list[str] = Field(default_factory=list, description="Every order number this conversation was about")


def context_tools(store: SupportStore) -> list[BaseTool]:
  @tool
  def my_orders(runtime: ToolRuntime[CustomerContext]) -> str:
    """The current customer's orders, NEWEST FIRST: number, date, status and total."""
    # Newest first, and labelled: listed oldest-first, the model took the first line as "the latest order".
    orders = list(reversed(store.orders_of(runtime.context.customer_id)))  # from the context, never the model
    rows = [f"{o['order_id']} (ordered {o['ordered_on']}, {o['status']}, ₹{o['total']})" for o in orders]
    return ("Newest first: " + "; ".join(rows)) if rows else "No orders."

  @tool
  def my_subscription(runtime: ToolRuntime[CustomerContext]) -> str:
    """The current customer's bean subscription: blend, frequency, next delivery, pause."""
    sub = store.subscriptions.get(runtime.context.customer_id)
    if sub is None:
      return "This customer has no subscription."
    paused = f", paused until {sub['paused_until']}" if sub["paused_until"] else ""
    return f"{sub['blend']} every {sub['every_weeks']} weeks, next delivery {sub['next_delivery']}{paused}"

  @tool
  def pause_subscription(weeks: int, runtime: ToolRuntime[CustomerContext]) -> Command:
    """Pause the current customer's subscription for 1 to 12 weeks."""
    sub = store.subscriptions.get(runtime.context.customer_id)
    if sub is None:
      raise ToolException("This customer has no subscription to pause.")
    if not 1 <= weeks <= 12:
      raise ToolException("Subscriptions can be paused for 1 to 12 weeks.")
    sub["paused_until"] = (store.today + timedelta(weeks=weeks)).isoformat()
    note = f"Paused {sub['blend']} until {sub['paused_until']}."
    return Command(
      update={  # a tool can update state: the action is recorded next to its ToolMessage
        "actions": [f"pause_subscription({weeks} weeks)"],
        "messages": [ToolMessage(note, tool_call_id=runtime.tool_call_id)],
      }
    )

  @tool
  def escalate(reason: str, urgency: Literal["normal", "high", "urgent"], runtime: ToolRuntime) -> Command:
    """Hand the conversation to a person: safety issues, legal threats, faulty opened bags, anything unresolved."""
    hours = {"urgent": "1 hour", "high": "4 hours", "normal": "1 business day"}[urgency]
    return Command(
      update={
        "actions": [f"escalate({urgency}): {reason}"],
        "messages": [
          ToolMessage(
            f"Escalated ({urgency}). The team will contact the customer within {hours}.",
            tool_call_id=runtime.tool_call_id,
          )
        ],
      }
    )

  for t in (pause_subscription,):
    t.handle_tool_error = True
  return [my_orders, my_subscription, pause_subscription, escalate]


def agent_tools(store: SupportStore) -> list[BaseTool]:
  lookups = [t for t in build_tools(store) if t.name != "escalate_to_human"]  # escalation now updates state instead
  return lookups + context_tools(store)


@dynamic_prompt
def support_prompt(request: ModelRequest) -> str:
  """The system prompt, built per call from the runtime context: the model knows who it's talking to."""
  customer = request.runtime.context
  return SYSTEM.format(name=customer.name, customer_id=customer.customer_id)


def build_support_agent(
  model: BaseChatModel,
  store: SupportStore | None = None,
  checkpointer=None,
  output: Literal["provider", "tool", "text"] = "tool",
  summarize_after_tokens: int | None = 4000,
):
  """The agent. `output`: how the outcome is requested (provider JSON schema, or as one more tool), or "text": a plain
  reply that can be streamed token by token (the HTTP service), with the routing read from the recorded actions."""
  store = store or SupportStore()
  strategy = {"provider": ProviderStrategy(SupportOutcome), "tool": ToolStrategy(SupportOutcome), "text": None}[output]
  middleware = [support_prompt]
  if summarize_after_tokens:  # long conversations: older turns become a summary instead of growing forever
    middleware.append(
      SummarizationMiddleware(model=model, trigger=("tokens", summarize_after_tokens), keep=("messages", 6))
    )
  return create_agent(
    model,
    tools=agent_tools(store),
    response_format=strategy,
    context_schema=CustomerContext,
    state_schema=SupportState,
    middleware=middleware,
    checkpointer=checkpointer,
  )
