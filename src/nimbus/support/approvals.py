"""Approvals for the support desk: sensitive tool calls pause for a person before they run.

  pause_subscription  the CUSTOMER confirms ("Pause your Chikmagalur Classic for 4 weeks?"): approve, edit, reject
  issue_refund        a TEAMMATE approves: approve, edit the amount (at most the order total), or reject with a reason
  everything else     runs without asking (lookups, escalations, memories)

The desk's review node asks with interrupt() (one question per sensitive call, always in the same order), records
the decisions in state, and rewrites edited calls in place (same message id). ToolNode then runs every call; a
wrapper answers rejected ones with the reviewer's reason instead of running them, so every call still gets its
ToolMessage and the conversation stays valid.
"""

from datetime import date
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from nimbus.support.tools import REFUND_WINDOW_DAYS, SupportStore

REVIEWERS = {"pause_subscription": "customer", "issue_refund": "teammate"}


class Review(BaseModel):
  """A reviewer's answer to one tool call (the interrupt's response_schema: a UI can render it as a form)."""

  decision: Literal["approve", "edit", "reject"]
  args: dict | None = Field(None, description="For edit: the corrected arguments")
  reason: str | None = Field(None, description="For reject: why, told to the agent")


def refund_tool(store: SupportStore):
  @tool
  def issue_refund(order_id: str, amount_rupees: int, runtime: ToolRuntime) -> Command:
    """Refund an order to the original payment method, after check_refund said it's eligible. A teammate approves
    every refund before it runs."""
    order = store.order(order_id)
    if order is None or order["customer_id"] != runtime.context.customer_id:  # ownership, in code
      raise ToolException(f"{order_id} isn't one of this customer's orders.")
    if order["status"] != "delivered" or order["opened"] or order["delivered_on"] is None:  # policy, in code
      raise ToolException(f"{order_id} isn't eligible for a refund.")
    if (store.today - date.fromisoformat(order["delivered_on"])).days > REFUND_WINDOW_DAYS:
      raise ToolException(f"{order_id} is outside the {REFUND_WINDOW_DAYS}-day window.")
    if not 0 < amount_rupees <= order["total"]:
      raise ToolException(f"A refund for {order_id} is between ₹1 and ₹{order['total']}.")
    return Command(
      update={
        "actions": [f"issue_refund({order_id}, ₹{amount_rupees})"],
        "messages": [ToolMessage(f"Refunded ₹{amount_rupees} for {order_id}.", tool_call_id=runtime.tool_call_id)],
      }
    )

  issue_refund.handle_tool_error = True
  return issue_refund


def ask_payload(call: dict, store: SupportStore) -> dict:
  """Plain data only: the payload is saved in the checkpoint while the run waits."""
  args = call["args"]
  if call["name"] == "issue_refund":
    order = store.order(str(args.get("order_id", ""))) or {}
    question = f"Refund ₹{args.get('amount_rupees')} for {args.get('order_id')} (order total ₹{order.get('total')})?"
  else:
    question = f"Pause your subscription for {args.get('weeks')} weeks?"
  return {"reviewer": REVIEWERS[call["name"]], "tool": call["name"], "args": args, "question": question}


def check_edit(call: dict, args: dict, tools_by_name: dict, store: SupportStore) -> str | None:
  """Why edited arguments can't be used, or None. The tool's own schema first, then business limits."""
  try:
    tools_by_name[call["name"]].tool_call_schema.model_validate(args)
  except ValidationError as error:
    return f"invalid arguments: {error.errors()[0]['msg']}"
  if call["name"] == "issue_refund":
    order = store.order(str(args.get("order_id", ""))) or {}
    if args.get("amount_rupees", 0) > order.get("total", 0):
      return f"the amount can't exceed the order total (₹{order.get('total')})"
  return None


def review_calls(message: AIMessage, ask, tools_by_name: dict, store: SupportStore) -> tuple[AIMessage, dict]:
  """(the message with edited calls, {call_id: decision}). `ask(payload)` returns a Review: interrupt() in the graph,
  a plain function in tests. Invalid edits are asked again (the same question, with the problem)."""
  calls, decisions = [], {}
  for call in message.tool_calls:
    if call["name"] not in REVIEWERS:
      calls.append(call)
      continue
    payload = ask_payload(call, store)
    while True:
      review = ask(payload)
      problem = check_edit(call, review.args or {}, tools_by_name, store) if review.decision == "edit" else None
      if problem is None:
        break
      payload = {**payload, "problem": problem}
    if review.decision == "edit":
      call = {**call, "args": review.args}
    decisions[call["id"]] = {"decision": review.decision, "reason": review.reason or "", "by": REVIEWERS[call["name"]]}
    calls.append(call)
  return message.model_copy(update={"tool_calls": calls}), decisions


def skip_rejected(request, execute):
  """ToolNode wrapper: a rejected call is answered with the reason instead of running."""
  decision = (request.state.get("reviews") or {}).get(request.tool_call["id"])
  if decision and decision["decision"] == "reject":
    who = "The customer" if decision["by"] == "customer" else "A teammate"
    reason = f": {decision['reason']}" if decision["reason"] else "."
    return ToolMessage(
      f"Not done. {who} declined{reason}", tool_call_id=request.tool_call["id"], name=request.tool_call["name"]
    )
  return execute(request)
