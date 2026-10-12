"""The support desk graph tested end to end with a scripted model: no API calls, no cost, the same result every run."""

import httpx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from nimbus.support.agent import CustomerContext
from nimbus.support.desk import UNAVAILABLE, build_support_desk
from nimbus.support.memory import memory_checkpointer
from nimbus.support.tools import SupportStore


class ScriptedModel(FakeMessagesListChatModel):
  """Returns the scripted messages in order; tools and schemas are accepted and ignored."""

  def bind_tools(self, tools, **kwargs):
    return self

  def with_structured_output(self, schema, **kwargs):
    return RunnableLambda(lambda _: schema(reply="(respond node)", resolved=False, needs_human=True))


def call(name: str, args: dict, call_id: str) -> AIMessage:
  return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def ask(desk, text: str, customer: str = "c-1042", config: dict | None = None):
  payload = {"messages": [HumanMessage(text)], "customer_id": customer, "actions": []}
  return desk.invoke(payload, config or {}, context=CustomerContext(customer, "Test"))


def test_the_whole_graph_with_a_scripted_model():
  model = ScriptedModel(
    responses=[
      call("my_subscription", {}, "c1"),  # the agent looks something up...
      call(
        "final_answer",
        {"reply": "Your next delivery is on 28 March 2026.", "resolved": True, "needs_human": False},
        "c2",
      ),
    ]
  )
  result = ask(build_support_desk(model, SupportStore()), "When is my next delivery?")
  ticket = result["ticket"]
  assert ticket.reply == "Your next delivery is on 28 March 2026."
  assert ticket.tools_used == ["my_subscription"] and not ticket.needs_human
  assert "2026-03-28" in result["messages"][2].content  # the real tool ran against the order data


def test_an_approval_pauses_the_run_and_resume_completes_it():
  model = ScriptedModel(
    responses=[
      call("issue_refund", {"order_id": "NC-90112", "amount_rupees": 650}, "c1"),
      call("final_answer", {"reply": "Refunded ₹650.", "resolved": True, "needs_human": False}, "c2"),
    ]
  )
  desk = build_support_desk(model, SupportStore(), checkpointer=memory_checkpointer(), refunds=True, review=True)
  config = {"configurable": {"thread_id": "t1"}}
  paused = ask(desk, "Refund NC-90112 please.", "c-6060", config)
  assert paused["__interrupt__"][0].value["reviewer"] == "teammate"
  done = desk.invoke(Command(resume={"decision": "approve"}), config, context=CustomerContext("c-6060", "Test"))
  assert done["actions"] == ["issue_refund(NC-90112, ₹650)"]


def test_one_part_of_the_graph_from_an_injected_state():
  model = ScriptedModel(responses=[AIMessage("unused")])
  desk = build_support_desk(model, SupportStore(), checkpointer=memory_checkpointer())
  config = {"configurable": {"thread_id": "t2"}}
  state = {
    "messages": [
      HumanMessage("Hi"),
      call("final_answer", {"reply": "Hello!", "resolved": True, "needs_human": False}, "c1"),
    ],
    "customer_id": "c-1042",
    "customer": {"name": "Test", "customer_id": "c-1042", "orders": 2, "has_subscription": True},
    "language": "English",
    "actions": [],
  }
  # as if the agent had answered and review passed it on. Not as_node="agent": routing after the agent reads
  # remaining_steps, which is 2 inside update_state, so the step guard sends it to the respond node instead
  desk.update_state(config, state, as_node="review")
  result = desk.invoke(None, config, context=CustomerContext("c-1042", "Test"))  # runs tools → file_ticket only
  assert result["ticket"].reply == "Hello!" and model.i == 0  # the model was never called


class DownModel(ScriptedModel):
  def _generate(self, *args, **kwargs):
    raise httpx.ConnectError("provider down")


def test_an_outage_reaches_a_person_and_a_fallback_answers():
  down = DownModel(responses=[])
  ticket = ask(build_support_desk(down, SupportStore()), "When is my next delivery?")["ticket"]
  assert ticket.reply == UNAVAILABLE and ticket.needs_human and ticket.flags == ["agent unavailable: network error"]
  backup = ScriptedModel(
    responses=[call("final_answer", {"reply": "28 March 2026.", "resolved": True, "needs_human": False}, "c1")]
  )
  ticket = ask(build_support_desk(down, SupportStore(), fallback=backup, retries=1), "When is my next delivery?")[
    "ticket"
  ]
  assert ticket.reply == "28 March 2026." and not ticket.flags
