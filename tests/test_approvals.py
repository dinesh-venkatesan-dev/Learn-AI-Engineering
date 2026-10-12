from langchain_core.messages import AIMessage

from nimbus.support.agent import agent_tools
from nimbus.support.approvals import Review, refund_tool, review_calls
from nimbus.support.tools import SupportStore


def _call(name, args, call_id):
  return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _setup():
  store = SupportStore()
  tools = {t.name: t for t in [*agent_tools(store), refund_tool(store)]}
  message = AIMessage(
    "",
    id="m1",
    tool_calls=[
      _call("my_orders", {}, "c1"),
      _call("issue_refund", {"order_id": "NC-90112", "amount_rupees": 650}, "c2"),
    ],
  )
  return store, tools, message


def test_only_sensitive_calls_are_asked_about_and_edits_keep_the_message_id():
  store, tools, message = _setup()
  asked = []

  def ask(payload):
    asked.append(payload)
    return Review(decision="edit", args={"order_id": "NC-90112", "amount_rupees": 550})

  edited, decisions = review_calls(message, ask, tools, store)
  assert [p["tool"] for p in asked] == ["issue_refund"] and asked[0]["reviewer"] == "teammate"
  assert edited.id == "m1" and edited.tool_calls[1]["args"]["amount_rupees"] == 550
  assert decisions == {"c2": {"decision": "edit", "reason": "", "by": "teammate"}}


def test_an_edit_above_the_order_total_is_asked_again():
  store, tools, message = _setup()
  answers = iter(
    [
      Review(decision="edit", args={"order_id": "NC-90112", "amount_rupees": 9000}),
      Review(decision="reject", reason="changed my mind"),
    ]
  )
  asked = []

  def ask(payload):
    asked.append(payload)
    return next(answers)

  _, decisions = review_calls(message, ask, tools, store)
  assert "problem" in asked[1] and "650" in asked[1]["problem"]
  assert decisions["c2"]["decision"] == "reject"


def test_an_approved_refund_outside_the_window_still_fails_in_code():
  from langgraph.graph import START, MessagesState, StateGraph
  from langgraph.prebuilt import ToolNode

  from nimbus.support.agent import CustomerContext

  store = SupportStore()

  class State(MessagesState):
    actions: list

  graph = StateGraph(State, context_schema=CustomerContext)
  graph = graph.add_node("tools", ToolNode([refund_tool(store)])).add_edge(START, "tools").compile()
  call = _call("issue_refund", {"order_id": "NC-14002", "amount_rupees": 100}, "c9")  # delivered 24 days ago
  result = graph.invoke(
    {"messages": [AIMessage("", tool_calls=[call])], "actions": []}, context=CustomerContext("c-8080", "Anita D.")
  )
  assert "14-day window" in result["messages"][-1].content and not result["actions"]
