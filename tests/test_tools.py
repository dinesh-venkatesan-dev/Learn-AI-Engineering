"""Support tools and the tool-use scoring, without calling a model."""

from langchain_core.messages import AIMessage

from nimbus.support.assistant import ToolRun
from nimbus.support.tool_eval import call_matches, score, terse
from nimbus.support.tools import SupportStore, build_tools

TOOLS = {t.name: t for t in build_tools(SupportStore())}


def call(name, **args):
  return TOOLS[name].invoke({"name": name, "args": args, "id": "c1", "type": "tool_call"})


def test_refund_policy():
  assert "eligible" in call("check_refund", order_id="NC-4410", opened=False).content  # 9 days, sealed
  assert "outside the 14-day window" in call("check_refund", order_id="NC-12093", opened=False).content  # 16 days
  assert "opened bags" in call("check_refund", order_id="NC-20417", opened=True).content
  assert "cancelled instead" in call("check_refund", order_id="NC-55021", opened=False).content  # not shipped


def test_failures_come_back_as_messages():
  missing = call("get_order", order_id="NC-99999")
  assert missing.status == "error" and "check the number" in missing.content
  malformed = call("get_order", order_id="12093")
  assert malformed.status == "error" and "pattern" in malformed.content


def test_artifact_and_return_direct():
  message = call("get_order", order_id="NC-12500")
  assert message.artifact["total"] == 2090 and "₹2090" in message.content
  assert TOOLS["escalate_to_human"].return_direct


def test_loyalty_points_by_order_or_amount():
  assert "209 points" in call("loyalty_points", order_id="NC-12500").content
  assert "130 points" in call("loyalty_points", amount_rupees=1300).content


def test_scoring_whole_numbers_and_arg_options():
  request = {
    "expect": [{"name": "loyalty_points", "args_any": [{"order_id": "NC-12500"}, {"amount_rupees": 2090}]}],
    "answer_numbers": [209],
  }
  run = ToolRun(
    "It earns 2090 rupees of value.",
    [AIMessage("x")],
    1,
    [{"name": "loyalty_points", "args": {"order_id": "nc-12500"}}],
  )
  assert call_matches(request["expect"][0], run.tool_calls[0])
  assert not score(request, run)["answer_ok"]  # 2090 is not 209
  run.answer = "That earns 209 points."
  assert score(request, run)["success"]


def test_terse_tools_keep_behaviour_without_descriptions():
  bare = {t.name: t for t in terse(list(TOOLS.values()))}
  assert bare["check_refund"].description == "check refund"
  assert bare["get_order"].invoke({"order_id": "NC-4410"}).startswith("NC-4410")
