"""The support agent's wiring and scoring, without calling a model."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from nimbus.llm.usage import Usage
from nimbus.support.agent import SupportOutcome, agent_tools, build_support_agent
from nimbus.support.agent_eval import AgentRun, score
from nimbus.support.tools import SupportStore


class ToolFake(GenericFakeChatModel):
  def bind_tools(self, tools, **kwargs):  # the fake can't call tools; building the graph only needs this to exist
    return self


def test_agent_graph_and_tools():
  agent = build_support_agent(ToolFake(messages=iter([AIMessage("hi")])), SupportStore())
  assert {"model", "tools"} <= set(agent.get_graph().nodes)
  names = {t.name for t in agent_tools(SupportStore())}
  assert {"my_orders", "pause_subscription", "escalate", "get_order"} <= names and "escalate_to_human" not in names


def test_store_lists_a_customers_orders_by_date():
  orders = SupportStore().orders_of("c-1042")
  assert [o["order_id"] for o in orders] == ["NC-20417", "NC-12093"]  # oldest first; my_orders reverses it


def run(reply: str, calls=(), actions=(), **outcome) -> AgentRun:
  fields = {"resolved": True, "needs_human": False, "order_ids": [], **outcome}
  return AgentRun(SupportOutcome(reply=reply, **fields), list(actions), list(calls), 2, Usage(), 1.0)


def test_scoring_actions_outcome_and_reply():
  scenario = {
    "expect": [{"name": "escalate", "args": {"urgency": "urgent"}}],
    "also_ok": ["get_order"],
    "actions": ["escalate(urgent)"],
    "outcome": {"needs_human": True},
  }
  good = run(
    "A teammate will call you.",
    [{"name": "escalate", "args": {"reason": "x", "urgency": "urgent"}}],
    ["escalate(urgent): insects"],
    needs_human=True,
  )
  assert score(scenario, good)["success"]
  wrong_urgency = run(
    "…", [{"name": "escalate", "args": {"urgency": "high"}}], ["escalate(high): insects"], needs_human=True
  )
  assert not score(scenario, wrong_urgency)["tools_ok"]


def test_no_actions_and_curly_apostrophes():
  scenario = {
    "expect": [],
    "also_ok": ["my_subscription"],
    "answer_any": ["don't have a subscription"],
    "no_actions": True,
  }
  assert score(scenario, run("You don’t have a subscription."))["success"]
  assert not score(scenario, run("You don’t have a subscription.", actions=["pause_subscription(4 weeks)"]))[
    "actions_ok"
  ]
