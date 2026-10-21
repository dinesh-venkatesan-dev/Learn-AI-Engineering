"""The "What the model saw" recorder: every chat-model call inside a lab run, as the model received it."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from academy.labs.model_calls import MAX_CHARS, recording
from academy.labs.runtime import Lab


def fake(*replies: AIMessage) -> GenericFakeChatModel:
  return GenericFakeChatModel(messages=iter(replies))


def test_records_each_message_with_its_role_and_the_reply():
  model = fake(AIMessage("24 days."))
  with recording() as recorder:
    model.invoke([SystemMessage("Answer briefly."), HumanMessage("How much leave?")])
  [call] = recorder.result()["calls"]
  assert [(m["role"], m["text"]) for m in call["messages"]] == [
    ("system", "Answer briefly."),
    ("human", "How much leave?"),
  ]
  assert call["reply"]["text"] == "24 days."
  assert call["seconds"] >= 0


def test_records_tool_calls_and_tool_results():
  asks = AIMessage("", tool_calls=[{"name": "get_order", "args": {"order_id": "NC-1"}, "id": "c1"}])
  model = fake(asks, AIMessage("Shipped."))
  with recording() as recorder:
    first = model.invoke("Where is NC-1?")
    model.invoke([HumanMessage("Where is NC-1?"), first, ToolMessage("shipped", tool_call_id="c1", name="get_order")])
  ask, answer = recorder.result()["calls"]
  assert ask["reply"]["tool_calls"] == [{"name": "get_order", "args": {"order_id": "NC-1"}}]
  assert [m["role"] for m in answer["messages"]] == ["human", "ai", "tool"]
  assert answer["messages"][2]["name"] == "get_order"


def test_only_calls_inside_the_block_are_recorded():
  model = fake(AIMessage("a"), AIMessage("b"))
  model.invoke("before")
  with recording() as recorder:
    model.invoke("inside")
  assert [c["messages"][0]["text"] for c in recorder.result()["calls"]] == ["inside"]


def test_a_huge_message_is_cut():
  model = fake(AIMessage("ok"))
  with recording() as recorder:
    model.invoke("x" * (MAX_CHARS + 500))
  text = recorder.result()["calls"][0]["messages"][0]["text"]
  assert text.startswith("x" * 100) and text.endswith("(500 more characters)")


def test_every_lab_result_carries_its_model_calls():
  model = fake(AIMessage("hi"))
  lab = Lab("t.fake", "Fake", lambda: [model.invoke("hello") and {"type": "markdown", "text": "done"}])
  result = lab.run({})
  assert result["model_calls"]["total"] == 1
  assert result["model_calls"]["calls"][0]["reply"]["text"] == "hi"
