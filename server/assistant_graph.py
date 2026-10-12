"""The Nimbus assistant for LangGraph Server: every hosted graph is also an MCP tool at /mcp.

Its input and output schemas ARE that tool's interface: `question` (and `customer_id`) in, `answer` out. Without
them an MCP caller would receive the whole internal state.
"""

from typing import TypedDict

from langgraph.graph import START, StateGraph


class Question(TypedDict):
  question: str
  customer_id: str


class Answer(TypedDict):
  answer: str


class State(Question, Answer):
  pass


def answer(state: State) -> dict:
  from nimbus.mcp.assistant_server import _assistant

  return {"answer": _assistant()(state.get("customer_id") or "c-1042", state["question"]).reply}


graph = (
  StateGraph(State, input_schema=Question, output_schema=Answer)
  .add_node("answer", answer)
  .add_edge(START, "answer")
  .compile(name="nimbus_assistant")
)
