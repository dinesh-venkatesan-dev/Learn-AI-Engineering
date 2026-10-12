"""The support assistant's tool loop, written out by hand: model → tool calls → run tools → results → model → ...

    messages = [system, question]
    repeat:
      reply = model_with_tools.invoke(messages)        the model answers, or asks for tools
      no tool calls?  → that's the answer
      otherwise run every requested tool, append its ToolMessage, and ask again
      (a return_direct tool's output IS the answer: stop without another model call)

LangChain Module 4 replaces this loop with create_agent; writing it once shows what an agent is.
"""

import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from nimbus.llm.usage import Usage

SYSTEM = (
  "You are Nimbus Coffee's support assistant. Use the tools to look things up; never guess order details, prices or "
  "refund eligibility. Answer in two or three short sentences, in the customer's language."
)


@dataclass
class ToolRun:
  answer: str
  messages: list[BaseMessage]
  rounds: int  # model calls made
  tool_calls: list[dict] = field(default_factory=list)  # every call the model asked for: {"name", "args"}
  artifacts: list = field(default_factory=list)  # full records the tools kept for the app
  direct: bool = False  # ended by a return_direct tool
  ms: float = 0.0

  def usage(self) -> Usage:
    usage = Usage()
    for m in self.messages:
      if isinstance(m, AIMessage):
        usage.add(m)
    return usage


def answer_with_tools(
  model: BaseChatModel, tools: list[BaseTool], question: str, system: str = SYSTEM, max_rounds: int = 5
) -> ToolRun:
  by_name = {t.name: t for t in tools}
  bound = model.bind_tools(tools)  # the model now knows the tools' schemas; it still never runs them
  messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(question)]
  calls, artifacts, started = [], [], time.perf_counter()

  def done(answer: str, rounds: int, direct: bool = False) -> ToolRun:
    return ToolRun(answer, messages, rounds, calls, artifacts, direct, (time.perf_counter() - started) * 1000)

  for round_ in range(1, max_rounds + 1):
    reply = bound.invoke(messages)
    messages.append(reply)
    if not reply.tool_calls:
      return done(reply.text, round_)
    for call in reply.tool_calls:  # several calls in one reply: independent lookups, run them all
      calls.append({"name": call["name"], "args": call["args"]})
      tool = by_name.get(call["name"])
      if tool is None:
        result = ToolMessage(f"Unknown tool {call['name']}.", tool_call_id=call["id"], status="error")
      else:
        result = tool.invoke(call)  # a ToolCall in → a ToolMessage out (with the artifact, if any)
      messages.append(result)
      if getattr(result, "artifact", None) is not None:
        artifacts.append(result.artifact)
      if tool is not None and tool.return_direct:
        return done(result.text, round_, direct=True)
  return done("Sorry, I couldn't finish that. A teammate will follow up.", max_rounds)


async def answer_with_tools_async(
  model: BaseChatModel, tools: list[BaseTool], question: str, system: str = SYSTEM, max_rounds: int = 5
) -> ToolRun:
  """The same loop with ainvoke: for async-only tools, e.g. tools served over MCP."""
  by_name = {t.name: t for t in tools}
  bound = model.bind_tools(tools)
  messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(question)]
  calls, artifacts, started = [], [], time.perf_counter()

  def done(answer: str, rounds: int, direct: bool = False) -> ToolRun:
    return ToolRun(answer, messages, rounds, calls, artifacts, direct, (time.perf_counter() - started) * 1000)

  for round_ in range(1, max_rounds + 1):
    reply = await bound.ainvoke(messages)
    messages.append(reply)
    if not reply.tool_calls:
      return done(reply.text, round_)
    for call in reply.tool_calls:
      calls.append({"name": call["name"], "args": call["args"]})
      tool = by_name.get(call["name"])
      if tool is None:
        result = ToolMessage(f"Unknown tool {call['name']}.", tool_call_id=call["id"], status="error")
      else:
        result = await tool.ainvoke(call)
      messages.append(result)
      if getattr(result, "artifact", None) is not None:
        artifacts.append(result.artifact)
      if tool is not None and tool.return_direct:
        return done(result.text, round_, direct=True)
  return done("Sorry, I couldn't finish that. A teammate will follow up.", max_rounds)
