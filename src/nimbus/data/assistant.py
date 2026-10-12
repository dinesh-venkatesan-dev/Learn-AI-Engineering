"""Nimbus's analytics assistant as it ships (Data Module 3, the capstone): metrics first, SQL when needed, charts,
checked numbers.

  get_metric                  governed metrics (the semantic layer): the same definitions everywhere
  describe_table, run_query   only when no metric fits; every query through the guard
  chart_metric, make_chart    charts drawn by our code; a metric's chart uses the metric's own SQL
  grounding_check             every number in the answer must appear in some tool output, or it's flagged

  assistant = AnalyticsAssistant(model)
  reply = assistant.ask("How did revenue develop month by month? Chart it.")
"""

import re
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from nimbus.data.charts import ChartSpec, metric_chart_spec, render
from nimbus.data.context import GLOSSARY
from nimbus.data.guard import run_safe
from nimbus.data.metrics import DIMENSIONS, METRICS, Dimension, Metric, get_metric
from nimbus.data.tools import describe_table, run_query

SYSTEM = (
  "You are Nimbus Coffee's analytics assistant.\n"
  f"1. For standard numbers, use get_metric (metrics: {', '.join(METRICS)}; dimensions: {', '.join(DIMENSIONS)}).\n"
  "2. Only if no metric fits, explore with describe_table and run_query.\n"
  "3. If a chart is asked for: chart_metric for standard metrics, make_chart only for anything else.\n"
  "Every number in your answer must be copied from a tool result: don't calculate new numbers yourself.\n"
  f"{GLOSSARY}\nAnswer in at most four sentences."
)


def numbers_in(text: str) -> list[float]:
  return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text) if re.fullmatch(r"[\d,]+\.?\d*", n)]


def grounding_check(answer: str, tool_outputs: list[str]) -> tuple[list[float], list[float]]:
  """Which numbers in the answer appear in some tool output (within 0.5%)? Years and small counts are skipped."""
  available = [n for output in tool_outputs for n in numbers_in(output)]
  grounded, ungrounded = [], []
  for number in numbers_in(answer):
    if (number < 10 and number.is_integer()) or 2000 <= number <= 2100:  # "3 cafés", "2026": not claims to check
      continue
    hit = any(abs(number - a) <= max(0.01, abs(a) * 0.005) for a in available)
    (grounded if hit else ungrounded).append(number)
  return grounded, ungrounded


@dataclass
class AssistantReply:
  question: str
  answer: str
  tools: list[str] = field(default_factory=list)
  outputs: list[str] = field(default_factory=list)
  charts: list[str] = field(default_factory=list)
  grounded: list[float] = field(default_factory=list)
  ungrounded: list[float] = field(default_factory=list)


class AnalyticsAssistant:
  def __init__(self, model: BaseChatModel, metrics_first: bool = True):
    """metrics_first=False is the comparison: the same assistant with SQL tools only (Module 2's agent + charts)."""
    self.charts: list[str] = []
    tools = [describe_table, run_query, self._make_chart_tool()]
    if metrics_first:
      tools = [get_metric, self._chart_metric_tool(), *tools]
    self.agent = create_agent(model, tools, system_prompt=SYSTEM if metrics_first else SQL_ONLY)

  def _chart_metric_tool(self):
    charts = self.charts

    @tool
    def chart_metric(metric: Metric, dimension: Dimension, title: str) -> str:
      """Chart a governed metric by one dimension. Prefer this for standard metrics: the chart then uses exactly the
      same definition as get_metric."""
      spec = metric_chart_spec(metric, dimension, title)
      charts.append(render(spec, f"assistant_{len(charts) + 1}.png"))
      _, rows = run_safe(spec.sql)
      return f"chart saved: {charts[-1]}\ndata: {rows}"  # the numbers too, so the answer can quote them

    return chart_metric

  def _make_chart_tool(self):
    charts = self.charts

    @tool
    def make_chart(spec: ChartSpec) -> str:
      """Draw a chart of anything else from a spec (kind, title, sql returning x then y, axis labels)."""
      try:
        charts.append(render(spec, f"assistant_{len(charts) + 1}.png"))
        _, rows = run_safe(spec.sql)
      except Exception as error:  # noqa: BLE001 - the model reads it and fixes the spec
        return f"ERROR: {error}"
      return f"chart saved: {charts[-1]}\ndata: {rows}"

    return make_chart

  def ask(self, question: str, config: dict | None = None) -> AssistantReply:
    before = len(self.charts)
    result = self.agent.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 30, **(config or {})})
    reply = AssistantReply(question, result["messages"][-1].text, charts=self.charts[before:])
    for message in result["messages"]:
      reply.tools += [c["name"] for c in getattr(message, "tool_calls", None) or []]
      if isinstance(message, ToolMessage):
        reply.outputs.append(message.text)
    reply.grounded, reply.ungrounded = grounding_check(reply.answer, reply.outputs)
    return reply


SQL_ONLY = (
  "You are Nimbus Coffee's analytics assistant. Explore with describe_table, then answer with run_query. If a chart "
  "is asked for, use make_chart. Every number in your answer must be copied from a tool result: don't calculate new "
  f"numbers yourself.\n{GLOSSARY}\nAnswer in at most four sentences."
)
