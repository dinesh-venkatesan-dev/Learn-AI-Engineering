"""A SQL agent (Data Module 2): it explores the warehouse with tools, queries, checks, and answers in as many steps
as the question needs. The glossary gives meaning, the tools give structure, the guard keeps every query a read.

  agent = build_sql_agent(model)
  result = ask_agent(agent, "Which loyalty tier has the highest average order value?")
"""

import json
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from nimbus.data.context import GLOSSARY
from nimbus.data.tools import SQL_TOOLS

SYSTEM = (
  "You are Nimbus Coffee's data analyst. Explore the warehouse with your tools before answering: list the tables, "
  "describe the ones you need, then run queries. Never guess a number: every number in your answer must come from a "
  "query result. Say in one short clause what you counted (e.g. which order statuses, which months), so a reader can "
  f"spot a misreading.\n\n{GLOSSARY}\nAnswer in at most three sentences."
)


def build_sql_agent(model: BaseChatModel, checkpointer=None):
  return create_agent(model, SQL_TOOLS, system_prompt=SYSTEM, checkpointer=checkpointer)


@dataclass
class AgentAnswer:
  """The agent's answer and the work behind it: every query it ran, and every row it saw."""

  question: str
  answer: str
  queries: list[str] = field(default_factory=list)
  rows: list[tuple] = field(default_factory=list)
  attempts: list[str] = field(default_factory=list)  # every run_query call, failed ones included
  sql: str | None = None  # the last query that ran


def ask_agent(agent, question: str, config: dict | None = None) -> AgentAnswer:
  result = agent.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 30, **(config or {})})
  out = AgentAnswer(question, result["messages"][-1].text)
  calls = {c["id"]: c["args"].get("sql", "") for m in result["messages"] for c in getattr(m, "tool_calls", None) or []}
  for message in result["messages"]:
    if isinstance(message, ToolMessage) and message.name == "run_query":
      sql = calls.get(message.tool_call_id, "")
      out.attempts.append(sql)
      if message.text.startswith("{"):
        out.queries.append(sql)
        out.rows += [tuple(r) for r in json.loads(message.text)["rows"]]
        out.sql = sql
  return out
