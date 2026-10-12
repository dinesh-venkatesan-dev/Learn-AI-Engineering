"""Text to SQL as a LangGraph workflow (Data Module 2): fixed, predictable steps; the model fills in each one.

pick_tables → write_sql → run ─(error, tries left)→ fix_sql → run ... → answer

pick_tables   the model picks the tables it needs (structured output); write_sql sees only their schemas
write_sql     schemas + the glossary + examples; small reference tables (cafés, products) are shown in FULL, so
              facts like a café's opening date can be looked up instead of guessed
run           the guard; with review=True, queries that read customer data wait for a data steward (interrupt)
answer        empty or all-zero results are reported as such, in code, never dressed up as an answer

workflow = build_sql_workflow(model)                                   # one-shot questions
workflow = build_sql_workflow(model, review=True, checkpointer=saver)  # stewarded
"""

import sqlite3
from typing import TypedDict

import sqlglot
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from nimbus.data.context import EXAMPLES, GLOSSARY
from nimbus.data.guard import UnsafeQuery, run_safe
from nimbus.data.sql import query
from nimbus.data.text_to_sql import SQLQuery
from nimbus.data.tools import table_description, table_names

MAX_TRIES = 3
REFERENCE_ROWS = 20  # tables this small go to the model in full
RELATIONS = "orders→customers, orders→cafes, order_items→orders, order_items→products, reviews→orders"


class SQLState(TypedDict, total=False):
  question: str
  tables: list[str]
  sql: str
  error: str | None
  tries: int
  attempts: list[str]
  columns: list[str]
  rows: list[tuple]
  answer: str


class Tables(BaseModel):
  tables: list[str] = Field(description="The tables needed to answer the question")


def needs_review(sql: str) -> bool:
  """The stewardship policy, in code: queries that read customer records go to a person first."""
  try:
    tables = {t.name.lower() for t in sqlglot.parse_one(sql, dialect="sqlite").find_all(sqlglot.exp.Table)}
  except sqlglot.errors.ParseError:
    return False  # the guard refuses unparseable SQL anyway
  return "customers" in tables


def schema_for(tables: list[str]) -> str:
  parts = []
  for table in tables:
    part = table_description(table)
    count = query(f"SELECT COUNT(*) FROM {table}")[1][0][0]
    if count <= REFERENCE_ROWS:
      part += f"\nall {count} rows: {query(f'SELECT * FROM {table}')[1]}"
    parts.append(part)
  return "\n\n".join(parts)


def build_sql_workflow(model: BaseChatModel, *, review: bool = False, checkpointer=None, reference_rows: bool = True):
  pick = model.with_structured_output(Tables, method="json_schema")
  write = model.with_structured_output(SQLQuery, method="json_schema")

  def pick_tables(state: SQLState) -> dict:
    names = table_names()
    prompt = (
      f"Tables: {', '.join(names)}. Relations: {RELATIONS}.\nWhich tables are needed to answer: {state['question']}"
    )
    tables = [t for t in pick.invoke(prompt).tables if t in names]
    return {"tables": tables or names, "tries": 0, "attempts": []}

  def write_sql(state: SQLState) -> dict:
    schemas = schema_for(state["tables"]) if reference_rows else "\n\n".join(map(table_description, state["tables"]))
    system = f"Write ONE SQLite query for Nimbus Coffee.\n\nTables:\n{schemas}\n\n{GLOSSARY}\n{EXAMPLES}"
    return {"sql": write.invoke([SystemMessage(system), HumanMessage(state["question"])]).sql}

  def run(state: SQLState) -> dict:
    sql = state["sql"]
    attempts = [*state.get("attempts", []), sql]
    if review and needs_review(sql):
      decision = interrupt({"question": state["question"], "sql": sql, "reason": "reads customer data"})
      if decision["type"] == "reject":
        return {"error": f"A data steward declined this query: {decision['reason']}", "tries": MAX_TRIES}
      if decision["type"] == "edit":
        sql = decision["sql"]
    try:
      columns, rows = run_safe(sql)
    except (UnsafeQuery, sqlite3.Error) as error:
      return {"error": str(error), "tries": state["tries"] + 1, "attempts": attempts}
    return {"sql": sql, "columns": columns, "rows": rows, "error": None, "attempts": attempts}

  def fix_sql(state: SQLState) -> dict:
    prompt = (
      f"{GLOSSARY}\nThis SQLite query failed.\nQuestion: {state['question']}\nQuery: {state['sql']}\n"
      f"Error: {state['error']}\nWrite a corrected query."
    )
    return {"sql": write.invoke(prompt).sql}

  def answer(state: SQLState) -> dict:
    if state.get("error"):
      return {"answer": f"I couldn't answer that: {state['error']}"}
    rows = state["rows"]
    if not rows or all(value in (0, None) for row in rows for value in row):  # a rule in code, not in the prompt
      return {"answer": f"No matching data: the query's filter may be wrong, so this isn't an answer. ({state['sql']})"}
    reply = model.invoke(
      "Answer in one sentence using ONLY this result. Quote numbers exactly; money in rupees.\n"
      f"Question: {state['question']}\nColumns: {state['columns']}\nRows: {rows[:20]}"
    )
    return {"answer": reply.text}

  def after_run(state: SQLState) -> str:
    return "fix_sql" if state.get("error") and state["tries"] < MAX_TRIES else "answer"

  builder = StateGraph(SQLState)
  for node in (pick_tables, write_sql, run, fix_sql, answer):
    builder.add_node(node)
  builder.add_edge(START, "pick_tables")
  builder.add_edge("pick_tables", "write_sql")
  builder.add_edge("write_sql", "run")
  builder.add_conditional_edges("run", after_run, ["fix_sql", "answer"])
  builder.add_edge("fix_sql", "run")
  builder.add_edge("answer", END)
  return builder.compile(checkpointer=checkpointer)


class WorkflowAnswer:
  """A finished run's state, shaped like the other systems' answers for evaluation."""

  def __init__(self, state: dict):
    self.sql = None if state.get("error") else state.get("sql")
    self.rows = state.get("rows") or []
    self.answer = state.get("answer", "")
    self.attempts = state.get("attempts", [])
    self.tables = state.get("tables", [])


def ask_workflow(workflow, question: str, config: dict | None = None) -> WorkflowAnswer:
  return WorkflowAnswer(workflow.invoke({"question": question}, config))
