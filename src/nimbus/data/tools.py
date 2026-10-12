"""The SQL tools Nimbus's data agents use (Data Module 2). Every query runs through the guard (nimbus.data.guard).

list_tables      tables and row counts
describe_table   columns, types, joins and 3 sample rows (personal data hidden)
run_query        one read-only SELECT; errors and refusals come back as text, so the agent can fix the query
"""

import json
import sqlite3

from langchain_core.tools import tool
from sqlalchemy import inspect

from nimbus.data.guard import ALLOWED_TABLES, UnsafeQuery, run_safe
from nimbus.data.sql import engine, query


def table_names() -> list[str]:
  return [t for t in inspect(engine()).get_table_names() if t in ALLOWED_TABLES]


def table_description(table: str) -> str:
  inspector = inspect(engine())
  if table not in table_names():
    return f"No table {table!r}. Tables: {', '.join(table_names())}"
  columns = ", ".join(f"{c['name']} {c['type']}" for c in inspector.get_columns(table))
  joins = "; ".join(
    f"{fk['constrained_columns'][0]} → {fk['referred_table']}" for fk in inspector.get_foreign_keys(table)
  )
  try:
    _, rows = run_safe(f"SELECT * FROM {table} LIMIT 3")
  except UnsafeQuery:
    rows = ["(sample rows hidden: personal data)"]
  return f"{table}({columns})\njoins: {joins or 'none'}\nsample rows: {rows}"


@tool
def list_tables() -> str:
  """List the tables in the Nimbus warehouse, with their row counts."""
  return "\n".join(f"{t} ({query(f'SELECT COUNT(*) FROM {t}')[1][0][0]} rows)" for t in table_names())


@tool
def describe_table(table: str) -> str:
  """Columns, types, joins and 3 sample rows of one table. Call it before writing SQL that uses the table."""
  return table_description(table)


def query_as_text(sql: str) -> str:
  """Run ONE read-only SQLite SELECT and get the rows as JSON (at most 200 rows). Errors come back as text: fix the
  query and try again."""
  try:
    columns, rows = run_safe(sql)
  except (UnsafeQuery, sqlite3.Error) as error:
    return f"ERROR: {error}"  # the agent reads it and fixes the query
  return json.dumps({"columns": columns, "rows": rows[:50], "row_count": len(rows)}, default=str)


run_query = tool("run_query")(query_as_text)


SQL_TOOLS = [list_tables, describe_table, run_query]
