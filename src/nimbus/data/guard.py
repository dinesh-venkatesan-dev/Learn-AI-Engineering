"""Model-written SQL is untrusted input (Data Module 1). Every query a model writes runs through run_safe().

Defence in depth, each layer catching what the one before misses:
  1. parse it (sqlglot): exactly ONE statement, and it must be a read (SELECT, WITH ... SELECT, UNION)
  2. allow-listed tables only; sensitive columns refused (customer names are personal data), also via SELECT *
  3. rows capped: LIMIT added when missing
  4. a READ-ONLY connection with a time limit: the database itself refuses a write that slipped through
Refusals raise UnsafeQuery with a message a model can read and act on.
"""

import sqlite3
import time
from contextlib import closing

import sqlglot
from sqlglot import exp

from nimbus.data.warehouse import DB_PATH, build_warehouse

ALLOWED_TABLES = {"cafes", "products", "customers", "orders", "order_items", "reviews"}
BLOCKED_COLUMNS = {("customers", "name")}  # personal data: never returned to a chat window
MAX_ROWS = 200
WRITES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)


class UnsafeQuery(ValueError):
  """The query was refused before it ran. The message says why, so a model can try again."""


def check_sql(sql: str) -> str:
  """Validate and rewrite a query; return the SQL that will actually run, or raise UnsafeQuery."""
  try:
    statements = [s for s in sqlglot.parse(sql, dialect="sqlite") if s is not None]
  except sqlglot.errors.ParseError as error:
    raise UnsafeQuery(f"Not valid SQL: {str(error).splitlines()[0]}") from error
  if len(statements) != 1:
    raise UnsafeQuery(f"Exactly one statement allowed, got {len(statements)}.")
  tree = statements[0]
  if not isinstance(tree, exp.Select | exp.Union):
    first_word = sql.strip().split()[0].upper() if sql.strip() else "nothing"
    raise UnsafeQuery(f"Only SELECT queries are allowed (this one starts with {first_word}).")
  for node in tree.walk():
    if isinstance(node, WRITES):
      raise UnsafeQuery(f"{node.key.upper()} is not allowed.")
  ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
  aliases = {}
  for table in tree.find_all(exp.Table):
    name = table.name.lower()
    if name not in ALLOWED_TABLES and name not in ctes:
      raise UnsafeQuery(f"Table {table.name!r} is not available.")
    aliases[(table.alias_or_name or name).lower()] = name
  for column in tree.find_all(exp.Column):
    owner = aliases.get(column.table.lower(), column.table.lower()) if column.table else None
    candidates = [owner] if owner else list(aliases.values())
    if any((t, column.name.lower()) in BLOCKED_COLUMNS for t in candidates):
      raise UnsafeQuery(f"Column {column.name!r} contains personal data and can't be queried.")
  selects_everything = any(  # SELECT * or c.* (COUNT(*) is fine)
    isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star))
    for select in tree.find_all(exp.Select)
    for p in select.expressions
  )
  if selects_everything and "customers" in aliases.values():
    raise UnsafeQuery("SELECT * on customers would return personal data: name the columns you need.")
  if isinstance(tree, exp.Select) and not tree.args.get("limit"):
    tree = tree.limit(MAX_ROWS)
  return tree.sql(dialect="sqlite")


def run_safe(sql: str, timeout_seconds: float = 2.0) -> tuple[list[str], list[tuple]]:
  """check_sql, then run on a read-only connection that stops the query after `timeout_seconds`."""
  checked = check_sql(sql)
  build_warehouse()
  with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as connection:  # sqlite3's `with` doesn't close
    deadline = time.monotonic() + timeout_seconds
    connection.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)  # non-zero → abort the query
    try:
      cursor = connection.execute(checked)
      return [d[0] for d in cursor.description], cursor.fetchmany(MAX_ROWS)
    except sqlite3.OperationalError as error:
      if "interrupted" in str(error):
        raise UnsafeQuery(f"Query took longer than {timeout_seconds}s and was stopped.") from error
      raise
