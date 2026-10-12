"""Reading the warehouse: a read-only connection, the schema as a model sees it, and checking answers."""

import sqlite3
from contextlib import closing
from functools import cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from nimbus.data.warehouse import DB_PATH, build_warehouse


@cache
def engine() -> Engine:
  """Read-only: SQLite opened with mode=ro, so even a DROP TABLE that slips through can't change anything."""
  build_warehouse()
  # NullPool: a fresh connection per use; SQLite connections must stay in the thread that made them.
  return create_engine(
    "sqlite://", poolclass=NullPool, creator=lambda: sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
  )


def query(sql: str, limit: int | None = None) -> tuple[list[str], list[tuple]]:
  with engine().connect() as connection:
    result = connection.execute(text(sql))
    columns = list(result.keys())
    rows = result.fetchmany(limit) if limit else result.fetchall()
  return columns, [tuple(row) for row in rows]


@cache
def schema_text(sample_rows: int = 3, ddl: bool = True) -> str:
  """What a model is told about the database: CREATE TABLE statements (or just column names) and sample rows."""
  build_warehouse()
  with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as connection:  # sqlite3's `with` doesn't close
    parts = []
    for name, create in connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY rowid"):
      if ddl:
        part = " ".join(create.split()) + ";"
      else:
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({name})")]
        part = f"{name}({', '.join(columns)})"
      if sample_rows:
        rows = connection.execute(f"SELECT * FROM {name} LIMIT {sample_rows}").fetchall()
        part += "\n/* sample rows:\n" + "\n".join(f"  {row}" for row in rows) + "\n*/"
      parts.append(part)
  return "\n\n".join(parts)


def matches(right, rows: list[tuple], normalize: bool = True) -> bool:
  """Does a query result contain the right answer? Numbers within 0.1%, text case-insensitively. normalize=True also
  accepts two harmless differences: numbers sent as text ('0' vs 0) and a rate as a percentage (25.44 ≈ 0.2544)."""

  def as_number(value):
    if isinstance(value, int | float):
      return float(value)
    if not normalize and isinstance(value, str):
      return None
    try:
      return float(value)
    except (TypeError, ValueError):
      return None

  cells = [value for row in rows for value in row]
  target = as_number(right)
  if target is None:
    return any(str(right).casefold() == str(v).casefold() for v in cells)
  candidates = [target, target * 100] if normalize and 0 < abs(target) < 1 else [target]
  return any(
    number is not None and abs(number - t) <= max(0.001, abs(t) * 0.001)
    for v in cells
    for number in [as_number(v)]
    for t in candidates
  )
