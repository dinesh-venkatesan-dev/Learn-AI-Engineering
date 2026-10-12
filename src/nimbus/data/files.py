"""Data that isn't in the warehouse (Data Module 3): files from suppliers, HR and finance, queried with DuckDB.

build_files() writes them once to .cache/data/files/:
  deliveries.csv        one row per supplier delivery: supplier, cafe, promised_on, delivered_on, items
  staff_shifts.xlsx     the September staff rota: cafe, shift_date, staff, hours
  sales_export.parquet  a monthly finance export: month, cafe, revenue

DuckDB reads CSV and Parquet in SQL (FROM 'deliveries.csv'); Excel goes through pandas and is registered as a table.
The files tools run DuckDB SQL through check_duckdb (SELECT only, no COPY / ATTACH / INSTALL / read_*(), files by
name only), on a connection the DATABASE limits to the data folder (allowed_directories, locked).
"""

import csv
import random
import sqlite3
from contextlib import closing
from datetime import date, timedelta

import sqlglot
from langchain_core.tools import tool
from sqlglot import exp

from nimbus.config import PROJECT_DIR
from nimbus.data.warehouse import CAFES, DB_PATH, build_warehouse

FILES_DIR = PROJECT_DIR / ".cache" / "data" / "files"
SUPPLIERS = {"Green Valley Dairy": 0.30, "Nilgiri Co-op": 0.08, "Coastal Bakers": 0.15}  # chance of being late
FILE_WRITES = (exp.Copy, exp.Attach, exp.Command, exp.Insert, exp.Create, exp.Drop, exp.Update, exp.Delete)


def build_files() -> None:
  if (FILES_DIR / "sales_export.parquet").exists():
    return
  import duckdb
  import openpyxl

  build_warehouse()
  FILES_DIR.mkdir(parents=True, exist_ok=True)
  rng = random.Random(41)
  with open(FILES_DIR / "deliveries.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["supplier", "cafe", "promised_on", "delivered_on", "items"])
    day = date(2026, 7, 1)
    while day <= date(2026, 9, 30):
      for supplier, late_chance in SUPPLIERS.items():
        if day.weekday() in (0, 3):  # deliveries on Mondays and Thursdays
          for cafe in CAFES[:5]:
            delay = rng.choice([1, 1, 2, 3]) if rng.random() < late_chance else 0
            writer.writerow(
              [supplier, cafe[1], day.isoformat(), (day + timedelta(days=delay)).isoformat(), rng.randint(20, 60)]
            )
      day += timedelta(days=1)
  book = openpyxl.Workbook()
  sheet = book.active
  sheet.title = "September rota"
  sheet.append(["cafe", "shift_date", "staff", "hours"])
  staff = ["Asha", "Ben", "Chitra", "Dev", "Esha", "Farid", "Gita", "Hari"]
  day = date(2026, 9, 1)
  while day <= date(2026, 9, 30):
    for cafe in CAFES:
      for person in rng.sample(staff, 3 if cafe[0] in (1, 3) else 2):
        sheet.append([cafe[1], day.isoformat(), person, rng.choice([6, 8, 8, 9])])
    day += timedelta(days=1)
  book.save(FILES_DIR / "staff_shifts.xlsx")
  book.close()
  with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as connection:
    rows = connection.execute(
      "SELECT strftime('%Y-%m', o.ordered_at), c.name, SUM(oi.quantity * oi.unit_price) FROM orders o JOIN order_items "
      "oi USING (order_id) JOIN cafes c USING (cafe_id) WHERE o.status = 'completed' GROUP BY 1, 2"
    ).fetchall()
  with duckdb.connect() as con:
    con.execute("CREATE TABLE export (month VARCHAR, cafe VARCHAR, revenue BIGINT)")
    con.executemany("INSERT INTO export VALUES (?, ?, ?)", rows)
    con.execute(f"COPY export TO '{FILES_DIR / 'sales_export.parquet'}' (FORMAT parquet)")


def connect():
  """A fresh in-process DuckDB connection that can read ONLY the data folder. Use it in a `with` block, so it's
  closed after."""
  import duckdb
  import pandas as pd

  build_files()
  con = duckdb.connect()
  con.register("staff_shifts", pd.read_excel(FILES_DIR / "staff_shifts.xlsx"))  # Excel → a table (read by pandas)
  con.execute(f"SET allowed_directories = ['{FILES_DIR}/']")  # the database itself refuses any other path
  con.execute("SET enable_external_access = false")
  con.execute("SET lock_configuration = true")  # and a query can't switch that back
  return con


def check_duckdb(sql: str) -> str:
  """SELECT only, nothing that writes or names a path; file names become absolute paths in the data folder. Raises
  ValueError with the reason."""
  try:
    statements = [s for s in sqlglot.parse(sql, dialect="duckdb") if s is not None]
  except sqlglot.errors.ParseError as error:
    raise ValueError(f"Not valid SQL: {str(error).splitlines()[0]}") from error
  if len(statements) != 1 or not isinstance(statements[0], exp.Select | exp.Union):
    raise ValueError("Exactly one SELECT query is allowed.")
  tree = statements[0]
  for node in tree.walk():
    if isinstance(node, FILE_WRITES):
      raise ValueError(f"{node.key.upper()} is not allowed.")
    if isinstance(node, exp.Func) and node.sql_name().lower().startswith(("read_", "glob", "install", "load")):
      raise ValueError(
        f"{node.sql_name()}() is not allowed: name the file in FROM instead, e.g. FROM 'deliveries.csv'."
      )
  for table in tree.find_all(exp.Table):
    if "/" in table.name or ".." in table.name or "\\" in table.name:
      raise ValueError(f"Only files in the data folder can be read, by name: not {table.name!r}.")
    if table.name.endswith((".csv", ".parquet")):
      table.set("this", exp.Identifier(this=str(FILES_DIR / table.name), quoted=True))
  return tree.sql(dialect="duckdb")


@tool
def list_files() -> str:
  """The data files you can query, plus the table staff_shifts (from the Excel rota)."""
  build_files()
  names = "\n".join(f"'{p.name}'" for p in sorted(FILES_DIR.iterdir()) if p.suffix in (".csv", ".parquet"))
  return names + "\ntable staff_shifts (from staff_shifts.xlsx)"


@tool
def describe_file(source: str) -> str:
  """Columns and 3 rows of a file (e.g. 'deliveries.csv') or of the table staff_shifts."""
  name = source.strip().strip("'")
  if name != "staff_shifts" and (name not in {p.name for p in FILES_DIR.iterdir()} or "/" in name):
    return f"No file {name!r}. Use list_files."
  with connect() as con:
    frame = con.sql(check_duckdb(f"SELECT * FROM {name if name == 'staff_shifts' else repr(name)} LIMIT 3")).df()
  return f"columns: {dict(frame.dtypes.astype(str))}\n{frame.to_string()}"


@tool
def run_duckdb(sql: str) -> str:
  """Run ONE DuckDB SELECT. Files are read by name in quotes, e.g. SELECT * FROM 'deliveries.csv'."""
  try:
    with connect() as con:
      frame = con.sql(check_duckdb(sql)).df().head(100)
  except Exception as error:  # noqa: BLE001 - the model reads the error and fixes the query
    return f"ERROR: {error}"
  return frame.to_json(orient="split", date_format="iso")


FILE_TOOLS = [list_files, describe_file, run_duckdb]
