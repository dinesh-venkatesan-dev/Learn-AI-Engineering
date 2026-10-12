"""The warehouse in PostgreSQL (Data Module 3): the database enforces the rules, not just our Python.

load_and_secure() copies the SQLite warehouse into the nimbus_warehouse database (on the course Postgres) and creates
the agent's role:
  nimbus_agent   SELECT only, on the tables it needs; read-only transactions; statements stopped after 2 s
  columns        customers: every column EXCEPT name (a column grant): the name can't be read, whatever the SQL
  rows           row-level security on orders: a session sees one café's orders when the APP sets app.cafe_id from
                 the signed-in manager (never from anything the model or the user typed)

  load_and_secure()
  columns, rows = as_agent("SELECT COUNT(*) FROM orders", cafe_id=1)
"""

import sqlite3
from contextlib import closing

import psycopg
from psycopg import sql as pgsql

from nimbus.config import get_settings
from nimbus.data.warehouse import DB_PATH, build_warehouse

DATABASE = "nimbus_warehouse"
TABLES = ["cafes", "products", "customers", "orders", "order_items", "reviews"]
AGENT_ROLE, AGENT_PASSWORD = "nimbus_agent", "agent-pass"  # a local course database; in production, from a secret


def _conninfo(dbname: str, user: str | None = None, password: str | None = None) -> str:
  url = psycopg.conninfo.conninfo_to_dict(get_settings().database_url.replace("+psycopg", ""))
  url["dbname"] = dbname
  if user:
    url["user"], url["password"] = user, password
  return psycopg.conninfo.make_conninfo(**url)


def admin_url() -> str:
  return _conninfo(DATABASE)


def agent_url() -> str:
  return _conninfo(DATABASE, AGENT_ROLE, AGENT_PASSWORD)


def create_database() -> None:
  server_db = psycopg.conninfo.conninfo_to_dict(get_settings().database_url.replace("+psycopg", ""))["dbname"]
  with psycopg.connect(_conninfo(server_db), autocommit=True) as pg:  # psycopg's `with` closes the connection
    if not pg.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,)).fetchone():
      pg.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(DATABASE)))


def load_and_secure() -> None:
  """Copy the warehouse in, then create the agent's locked-down role. Safe to run again."""
  build_warehouse()
  create_database()
  with (
    closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as source,  # sqlite3's `with` doesn't close
    psycopg.connect(admin_url(), autocommit=True) as pg,
  ):
    for table in reversed(TABLES):
      pg.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    for table in TABLES:
      ddl = source.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
      pg.execute(ddl)  # these CREATE TABLE statements are valid PostgreSQL too
      rows = source.execute(f"SELECT * FROM {table}").fetchall()
      with pg.cursor() as cursor, cursor.copy(f"COPY {table} FROM STDIN") as copy:
        for row in rows:
          copy.write_row(row)
    if pg.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (AGENT_ROLE,)).fetchone():
      pg.execute(f"DROP OWNED BY {AGENT_ROLE}")
      pg.execute(f"DROP ROLE {AGENT_ROLE}")
    pg.execute(f"CREATE ROLE {AGENT_ROLE} LOGIN PASSWORD '{AGENT_PASSWORD}'")
    pg.execute(f"ALTER ROLE {AGENT_ROLE} SET default_transaction_read_only = on")  # 1. reads only
    pg.execute(f"ALTER ROLE {AGENT_ROLE} SET statement_timeout = '2s'")  # 1. briefly
    pg.execute(f"GRANT CONNECT ON DATABASE {DATABASE} TO {AGENT_ROLE}")
    pg.execute(f"GRANT SELECT ON cafes, products, orders, order_items, reviews TO {AGENT_ROLE}")
    # 2. column privileges: everything about customers EXCEPT their name
    pg.execute(f"GRANT SELECT (customer_id, city, signup_date, loyalty_tier) ON customers TO {AGENT_ROLE}")
    # 3. row-level security: a session sees the orders of the café in app.cafe_id (all cafés when it isn't set)
    pg.execute("ALTER TABLE orders ENABLE ROW LEVEL SECURITY")
    pg.execute(
      f"""CREATE POLICY cafe_scope ON orders FOR SELECT TO {AGENT_ROLE} USING (
            current_setting('app.cafe_id', true) IS NULL OR current_setting('app.cafe_id', true) = ''
            OR cafe_id = current_setting('app.cafe_id', true)::int)"""
    )


def as_agent(sql: str, cafe_id: int | None = None) -> tuple[list[str], list[tuple]]:
  """Run SQL as the agent's role; scoped to one café when cafe_id is given (the APP sets it, from the login)."""
  with psycopg.connect(agent_url()) as connection:
    if cafe_id is not None:
      connection.execute("SELECT set_config('app.cafe_id', %s, false)", (str(cafe_id),))
    cursor = connection.execute(sql)
    return [d.name for d in cursor.description], cursor.fetchall()
