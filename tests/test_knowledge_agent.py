import json
import sqlite3
from pathlib import Path

import pytest

from nimbus.rag.agent import build_tools, structured_data

DATA = Path(__file__).resolve().parents[1] / "data" / "nimbus"


class NoRetriever:
  def search(self, query, k=4):
    return None, []


@pytest.fixture
def sql_tool():
  db = structured_data(DATA)
  yield next(t for t in build_tools(NoRetriever(), db) if t.name == "query_menu_and_orders")
  db.close()


def test_select_returns_rows_as_json(sql_tool):
  result = json.loads(sql_tool.invoke({"sql": "select count(*) from menu where vegan = 'yes'"}))
  assert result == {"columns": ["count(*)"], "rows": [[3]]}


def test_writes_are_refused_before_and_inside_sqlite(sql_tool):
  assert "Only a single SELECT" in sql_tool.invoke({"sql": "delete from menu"})
  # even a write hidden behind a SELECT prefix can't change anything: the connection is query-only
  db = structured_data(DATA)
  with pytest.raises(sqlite3.OperationalError):
    db.execute("insert into menu values ('x', 'hot', 1, 'no')")
  db.close()


def test_sql_errors_become_messages_the_model_can_fix(sql_tool):
  assert "SQL error" in sql_tool.invoke({"sql": "select price from menu"})
