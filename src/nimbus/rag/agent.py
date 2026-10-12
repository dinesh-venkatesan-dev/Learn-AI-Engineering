"""The knowledge agent: agentic RAG over Nimbus's documents AND its structured data.

The Answerer (Module 7) always searches once and answers. Some questions need something else: "how many vegan
items?" needs every row of the menu (SQL, not search), "hi!" needs no search at all, and "is the cheapest vegan
item cheaper than a bag of beans?" needs both sources. The agent gets one tool per source and decides which to call,
how often, and in what order.

    with build_knowledge_agent(settings, retriever, model) as agent:
      agent.invoke({"messages": [("user", question)]})
"""

import csv
import json
import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import ToolException, tool

from nimbus.config import Settings
from nimbus.rag.answer import format_sources

SYSTEM = """You are Nimbus Coffee's knowledge assistant.
- For questions about policies, products, reports, contracts, cafés or notices, use search_documents.
- For counting, averaging, totals or comparisons over the café menu or customer orders, use query_menu_and_orders.
- For greetings and thanks, just reply briefly: no tools.
- Answer only from what the tools return; if they don't contain the answer, say you don't know.
- Cite the source files you used in brackets, like [handbook.md]. Answer in at most three sentences."""

SCHEMA = """Tables (SQLite, read-only):
menu(item TEXT, category TEXT, price_rupees INTEGER, vegan TEXT)  -- category: hot|cold|food; vegan: yes|no
orders(order_id TEXT, customer TEXT, total INTEGER, status TEXT)  -- status: delivered|shipped|processing|cancelled
Sample menu row: ('Oat Latte', 'hot', 220, 'yes'). Prices and totals are in rupees."""


def structured_data(data_dir: Path) -> sqlite3.Connection:
  """The menu and orders as an in-memory SQLite database (re-built at start; read-only from the tool)."""
  db = sqlite3.connect(":memory:", check_same_thread=False)
  db.execute("create table menu (item text, category text, price_rupees integer, vegan text)")
  with (data_dir / "menu.csv").open(newline="") as f:
    db.executemany("insert into menu values (?, ?, ?, ?)", [tuple(r.values()) for r in csv.DictReader(f)])
  db.execute("create table orders (order_id text, customer text, total integer, status text)")
  orders = json.loads((data_dir / "orders.json").read_text())
  db.executemany(
    "insert into orders values (?, ?, ?, ?)", [(o["order_id"], o["customer"], o["total"], o["status"]) for o in orders]
  )
  db.commit()
  db.execute("pragma query_only = on")  # from here on, nothing can be written
  return db


def build_tools(retriever, db: sqlite3.Connection) -> list:
  @tool(response_format="content_and_artifact")
  def search_documents(query: str) -> tuple[str, list[Document]]:
    """Search Nimbus Coffee's documents (policies, handbook, product guide, reports, contracts, café pages, notices,
    emails). Pass a complete, standalone question. Returns numbered sources with their file names."""
    _, hits = retriever.search(query, k=4)
    docs = [d for d, _ in hits]
    return format_sources(docs), docs

  @tool(description=f"Run ONE read-only SQL SELECT over the café menu and customer orders.\n{SCHEMA}")
  def query_menu_and_orders(sql: str) -> str:
    if not sql.lstrip().lower().startswith("select"):
      raise ToolException("Only a single SELECT statement is allowed.")
    try:
      cursor = db.execute(sql)
      rows = cursor.fetchmany(50)  # a row limit
    except sqlite3.Error as error:
      raise ToolException(f"SQL error: {error}. Check the table and column names in the schema.") from error
    columns = [c[0] for c in cursor.description or []]
    return json.dumps({"columns": columns, "rows": rows})

  for t in (search_documents, query_menu_and_orders):
    t.handle_tool_error = True  # an error becomes a message the model can act on
  return [search_documents, query_menu_and_orders]


class KnowledgeAgent:
  """The agent plus the resources its tools hold (the SQLite connection). Use it as a context manager."""

  def __init__(self, settings: Settings, retriever, model: BaseChatModel, checkpointer=None):
    self.db = structured_data(settings.data_dir)
    self.graph = create_agent(
      model, tools=build_tools(retriever, self.db), system_prompt=SYSTEM, checkpointer=checkpointer
    )

  def invoke(self, state: dict, config: dict | None = None) -> dict:
    return self.graph.invoke(state, config)

  def close(self) -> None:
    self.db.close()

  def __enter__(self) -> "KnowledgeAgent":
    return self

  def __exit__(self, *exc) -> None:
    self.close()


def build_knowledge_agent(settings: Settings, retriever, model: BaseChatModel, checkpointer=None) -> KnowledgeAgent:
  return KnowledgeAgent(settings, retriever, model, checkpointer)
