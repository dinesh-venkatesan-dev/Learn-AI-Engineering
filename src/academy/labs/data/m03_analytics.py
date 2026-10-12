"""Data agents · Module 3 labs: analytics assistants. pandas for statistics, charts from specs, a semantic layer of
governed metrics, files with DuckDB, and the rules enforced by PostgreSQL. Then the capstone: the analytics
assistant, measured against a SQL-only assistant."""

import json
import time
from collections import Counter

from langchain_core.messages import HumanMessage

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.llm import get_chat_model


def _tool_lines(messages) -> list[str]:
  out = []
  for m in messages:
    for c in getattr(m, "tool_calls", None) or []:
      detail = c["args"].get("sql") or c["args"].get("code") or c["args"].get("table") or c["args"].get("source") or ""
      out.append(f"{c['name']}: {' '.join(str(detail).split())[:220]}")
  return out


# ── Lab 1: analysis with pandas ────────────────────────────────────────────────
WEEKDAY_QUESTION = (
  "Is there a weekday pattern in revenue? Which weekday is strongest, and by what percentage is it above the average "
  "day?"
)
REVIEWS_QUESTION = (
  "Do orders with more items get better reviews? Give the correlation between items per order and rating."
)


def analyst(model, session):
  from langchain.agents import create_agent

  from nimbus.data.analysis import ANALYST
  from nimbus.data.tools import describe_table

  # load_dataframe runs SQL through the guard; run_pandas runs the model's code in a separate process
  return create_agent(model, [describe_table, *session.tools()], system_prompt=ANALYST)


def strongest_weekday() -> str:
  from nimbus.data.guard import run_safe

  _, rows = run_safe(
    "SELECT CAST(strftime('%w', o.ordered_at) AS INTEGER) AS day, SUM(oi.quantity * oi.unit_price) FROM orders o "
    "JOIN order_items oi USING (order_id) WHERE o.status = 'completed' GROUP BY day"
  )
  revenue = dict(rows)  # SQLite numbers weekdays from 0 = Sunday (pandas from 0 = Monday)
  best = max(revenue, key=revenue.get)
  names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
  return f"{names[best]}, {revenue[best] / (sum(revenue.values()) / len(revenue)) - 1:.1%} above the average weekday"


def subprocess_isolation(session) -> dict:
  return {
    "API key visible?": session.run("import os; print('MISTRAL_API_KEY' in os.environ)"),
    "an endless loop": session.run("while True: pass"),
  }


@lab("data.pandas", "Analysis with pandas: SQL fetches, Python computes", uses_model=True, calls="~12 calls to Mistral")
def pandas_lab():
  """A weekday pattern and a correlation, worked out by an agent that loads DataFrames with SQL and runs pandas in a
  separate process; the weekday answer checked in plain Python; what the subprocess can't do."""
  from nimbus.data.analysis import AnalysisSession

  model = get_chat_model()
  rows, answers = [], {}
  with AnalysisSession() as session:
    agent = analyst(model, session)
    for question in (WEEKDAY_QUESTION, REVIEWS_QUESTION):
      result = agent.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 30})
      answers[question] = result["messages"][-1].text
      rows.append([question, "\n".join(_tool_lines(result["messages"])), answers[question]])
    isolation = subprocess_isolation(session)
  check = strongest_weekday()
  work = table(["question", "tools", "answer"], rows)
  return [
    work,
    table(["check in plain Python", "value"], [["strongest weekday", check]]),
    table(["what model code can do", "result"], [[k, v] for k, v in isolation.items()]),
    callout(
      f"The weekday answer, checked in plain Python: {check}. Compare it with the agent's answer above: a mismatch "
      "is usually SQLite's weekday numbering (0 = Sunday) read with pandas' (0 = Monday). The code ran without the "
      "API key and was stopped after 20 s; that limits accidents, not attacks.",
      "info",
      "Check what you can",
    ),
    example("data.l10.1", "load_dataframe", AnalysisSession.load, "", [work]),
    example(
      "data.l10.2",
      "run_pandas in a separate process",
      AnalysisSession.run,
      "",
      [table(["", ""], [[k, v] for k, v in isolation.items()])],
    ),
    example("data.l10.3", "Analyses that need code", analyst, "", [work]),
    example(
      "data.l10.4",
      "Check what you can",
      strongest_weekday,
      "",
      [code(check, "text"), answer(answers[WEEKDAY_QUESTION])],
    ),
    example(
      "data.l10.5",
      "The risk, said plainly",
      subprocess_isolation,
      "",
      [table(["", ""], [[k, v] for k, v in isolation.items()])],
    ),
  ]


# ── Lab 2: charts ──────────────────────────────────────────────────────────────
CHART_REQUESTS = [
  "Show weekly revenue over the quarter.",
  "Compare revenue by café for September 2026.",
  "How does the refund rate differ by channel?",
]
TRUSTED_REFUNDS = "SELECT channel, 100.0 * SUM(status = 'refunded') / COUNT(*) FROM orders GROUP BY channel"
AS_FRACTIONS = "SELECT channel, 1.0 * SUM(status = 'refunded') / COUNT(*) FROM orders GROUP BY channel"


def design_chart(model, request):
  from langchain_core.messages import SystemMessage

  from nimbus.data.charts import ChartSpec
  from nimbus.data.context import GLOSSARY
  from nimbus.data.sql import schema_text

  # The model returns a ChartSpec (what to plot); nothing it writes is run as Python
  designer = model.with_structured_output(ChartSpec, method="json_schema")
  system = (
    f"You design charts for Nimbus Coffee's data.\n\nSchema:\n{schema_text()}\n\n{GLOSSARY}\nReturn a chart spec "
    "whose SQL returns exactly two columns (x, then y), sorted the way the chart should read."
  )
  return designer.invoke([SystemMessage(system), HumanMessage(request)])


def first_and_last_week(spec) -> str:
  from nimbus.data.charts import chart_data

  xs, ys = chart_data(spec)
  middle = sum(ys[1:-1]) / max(len(ys) - 2, 1)
  return f"first week {ys[0]:,.0f}, last week {ys[-1]:,.0f}, a full week on average {middle:,.0f}"


@lab(
  "data.charts", "Charts: the model chooses WHAT, your code decides HOW", uses_model=True, calls="3 calls to Mistral"
)
def charts():
  """Three requests → three ChartSpecs → PNGs drawn by our renderer; the refund chart's data checked against a trusted
  query; the partial first and last weeks."""
  from nimbus.data.charts import check_data, render

  model = get_chat_model()
  rows, specs, paths = [], [], []
  for n, request in enumerate(CHART_REQUESTS, 1):
    spec = design_chart(model, request)
    try:
      path = render(spec, f"lab_chart_{n}.png")
    except ValueError as error:
      path = f"not drawn: {error}"
    specs.append(spec)
    paths.append(path)
    rows.append([request, spec.kind, spec.title, spec.sql, path])
  wrong = check_data(specs[2], TRUSTED_REFUNDS)  # unrounded, so rounding never reads as an error
  fractions = bool(wrong) and not check_data(specs[2], AS_FRACTIONS)  # right numbers, as 0.0951 instead of 9.51 %
  verdict = "✓ matches" if not wrong else ("✓ right, but as fractions: label the axis" if fractions else f"✗ {wrong}")
  edges = first_and_last_week(specs[0]) if not paths[0].startswith("not drawn") else paths[0]
  view = table(["request", "kind", "title", "SQL", "PNG"], rows)
  spec_json = code(json.dumps(specs[1].model_dump(), indent=1), "json")
  return [
    view,
    table(
      ["check", "result"],
      [
        ["refund-rate chart vs trusted numbers", verdict],
        ["weekly revenue: partial weeks", edges],
      ],
    ),
    callout(
      (
        "The refund chart's data matches the trusted query. "
        if not wrong
        else "The refund chart's numbers are right but fractions (0.0951) under a title that says rate: a reader "
        "takes 0.1 for 0.1 %. "
        if fractions
        else f"The refund chart's SQL computed the wrong numbers ({len(wrong)} of 3 points): a correct-looking chart. "
      )
      + "The first and last weeks of the quarter are partial, so the line drops at the edges without anything "
      "happening: drop or label partial periods. Governed metrics (the next lab) take the SQL out of the model's "
      "hands.",
      "decision",
      "Specs from the model, drawing and checks in code",
    ),
    example("data.l11.1", "A ChartSpec", design_chart, "", [spec_json]),
    example("data.l11.2", "Your renderer", render, "src/nimbus/data/charts.py", [view]),
    example("data.l11.3", "Different questions, different charts", design_chart, "", [view]),
    example(
      "data.l11.4",
      "The spec is the contract",
      design_chart,
      "The same JSON could feed Vega-Lite or Chart.js.",
      [spec_json],
    ),
    example(
      "data.l11.5",
      "A chart is only as right as its SQL",
      check_data,
      "",
      [code(verdict, "text"), code(edges, "text")],
    ),
  ]


# ── Lab 3: a semantic layer ────────────────────────────────────────────────────
PHRASINGS = [
  "What's the refund rate for delivery orders?",
  "How often do delivery orders get refunded, in %?",
  "delivery refunds: what share of orders?",
]


def metrics_agent(model):
  from langchain.agents import create_agent

  from nimbus.data.metrics import get_metric, metrics_prompt

  # The model picks a metric, dimensions and filters from fixed lists: it never writes SQL for these
  return create_agent(
    model,
    [get_metric],
    system_prompt=f"You answer questions about Nimbus Coffee using ONLY get_metric. {metrics_prompt()} Quote numbers "
    "exactly as the tool returns them. One or two sentences.",
  )


def sql_agent(model):
  from nimbus.data.agent import build_sql_agent

  return build_sql_agent(model)


def consistency(agent, runs: int = 3) -> tuple[Counter, int]:
  from nimbus.data.guard import run_safe
  from nimbus.data.metrics import metric_sql
  from nimbus.data.sql import matches

  rate = run_safe(metric_sql("refund_rate", channel="delivery"))[1][0][0]  # the governed number: 9.51
  results, from_a_tool = Counter(), 0
  for phrasing in PHRASINGS:
    for _ in range(runs):
      result = agent.invoke({"messages": [HumanMessage(phrasing)]}, {"recursion_limit": 30})
      outputs = [m.text for m in result["messages"] if m.type == "tool" and not m.text.startswith("ERROR")]
      results[outputs[-1][:120] if outputs else "no tool result"] += 1
      rows = [tuple(r) for o in outputs if o.startswith("{") for r in json.loads(o).get("rows", [])]
      from_a_tool += matches(rate, rows)  # the rate itself came out of a tool, not the model's arithmetic
  return results, from_a_tool


@lab(
  "data.metrics",
  "A semantic layer: governed metrics, the same numbers every time",
  uses_model=True,
  calls="~40 calls to Mistral",
)
def metrics():
  """The layer's metrics and the SQL it builds; one question in three phrasings, three times each, through the
  metrics agent and through Module 2's SQL agent; a rule the layer enforces."""
  from nimbus.data.metrics import DIMENSIONS, METRICS, get_metric, metric_sql

  model = get_chat_model()
  governed, governed_ok = consistency(metrics_agent(model))
  free, free_ok = consistency(sql_agent(model), runs=1)
  rule = get_metric.invoke({"metric": "refund_rate", "group_by": ["category"]})
  layer = table(
    ["metric", "means"], [[k, v[1]] for k, v in METRICS.items()], title=f"Dimensions: {', '.join(DIMENSIONS)}"
  )
  built = code(metric_sql("refund_rate", ["channel"]), "sql", "refund_rate by channel")
  runs = table(
    ["system", "runs where a tool returned the rate", "last tool result (count)"],
    [
      ["get_metric", f"{governed_ok}/9", "\n".join(f"{v}× {k}" for k, v in governed.items())],
      ["SQL agent", f"{free_ok}/3", "\n".join(f"{v}× {k}" for k, v in free.items())],
    ],
  )
  return [
    layer,
    built,
    runs,
    table(["refund_rate by category", "the layer says"], [["get_metric", rule]]),
    callout(
      f"Through get_metric the refund rate came out of the tool in {governed_ok} of 9 runs; through free SQL in "
      f"{free_ok} of 3 (when it didn't, the model worked the rate out from counts itself). The layer builds the query "
      "from typed choices, so phrasing can't change the definition and the number is never the model's arithmetic. "
      "Rules live there too: "
      "refund_rate is counted per order, so splitting it by product category is refused instead of misleading.",
      "decision",
      "Standard numbers come from the layer",
    ),
    example(
      "data.l12.1", "METRICS, DIMENSIONS, metric_sql()", metric_sql, "src/nimbus/data/metrics.py", [layer, built]
    ),
    example("data.l12.2", "A typed get_metric tool", metrics_agent, "", [runs]),
    example("data.l12.3", "Consistency, measured", consistency, "", [runs]),
    example("data.l12.4", "Rules live in the layer", metric_sql, "", [code(rule, "text")]),
  ]


# ── Lab 4: files with DuckDB ───────────────────────────────────────────────────
FILE_QUESTIONS = {
  "Which supplier delivered late most often, as a share of its deliveries?": (
    "SELECT supplier FROM 'deliveries.csv' GROUP BY supplier ORDER BY AVG((delivered_on > promised_on)::INT) DESC "
    "LIMIT 1"
  ),
  "Which café earned the most revenue per staff hour in September 2026?": (
    "SELECT s.cafe FROM staff_shifts s JOIN 'sales_export.parquet' e ON e.cafe = s.cafe AND e.month = '2026-09' "
    "GROUP BY s.cafe ORDER BY MAX(e.revenue) / SUM(s.hours) DESC LIMIT 1"
  ),
}


def files_in_sql():
  from nimbus.data.files import check_duckdb, connect

  with connect() as con:  # a fresh in-process DuckDB; CSV and Parquet are read straight from the files
    late = con.sql(
      check_duckdb(
        "SELECT supplier, COUNT(*) AS deliveries, ROUND(AVG((delivered_on > promised_on)::INT), 3) AS late_share "
        "FROM 'deliveries.csv' GROUP BY supplier ORDER BY supplier"
      )
    ).fetchall()
    joined = con.sql(  # staff_shifts came from Excel through pandas; the same SQL joins it with the Parquet export
      check_duckdb(
        "SELECT s.cafe, SUM(s.hours) AS staff_hours, MAX(e.revenue) AS revenue FROM staff_shifts s JOIN "
        "'sales_export.parquet' e ON e.cafe = s.cafe AND e.month = '2026-09' GROUP BY s.cafe ORDER BY revenue DESC"
      )
    ).fetchall()
  return late, joined


def files_guard() -> dict:
  from nimbus.data.files import check_duckdb, connect

  cases = {
    "a read": "SELECT COUNT(*) FROM 'deliveries.csv'",
    "COPY out": "COPY (SELECT 1) TO '/tmp/out.csv'",
    "ATTACH": "ATTACH '/tmp/other.db'",
    "a path outside": "SELECT * FROM '/etc/passwd'",
    "read_csv on any path": "SELECT * FROM read_csv('/etc/hosts')",
  }
  out = {}
  for label, sql in cases.items():
    try:
      out[label] = f"runs: {check_duckdb(sql)}"
    except ValueError as refusal:
      out[label] = f"refused: {refusal}"
  with connect() as con:  # skip the guard entirely: the DATABASE only reads the data folder
    try:
      con.sql("SELECT * FROM read_csv('/etc/hosts')").fetchall()
      out["no guard at all"] = "runs"
    except Exception as error:  # noqa: BLE001 - DuckDB's PermissionException
      out["no guard at all"] = f"refused by DuckDB: {str(error).splitlines()[0][:110]}"
  return out


def files_agent(model):
  from langchain.agents import create_agent

  from nimbus.data.files import FILE_TOOLS

  return create_agent(
    model,
    FILE_TOOLS,
    system_prompt="You are Nimbus Coffee's analyst for data in files. Look at the files and their columns before "
    "querying. Every number in your answer must come from a query; money is in rupees. At most three sentences.",
  )


@lab("data.files", "Data in files: CSV, Excel and Parquet with DuckDB", uses_model=True, calls="~12 calls to Mistral")
def files():
  """Supplier deliveries (CSV), the staff rota (Excel) and the finance export (Parquet) in SQL; the files guard; an
  agent answering across sources."""
  from nimbus.data.files import check_duckdb, connect

  late, joined = files_in_sql()
  guard = files_guard()
  agent = files_agent(get_chat_model())
  rows = []
  with connect() as con:
    for question, trusted in FILE_QUESTIONS.items():
      result = agent.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 30})
      right = con.sql(check_duckdb(trusted)).fetchone()[0]
      text = result["messages"][-1].text
      rows.append(
        [
          question,
          "\n".join(_tool_lines(result["messages"])),
          text,
          right,
          "✓" if right.lower() in text.lower() else "✗",
        ]
      )
  sql_views = [
    table(["supplier", "deliveries", "late share"], [list(r) for r in late], title="deliveries.csv"),
    table(["café", "staff hours", "revenue"], [list(r) for r in joined], title="Excel rota ⨝ Parquet export"),
  ]
  agent_view = table(["question", "tools", "answer", "right", "✓"], rows)
  guard_view = table(["case", "result"], [[k, v] for k, v in guard.items()])
  return [
    *sql_views,
    guard_view,
    agent_view,
    callout(
      f"The agent got {sum(r[-1] == '✓' for r in rows)}/{len(rows)} questions across files right. DuckDB runs SQL on "
      "the files where they are; the guard keeps it to SELECTs on files named in FROM, and the connection itself "
      "can't read outside the data folder (allowed_directories, locked), even with the guard skipped.",
      "decision",
      "Files through DuckDB, guarded",
    ),
    example("data.l13.1", "DuckDB reads files in SQL", files_in_sql, "", [sql_views[0]]),
    example(
      "data.l13.2",
      "Excel through pandas",
      files_in_sql,
      "nimbus.data.files.connect registers the sheet.",
      [sql_views[1]],
    ),
    example("data.l13.3", "A files agent, guarded", files_guard, "", [guard_view]),
    example("data.l13.4", "Questions across sources", files_agent, "", [agent_view]),
  ]


# ── Lab 5: the production database ─────────────────────────────────────────────
REVENUE_BY_CAFE = (
  "SELECT c.name, SUM(oi.quantity * oi.unit_price) AS revenue FROM orders o JOIN order_items oi ON "
  "oi.order_id = o.order_id JOIN cafes c ON c.cafe_id = o.cafe_id WHERE o.status = 'completed' GROUP BY c.name "
  "ORDER BY revenue DESC"
)


def what_the_role_can_do() -> list[list[str]]:
  import psycopg

  from nimbus.data.postgres import as_agent

  rows = []
  for label, sql in (
    ("read orders", "SELECT COUNT(*) FROM orders"),
    ("write", "DELETE FROM orders"),
    ("customer names", "SELECT name FROM customers LIMIT 3"),
    ("other customer columns", "SELECT loyalty_tier, COUNT(*) FROM customers GROUP BY 1"),
    ("a 5-second query", "SELECT pg_sleep(5)"),
  ):
    try:
      rows.append([label, "✓ " + str(as_agent(sql)[1][:3])])
    except psycopg.Error as error:  # the DATABASE refuses, whatever SQL the model wrote
      rows.append([label, "refused: " + str(error).splitlines()[0]])
  return rows


def two_managers() -> list[list[str]]:
  from nimbus.data.postgres import as_agent

  # The APP sets app.cafe_id from the signed-in manager; the same SQL returns only their café's orders
  return [
    [manager, str(as_agent(REVENUE_BY_CAFE, cafe_id)[1])]
    for manager, cafe_id in (("Priya (manages Bandra, café 1)", 1), ("Arjun (manages Indiranagar, café 3)", 3))
  ]


@lab("data.postgres", "The production database: PostgreSQL enforces the rules", calls="No model calls")
def postgres():
  """The warehouse loaded into the course Postgres with a locked-down role: what it can and can't do, and row-level
  security for two café managers. Needs the course database: docker compose up -d."""
  from nimbus.data.postgres import load_and_secure

  load_and_secure()
  can = table(["as nimbus_agent", "result"], what_the_role_can_do())
  managers = table(["manager", "the same revenue query"], two_managers())
  refused = sum(r[1].startswith("refused") for r in can.get("rows", []))
  return [
    can,
    managers,
    callout(
      f"{refused} of 5 attempts were refused by the database itself: a write in a read-only transaction, a column "
      "without a grant, a statement past its timeout. Row-level security gave each manager only their café's "
      "revenue from the same SQL. These rules hold whatever the model writes, and whatever the Python guard misses.",
      "decision",
      "The agent connects as nimbus_agent",
    ),
    example("data.l14.1", "A dedicated, read-only role", load_and_secure, "", [can]),
    example("data.l14.2", "Column privileges", what_the_role_can_do, "", [can]),
    example("data.l14.3", "Row-level security", two_managers, "", [managers]),
    example("data.l14.4", "Two managers, two answers", two_managers, "", [managers]),
  ]


# ── Production path: the analytics assistant ───────────────────────────────────
CAPSTONE_QUESTIONS = [
  {
    "question": "What was our revenue in each month of the quarter?",
    "sql": (
      "SELECT SUM(oi.quantity * oi.unit_price) FROM orders o JOIN order_items oi USING (order_id) WHERE o.status = "
      "'completed' AND strftime('%Y-%m', o.ordered_at) = '2026-09'"
    ),
  },
  {
    "question": "Which channel has the worst refund rate, and what is it?",
    "sql": (
      "SELECT channel, ROUND(100.0 * SUM(status = 'refunded') / COUNT(*), 2) FROM orders GROUP BY channel ORDER BY "
      "2 DESC LIMIT 1"
    ),
  },
  {
    "question": "Which café has the lowest average order value? Give its AOV.",
    "sql": (
      "SELECT c.name, ROUND(1.0 * SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.order_id), 2) FROM orders o "
      "JOIN order_items oi USING (order_id) JOIN cafes c USING (cafe_id) WHERE o.status = 'completed' GROUP BY "
      "c.name ORDER BY 2 LIMIT 1"
    ),
  },
  {
    "question": "How much revenue did delivery orders bring in September 2026?",
    "sql": (
      "SELECT SUM(oi.quantity * oi.unit_price) FROM orders o JOIN order_items oi USING (order_id) WHERE o.status = "
      "'completed' AND o.channel = 'delivery' AND strftime('%Y-%m', o.ordered_at) = '2026-09'"
    ),
  },
  {
    "question": "What's the average review rating for delivery orders compared with in-store orders?",
    "sql": (
      "SELECT ROUND(AVG(CASE WHEN o.channel = 'delivery' THEN r.rating END), 2), ROUND(AVG(CASE WHEN o.channel = "
      "'in_store' THEN r.rating END), 2) FROM reviews r JOIN orders o USING (order_id)"
    ),
  },
  {
    "question": "Chart revenue by city and tell me which city leads.",
    "sql": (
      "SELECT c.city, SUM(oi.quantity * oi.unit_price) FROM orders o JOIN order_items oi USING (order_id) JOIN "
      "cafes c USING (cafe_id) WHERE o.status = 'completed' GROUP BY c.city ORDER BY 2 DESC LIMIT 1"
    ),
  },
]


def right_in(reply, trusted_sql: str) -> bool:
  from nimbus.data.assistant import numbers_in
  from nimbus.data.guard import run_safe

  # every value of the trusted row must be in the answer: numbers within 0.5% (₹6,02,380 = 602380), names as text
  _, rows = run_safe(trusted_sql)
  numbers, text = numbers_in(reply.answer), reply.answer.lower()
  for value in rows[0]:
    if isinstance(value, int | float):
      if not any(abs(n - value) <= max(0.01, abs(value) * 0.005) for n in numbers):
        return False
    elif str(value).lower() not in text:
      return False
  return True


def evaluate_assistants() -> dict:
  from nimbus.data.assistant import AnalyticsAssistant
  from nimbus.llm.usage import UsageRecorder

  model = get_chat_model()
  out = {}
  for name, metrics_first in (
    ("SQL tools only (Module 2 agent + charts)", False),
    ("Analytics assistant: metrics first", True),
  ):
    assistant = AnalyticsAssistant(model, metrics_first=metrics_first)
    runs = []
    for q in CAPSTONE_QUESTIONS:
      recorder = UsageRecorder()
      started = time.perf_counter()
      reply = assistant.ask(q["question"], {"callbacks": [recorder]})
      runs.append((q, reply, recorder.usage(), time.perf_counter() - started))
    out[name] = runs
  return out


@lab(
  "data.assistant-eval",
  "Evaluate: the analytics assistant",
  uses_model=True,
  calls="~60 calls to Mistral: 6 questions × 2 assistants",
)
def assistant_eval():
  """Six analytics questions (monthly revenue, the worst refund rate, the weakest AOV, a filtered revenue, ratings by
  channel, a chart) through an assistant with SQL tools only and through the metrics-first analytics assistant:
  right numbers, numbers not found in any tool output, charts, model calls."""
  from nimbus.data.assistant import grounding_check

  started = time.perf_counter()
  results = evaluate_assistants()
  seconds = time.perf_counter() - started
  rows, detail = [], []
  for name, runs in results.items():
    n = len(runs)
    rows.append(
      {
        "variant": name,
        "correct": round(sum(right_in(r, q["sql"]) for q, r, _, _ in runs) / n, 3),
        "ungrounded": sum(len(r.ungrounded) for _, r, _, _ in runs),
        "charts": sum(len(r.charts) for _, r, _, _ in runs),
        "calls": round(sum(u.calls for _, _, u, _ in runs) / n, 2),
        "tokens": round(sum(u.input_tokens + u.output_tokens for _, _, u, _ in runs) / n),
        "latency_p50_ms": round(sorted(s for *_, s in runs)[n // 2] * 1000),
      }
    )
    for q, r, _, _ in runs:
      detail.append(
        [
          name,
          q["question"],
          ", ".join(r.tools),
          r.answer,
          "✓" if right_in(r, q["sql"]) else "✗",
          ", ".join(map(str, r.ungrounded)) or "—",
        ]
      )
  sql_only, assistant = rows
  return [
    scorecard(
      rows,
      [
        Metric("correct", "right numbers"),
        Metric("ungrounded", "numbers not in any tool output", "lower", "int"),
        Metric("charts", "charts drawn", "none", "int"),
        Metric("calls", "model calls per question", "lower", "text"),
        Metric("tokens", "tokens per question", "lower", "int"),
        Metric("latency_p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"{len(CAPSTONE_QUESTIONS)} questions · {seconds:.0f} s",
    ),
    table(["assistant", "question", "tools", "answer", "✓", "ungrounded numbers"], detail),
    callout(
      f"Right numbers: {assistant['correct']:.0%} metrics-first against {sql_only['correct']:.0%} with SQL only; "
      f"numbers not found in any tool output: {assistant['ungrounded']} against {sql_only['ungrounded']}; "
      f"{assistant['calls']} model calls per question against {sql_only['calls']}. The grounding check flags "
      "invented NUMBERS; claims in words still need a reader or a judge.",
      "decision",
      "Ship the analytics assistant" if assistant["correct"] >= sql_only["correct"] else "Keep SQL-only for now",
    ),
    example(
      "data.l15.0",
      "The capstone",
      grounding_check,
      "",
      [table(["assistant", "question", "✓"], [[d[0], d[1], d[4]] for d in detail])],
    ),
  ]
