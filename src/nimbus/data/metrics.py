"""Nimbus's semantic layer (Data Module 3): metrics and dimensions defined ONCE, in code the whole company shares.

The model never writes SQL for these: it picks a metric, dimensions and filters from fixed lists (typed, so invalid
names can't even be sent), and metric_sql() builds the query. "Revenue" then means the same thing in every answer,
chart and dashboard, and rules live in the layer (refund_rate can't be split by product category).
"""

import json
import re
from typing import Literal

from langchain_core.tools import tool

from nimbus.data.guard import run_safe

COMPLETED_REVENUE = "SUM(CASE WHEN o.status = 'completed' THEN oi.quantity * oi.unit_price ELSE 0 END)"
COMPLETED_ORDERS = "COUNT(DISTINCT CASE WHEN o.status = 'completed' THEN o.order_id END)"
METRICS = {  # name → (SQL expression, meaning)
  "revenue": (COMPLETED_REVENUE, "rupees from completed orders"),
  "orders": ("COUNT(DISTINCT o.order_id)", "number of orders, any status"),
  "completed_orders": (COMPLETED_ORDERS, "completed orders"),
  "average_order_value": (
    f"ROUND(1.0 * {COMPLETED_REVENUE} / NULLIF({COMPLETED_ORDERS}, 0), 2)",
    "revenue ÷ completed orders",
  ),
  "refund_rate": (
    "ROUND(100.0 * COUNT(DISTINCT CASE WHEN o.status = 'refunded' THEN o.order_id END) "
    "/ COUNT(DISTINCT o.order_id), 2)",
    "% of orders refunded",
  ),
}
DIMENSIONS = {  # name → SQL expression
  "cafe": "c.name",
  "city": "c.city",
  "channel": "o.channel",
  "month": "strftime('%Y-%m', o.ordered_at)",
  "weekday": "CAST(strftime('%w', o.ordered_at) AS INTEGER)",
  "category": "p.category",
}
PER_ORDER = {"orders", "completed_orders", "refund_rate"}  # counted per order: can't be split by product category
Metric = Literal["revenue", "orders", "completed_orders", "average_order_value", "refund_rate"]
Dimension = Literal["cafe", "city", "channel", "month", "weekday", "category"]
Channel = Literal["in_store", "app", "delivery"]


def metric_sql(
  metric: str,
  group_by: list[str] | None = None,
  month: str | None = None,
  cafe: str | None = None,
  channel: str | None = None,
  sort_by_dimension: bool = False,
) -> str:
  """Build the query for one metric, grouped and filtered. Unknown names raise: they never reach the database."""
  if metric not in METRICS:
    raise ValueError(f"Unknown metric {metric!r}. Metrics: {', '.join(METRICS)}")
  group_by = group_by or []
  for dimension in group_by:
    if dimension not in DIMENSIONS:
      raise ValueError(f"Unknown dimension {dimension!r}. Dimensions: {', '.join(DIMENSIONS)}")
  if "category" in group_by and metric in PER_ORDER:
    raise ValueError(f"{metric} is counted per ORDER, so it can't be split by product category.")
  if month and not re.fullmatch(r"2026-0[789]", month[:7]):
    raise ValueError("month must be 'YYYY-MM' between 2026-07 and 2026-09 (the data's quarter).")
  columns = [f"{DIMENSIONS[d]} AS {d}" for d in group_by] + [f"{METRICS[metric][0]} AS {metric}"]
  where = []
  if month:
    where.append(f"strftime('%Y-%m', o.ordered_at) = '{month[:7]}'")
  if cafe:
    where.append(f"c.name = '{cafe.replace(chr(39), '')}'")
  if channel:
    where.append(f"o.channel = '{channel.replace(chr(39), '')}'")
  sql = (
    "SELECT " + ", ".join(columns) + " FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
    "JOIN cafes c ON c.cafe_id = o.cafe_id JOIN products p ON p.product_id = oi.product_id"
  )
  if where:
    sql += " WHERE " + " AND ".join(where)
  if group_by:
    order = (
      ", ".join(DIMENSIONS[d] for d in group_by) if sort_by_dimension else f"{metric} DESC"
    )  # time series: by date
    sql += " GROUP BY " + ", ".join(DIMENSIONS[d] for d in group_by) + f" ORDER BY {order}"
  return sql


@tool
def get_metric(
  metric: Metric,
  group_by: list[Dimension] | None = None,
  month: str | None = None,
  cafe: str | None = None,
  channel: Channel | None = None,
) -> str:
  """Get a governed Nimbus metric, optionally grouped by dimensions and filtered.
  month: 'YYYY-MM' (2026-07 to 2026-09). cafe: e.g. 'Nimbus Bandra'. channel: in_store, app or delivery."""
  try:
    columns, rows = run_safe(metric_sql(metric, group_by, month, cafe, channel))
  except ValueError as error:
    return f"ERROR: {error}"
  return json.dumps({"columns": columns, "rows": rows})


def metrics_prompt() -> str:
  return (
    "Metrics: "
    + "; ".join(f"{name} = {meaning}" for name, (_, meaning) in METRICS.items())
    + f". Dimensions: {', '.join(DIMENSIONS)}."
  )
