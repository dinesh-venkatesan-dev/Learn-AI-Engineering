"""Charts (Data Module 3): the model chooses WHAT to plot (a ChartSpec), our code decides HOW.

render() runs the spec's SQL through the guard, checks the shape, and draws it with matplotlib in the house style; no
Python the model writes is ever executed. The same spec could feed a web front end (Vega-Lite, Chart.js): the spec
is the contract. chart_metric draws a governed metric, so the chart and get_metric always agree.
"""

from typing import Literal

from pydantic import BaseModel, Field

from nimbus.config import PROJECT_DIR
from nimbus.data.guard import run_safe
from nimbus.data.metrics import metric_sql
from nimbus.data.sql import matches

CHART_DIR = PROJECT_DIR / ".cache" / "data" / "charts"
COFFEE = "#6f4e37"


class ChartSpec(BaseModel):
  kind: Literal["bar", "line"] = Field(description="line for trends over time, bar for comparing categories")
  title: str
  sql: str = Field(description="One SQLite SELECT returning exactly two columns: x first, then y")
  x_label: str
  y_label: str


def chart_data(spec: ChartSpec) -> tuple[list[str], list[float]]:
  columns, rows = run_safe(spec.sql)
  if len(columns) != 2 or not rows:
    raise ValueError(f"The spec's SQL must return two columns and some rows, got {columns} with {len(rows)} rows.")
  return [str(r[0]) for r in rows], [float(r[1] or 0) for r in rows]


def render(spec: ChartSpec, filename: str) -> str:
  """Draw the spec in the house style and save a PNG; returns its path."""
  import matplotlib

  matplotlib.use("Agg")  # draw to files, no window
  import matplotlib.pyplot as plt

  xs, ys = chart_data(spec)
  figure, axes = plt.subplots(figsize=(8, 4))
  try:
    if spec.kind == "line":
      axes.plot(xs, ys, marker="o", color=COFFEE)
    else:
      axes.bar(xs, ys, color=COFFEE)
    axes.set(title=spec.title, xlabel=spec.x_label, ylabel=spec.y_label)
    axes.spines[["top", "right"]].set_visible(False)
    if len(xs) > 8:
      axes.set_xticks(xs[:: max(1, len(xs) // 8)])
    axes.tick_params(axis="x", labelrotation=30)
    figure.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / filename
    figure.savefig(path, dpi=110)
  finally:
    plt.close(figure)  # figures stay in memory until closed
  return str(path)


def metric_chart_spec(metric: str, dimension: str, title: str) -> ChartSpec:
  """A chart of a governed metric: its SQL comes from the semantic layer, never from the model."""
  over_time = dimension in ("month", "weekday")
  return ChartSpec(
    kind="line" if over_time else "bar",
    title=title,
    x_label=dimension,
    y_label=metric,
    sql=metric_sql(metric, [dimension], sort_by_dimension=over_time),
  )


def check_data(spec: ChartSpec, trusted_sql: str) -> list[str]:
  """The points of a trusted query that the chart's data doesn't contain (empty: the chart is right)."""
  _, chart_rows = run_safe(spec.sql)
  _, trusted_rows = run_safe(trusted_sql)
  return [f"{label}: {value}" for label, value in trusted_rows if not matches(value, chart_rows)]
