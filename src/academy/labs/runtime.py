"""Labs: small, runnable experiments the course pages call.

A lab is a function decorated with @lab. It declares its inputs (the page renders a form) and returns a list of
views (the page renders each one: tables, ranked results, scorecards, answers...). Labs explore and compare; the
production code they exercise lives in `nimbus`.
"""

import ast
import inspect
import textwrap
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from academy.labs.model_calls import recording

View = dict[str, Any]


@dataclass
class Input:
  name: str
  label: str
  kind: Literal["text", "textarea", "number", "select", "bool"] = "text"
  default: Any = None
  options: list[str] | None = None
  min: float | None = None
  max: float | None = None
  help: str = ""


@dataclass
class Lab:
  id: str
  title: str
  fn: Callable[..., list[View]]
  inputs: list[Input] = field(default_factory=list)
  uses_model: bool = False  # calls the chat model: the page warns that it costs requests
  calls: str = ""  # how many model calls one run makes, e.g. "6 calls to Mistral"

  def describe(self) -> dict:
    return {
      "id": self.id,
      "title": self.title,
      "uses_model": self.uses_model,
      "calls": self.calls,
      "inputs": [asdict(i) for i in self.inputs],
      "module": self.fn.__module__,
      "symbol": self.fn.__name__,
    }

  def run(self, values: dict[str, Any]) -> dict:
    args = {}
    for spec in self.inputs:  # coerce what the browser sent to the declared types
      value = values.get(spec.name, spec.default)
      if spec.kind == "number":
        value = float(value) if isinstance(spec.default, float) else int(value)
        value = min(
          max(value, spec.min if spec.min is not None else value), spec.max if spec.max is not None else value
        )
      elif spec.kind == "bool":
        value = value in (True, "true", "1", 1)
      elif spec.kind == "select" and spec.options and value not in spec.options:
        value = spec.default
      args[spec.name] = value
    started = time.perf_counter()
    with recording() as recorder:  # every chat-model call, for the "What the model saw" tab
      views = self.fn(**args)
    return {
      "lab": self.id,
      "inputs": args,
      "ms": round((time.perf_counter() - started) * 1000),
      "views": [v for v in views if v["type"] != "example"],  # the Run tab: results only
      "examples": [v for v in views if v["type"] == "example"],  # the Covers tab: code per learning point
      "model_calls": recorder.result(),
    }


REGISTRY: dict[str, Lab] = {}


def lab(id: str, title: str, inputs: list[Input] | None = None, uses_model: bool = False, calls: str = ""):  # noqa: A002
  def register(fn: Callable[..., list[View]]) -> Callable[..., list[View]]:
    declared = set(inspect.signature(fn).parameters)
    names = {i.name for i in inputs or []}
    if declared != names:
      raise TypeError(f"lab {id}: parameters {declared} don't match inputs {names}")
    REGISTRY[id] = Lab(id, title, fn, inputs or [], uses_model or bool(calls), calls)
    return fn

  return register


# ── Views: what a lab returns; each kind has a renderer in the UI ──────────────
def md(text: str) -> View:
  return {"type": "markdown", "text": text}


def callout(text: str, tone: Literal["info", "success", "warning", "decision"] = "info", title: str = "") -> View:
  return {"type": "callout", "tone": tone, "title": title, "text": text}


def table(columns: list[str], rows: list[list[Any]], title: str = "") -> View:
  return {"type": "table", "title": title, "columns": columns, "rows": rows}


@dataclass
class Metric:
  key: str
  label: str
  better: Literal["higher", "lower", "none"] = "higher"
  format: Literal["ratio", "percent", "ms", "int", "text", "usd"] = "ratio"
  help: str = ""


def scorecard(rows: list[dict[str, Any]], metrics: list[Metric], title: str = "", label_key: str = "variant") -> View:
  """Variants (rows) × metrics (columns); the UI highlights the best value in each column."""
  return {
    "type": "scorecard",
    "title": title,
    "label_key": label_key,
    "rows": rows,
    "metrics": [asdict(m) for m in metrics],
  }


def results(hits: list[dict[str, Any]], title: str = "", score_label: str = "score") -> View:
  """Ranked search results: each hit has text, source, score and (optionally) relevant: bool."""
  return {"type": "results", "title": title, "score_label": score_label, "hits": hits}


def bars(items: list[dict[str, Any]], title: str = "", unit: str = "", max_value: float | None = None) -> View:
  """Grouped horizontal bars: items = [{label, values: {series: number}}]."""
  return {"type": "bars", "title": title, "unit": unit, "items": items, "max": max_value}


def radar(points: list[dict[str, Any]], rings: list[float], title: str = "") -> View:
  """A map around one centre (the query): points = [{x, y, label, text, rank?}], at their true distance from the
  centre; rings = distances to draw as circles (labelled with the similarity they stand for: 1 - distance)."""
  return {"type": "radar", "title": title, "points": points, "rings": rings}


def highlight(text: str, spans: list[tuple[int, int]], title: str = "") -> View:
  """A passage with each chunk in its own colour: spans = [(start, end)] character positions in `text`, in order.
  Text inside two spans is overlap; text inside none was dropped by the splitter."""
  return {"type": "highlight", "title": title, "text": text, "spans": [list(s) for s in spans]}


def spans_of(text: str, chunks: list[str]) -> list[tuple[int, int]]:
  """Where each chunk sits in the text it was cut from (chunks overlap, so search from just after the last start)."""
  spans, start = [], 0
  for chunk in chunks:
    at = text.find(chunk, start)
    if at < 0:
      continue  # the splitter changed the text (stripped or joined it): skip rather than guess
    spans.append((at, at + len(chunk)))
    start = at + 1
  return spans


def answer(text: str, title: str = "", meta: dict[str, Any] | None = None) -> View:
  return {"type": "answer", "title": title, "text": text, "meta": meta or {}}


def columns(children: list[View], title: str = "") -> View:
  """Views side by side (comparisons)."""
  return {"type": "columns", "title": title, "children": children}


def images(items: list[dict[str, str]], title: str = "") -> View:
  """Pictures the lab sent to a model: items = [{src: data URI or URL, caption}]."""
  return {"type": "images", "title": title, "items": items}


def code(text: str, language: str = "json", title: str = "") -> View:
  """A block of code or data, shown verbatim (a message as JSON, a prompt as the model receives it)."""
  return {"type": "code", "title": title, "language": language, "text": text}


def snippet_source(fn: Callable) -> str:
  """The body of a function as a learner would write it: no def line, no docstring, no final `return`."""
  source = textwrap.dedent(inspect.getsource(fn))
  body = ast.parse(source).body[0].body
  if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
    body = body[1:]  # the docstring
  if len(body) > 1 and isinstance(body[-1], ast.Return):
    body = body[:-1]  # `return x` only hands the result to the lab
  lines = source.splitlines()
  first, last = body[0].lineno - 1, body[-1].end_lineno
  return textwrap.dedent("\n".join(lines[first:last])).strip()


def example(point: str, title: str, fn: Callable, explain: str, outputs: list[View]) -> View:
  """One learning point: why, the exact code that ran (read from `fn`), and what it produced."""
  return {
    "type": "example",
    "point": point,
    "title": title,
    "explain": explain,
    "code": snippet_source(fn),
    "outputs": outputs,
  }
