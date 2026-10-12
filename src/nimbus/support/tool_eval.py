"""Tool-use evaluation: did the assistant call the right tools with the right arguments, and answer correctly?

tools_ok     every expected call was made (name + the listed arguments) and no unexpected tool was used
args_ok      of the expected calls, the share made with the right arguments
answer_ok    the answer contains the facts it must (or one of the acceptable phrasings)
success      tools_ok and answer_ok: the request was handled
rounds       model calls per request; tokens and cost per request
"""

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import create_model

from nimbus.support.assistant import ToolRun, answer_with_tools


def load_requests(path: Path) -> list[dict]:
  return json.loads(path.read_text())


def _same(expected, got) -> bool:
  if isinstance(expected, str) and isinstance(got, str):
    return expected.strip().casefold() == got.strip().casefold()
  return expected == got


def call_matches(expected: dict, call: dict) -> bool:
  """Right tool, and one of the acceptable argument sets ("args", or any of "args_any")."""
  options = expected.get("args_any") or [expected.get("args", {})]
  return call["name"] == expected["name"] and any(
    all(_same(v, call["args"].get(k)) for k, v in args.items()) for args in options
  )


def score(request: dict, run: ToolRun) -> dict:
  expected = request["expect"]
  allowed = {e["name"] for e in expected} | set(request.get("also_ok", []))
  made = [e for e in expected if any(call_matches(e, c) for c in run.tool_calls)]
  named = [e for e in expected if any(c["name"] == e["name"] for c in run.tool_calls)]
  stray = [c for c in run.tool_calls if c["name"] not in allowed]
  answer = run.answer.casefold().replace("\u2019", "'").replace("\u2018", "'")  # curly apostrophes count as straight
  numbers = {int(n) for n in re.findall(r"\d+", answer.replace(",", ""))}  # whole numbers: 209 ≠ 2090
  answer_ok = (
    all(f.casefold() in answer for f in request.get("answer_all", []))
    and (not request.get("answer_any") or any(f.casefold() in answer for f in request["answer_any"]))
    and all(n in numbers for n in request.get("answer_numbers", []))
  )
  tools_ok = len(made) == len(expected) and not stray
  return {
    "tools_ok": tools_ok,
    "args_ok": len(made) / len(named) if named else (1.0 if not expected else 0.0),
    "answer_ok": answer_ok,
    "success": tools_ok and answer_ok,
  }


@dataclass
class ToolReport:
  variant: str
  requests: int
  rows: list[tuple[dict, ToolRun, dict]]

  def summary(self) -> dict:
    n = max(len(self.rows), 1)
    usage = [r.usage() for _, r, _ in self.rows]
    ms = sorted(r.ms for _, r, _ in self.rows) or [0.0]
    return {
      "variant": self.variant,
      "requests": self.requests,
      "success": round(sum(s["success"] for *_, s in self.rows) / n, 3),
      "tools_ok": round(sum(s["tools_ok"] for *_, s in self.rows) / n, 3),
      "args_ok": round(sum(s["args_ok"] for *_, s in self.rows) / n, 3),
      "answer_ok": round(sum(s["answer_ok"] for *_, s in self.rows) / n, 3),
      "rounds": round(sum(r.rounds for _, r, _ in self.rows) / n, 2),
      "tokens": round(sum(u.input_tokens + u.output_tokens for u in usage) / n),
      "cost_per_1k_usd": round(sum(u.cost_usd() for u in usage) / n * 1000, 3),
      "latency_p50_ms": round(statistics.median(ms)),
    }


def evaluate_tools(variant: str, model: BaseChatModel, tools: list[BaseTool], requests: list[dict]) -> ToolReport:
  rows = []
  for request in requests:  # one at a time: each request is a multi-call conversation already
    run = answer_with_tools(model, tools, request["request"])
    rows.append((request, run, score(request, run)))
  return ToolReport(variant, len(requests), rows)


def terse(tools: list[BaseTool]) -> list[BaseTool]:
  """The same tools with the descriptions stripped: names and bare argument types only. The comparison baseline."""
  out = []
  for t in tools:
    fields = {name: (f.annotation, ...) for name, f in t.args_schema.model_fields.items()} if t.args_schema else {}
    schema = create_model(f"{t.name}_args", **fields)
    out.append(
      StructuredTool(
        name=t.name,
        description=t.name.replace("_", " "),
        args_schema=schema,
        func=t.func,
        response_format=t.response_format,
        return_direct=t.return_direct,
        handle_tool_error=t.handle_tool_error,
      )
    )
  return out


async def evaluate_tools_async(
  variant: str, model: BaseChatModel, tools: list[BaseTool], requests: list[dict]
) -> ToolReport:
  """evaluate_tools for async-only tools (MCP)."""
  from nimbus.support.assistant import answer_with_tools_async

  rows = []
  for request in requests:
    run = await answer_with_tools_async(model, tools, request["request"])
    rows.append((request, run, score(request, run)))
  return ToolReport(variant, len(requests), rows)
