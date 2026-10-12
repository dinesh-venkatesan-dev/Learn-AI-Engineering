"""Tokens and cost: every model reply reports its token usage; add it up and price it."""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

# Reference list prices (USD per million tokens) to show what the same traffic costs on a paid model.
# The course runs on free plans; these are for comparison only. Claude Sonnet 5.5, October 2026.
REFERENCE_PRICE_IN, REFERENCE_PRICE_OUT = 2.00, 10.00


@dataclass
class Usage:
  input_tokens: int = 0
  output_tokens: int = 0
  reasoning_tokens: int = 0
  calls: int = 0

  def add(self, message: AIMessage | None) -> "Usage":
    meta = (message.usage_metadata if message is not None else None) or {}
    self.input_tokens += meta.get("input_tokens", 0)
    self.output_tokens += meta.get("output_tokens", 0)
    self.reasoning_tokens += (meta.get("output_token_details") or {}).get("reasoning", 0)
    self.calls += 1
    return self

  def cost_usd(self, price_in: float = REFERENCE_PRICE_IN, price_out: float = REFERENCE_PRICE_OUT) -> float:
    return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000

  def per_call(self) -> dict[str, float]:
    n = max(self.calls, 1)
    return {
      "input_tokens": self.input_tokens / n,
      "output_tokens": self.output_tokens / n,
      "cost_usd": self.cost_usd() / n,
    }


def total(messages: Iterable[AIMessage | None]) -> Usage:
  usage = Usage()
  for m in messages:
    usage.add(m)
  return usage


@dataclass
class ModelCall:
  model: str
  input_tokens: int
  output_tokens: int
  ms: float
  tags: list[str]


class UsageRecorder(BaseCallbackHandler):
  """Records every model call made under it: pass it in `config["callbacks"]` and every step of a chain reports.

  Production uses the same hook to export tokens, cost and latency per request to metrics.
  """

  def __init__(self) -> None:
    self.calls: list[ModelCall] = []
    self._started: dict[UUID, tuple[float, list[str]]] = {}

  def on_chat_model_start(self, serialized, messages, *, run_id: UUID, tags=None, **kwargs) -> None:
    self._started[run_id] = (time.perf_counter(), list(tags or []))

  def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
    started, tags = self._started.pop(run_id, (time.perf_counter(), []))
    message = getattr(response.generations[0][0], "message", None) if response.generations else None
    usage = (getattr(message, "usage_metadata", None) or {}) if message is not None else {}
    meta = getattr(message, "response_metadata", {}) or {}
    self.calls.append(
      ModelCall(
        model=meta.get("model_name") or meta.get("model") or "?",
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        ms=(time.perf_counter() - started) * 1000,
        tags=tags,
      )
    )

  def usage(self) -> Usage:
    return Usage(
      input_tokens=sum(c.input_tokens for c in self.calls),
      output_tokens=sum(c.output_tokens for c in self.calls),
      calls=len(self.calls),
    )
