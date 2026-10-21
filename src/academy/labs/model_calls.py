"""Record every chat-model call a lab makes, so the page can show exactly what the model saw and said.

Registered as a LangChain configure hook: while a lab runs, every model call in that context (direct, in chains,
in agents, in graphs) reports to the recorder, without the lab's code knowing. Cached lab calls never reach a model,
so they don't appear.
"""

import time
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

MAX_CALLS = 40  # a lab with 100+ calls keeps the first 40 and says how many there were
MAX_CHARS = 20_000  # per message: a stuffed prompt is shown whole, a runaway one is cut

_recorder: ContextVar["ModelCallRecorder | None"] = ContextVar("lab_model_calls", default=None)
register_configure_hook(_recorder, inheritable=True)


def _text(content: Any) -> str:
  if isinstance(content, str):
    return content
  parts = []
  for part in content or []:  # multimodal content: text parts kept, images and files named
    if isinstance(part, dict) and part.get("type") == "text":
      parts.append(part.get("text", ""))
    elif isinstance(part, dict):
      parts.append(f"[{part.get('type', 'part')}]")
    else:
      parts.append(str(part))
  return "\n".join(parts)


def _cut(text: str) -> str:
  return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f"\n… ({len(text) - MAX_CHARS:,} more characters)"


def _message(m: BaseMessage) -> dict:
  out = {"role": m.type, "text": _cut(_text(m.content))}
  if getattr(m, "tool_calls", None):
    out["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
  if m.type == "tool":
    out["name"] = getattr(m, "name", None) or ""
  return out


class ModelCallRecorder(BaseCallbackHandler):
  def __init__(self) -> None:
    self.calls: dict[UUID, dict] = {}
    self.total = 0

  def on_chat_model_start(
    self, serialized: dict, messages: list[list[BaseMessage]], *, run_id: UUID, metadata: dict | None = None, **kw
  ) -> None:
    self.total += 1
    if len(self.calls) >= MAX_CALLS:
      return
    model = (metadata or {}).get("ls_model_name") or (kw.get("invocation_params") or {}).get("model", "")
    self.calls[run_id] = {
      "model": model,
      "messages": [_message(m) for m in messages[0]],
      "started": time.perf_counter(),
    }

  def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kw) -> None:
    call = self.calls.get(run_id)
    if call is None:
      return
    generation = response.generations[0][0]
    message = getattr(generation, "message", None)
    usage = getattr(message, "usage_metadata", None) or {}
    call["reply"] = _message(message) if message is not None else {"role": "ai", "text": generation.text}
    call["tokens_in"], call["tokens_out"] = usage.get("input_tokens"), usage.get("output_tokens")
    call["seconds"] = round(time.perf_counter() - call.pop("started"), 2)

  def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw) -> None:
    call = self.calls.get(run_id)
    if call is not None:
      call["error"] = f"{type(error).__name__}: {str(error)[:300]}"
      call["seconds"] = round(time.perf_counter() - call.pop("started", time.perf_counter()), 2)

  def result(self) -> dict:
    calls = [{k: v for k, v in c.items() if k != "started"} for c in self.calls.values()]
    return {"calls": calls, "total": self.total}


class recording:  # noqa: N801 — used as `with recording() as recorder:`
  """Record the chat-model calls made inside the block."""

  def __enter__(self) -> ModelCallRecorder:
    self.recorder = ModelCallRecorder()
    self.token = _recorder.set(self.recorder)
    return self.recorder

  def __exit__(self, *exc) -> None:
    _recorder.reset(self.token)
