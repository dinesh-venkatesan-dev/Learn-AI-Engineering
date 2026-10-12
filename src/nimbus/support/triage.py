"""Support triage: a customer message in, a validated TicketTriage out, with usage, latency and failures recorded.

One call per ticket. Many tickets run concurrently with `batch` (the provider's rate limiter still applies), and the
async version serves a web request without blocking the server. A reply draft streams token by token.
"""

import asyncio
import base64
import mimetypes
import time
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from nimbus.llm import classify, supports
from nimbus.support.prompts import NEXT_STEPS, REPLY_PROMPT, TRIAGE_PROMPTS
from nimbus.support.schema import Ticket, TicketTriage


@dataclass
class TriageResult:
  ticket_id: str
  triage: TicketTriage | None  # None when the call failed or the reply didn't match the schema
  raw: AIMessage | None  # the model's actual reply: usage, finish reason, what failed to parse
  latency_ms: float
  error: str | None = None
  action: str | None = None  # for errors: "retry" (temporary) or "fix" (the request is wrong)


def image_block(path) -> dict:
  """A standard LangChain image content block; the provider package translates it to its own format."""
  mime = mimetypes.guess_type(path)[0] or "image/png"
  return {"type": "image", "base64": base64.b64encode(path.read_bytes()).decode(), "mime_type": mime}


class Triager:
  def __init__(self, model: BaseChatModel, prompt_version: str = "v2", method: str = "json_schema"):
    if prompt_version not in TRIAGE_PROMPTS:
      raise ValueError(f"unknown prompt version {prompt_version!r}; have {sorted(TRIAGE_PROMPTS)}")
    self.model = model
    self.prompt_version = prompt_version
    self.method = method  # function_calling: the schema goes as a tool · json_schema: as the response format
    self.prompt = TRIAGE_PROMPTS[prompt_version]
    # include_raw: keep the AIMessage next to the parsed object, so usage and parse failures are visible
    self.structured = model.with_structured_output(TicketTriage, method=method, include_raw=True)

  def messages(self, ticket: Ticket) -> list[BaseMessage]:
    messages = self.prompt.invoke({"ticket": ticket.text}).to_messages()
    if ticket.attachment and supports(self.model, "image_inputs"):
      messages[-1] = HumanMessage(content=[{"type": "text", "text": ticket.text}, image_block(ticket.attachment)])
    return messages

  def _result(self, ticket: Ticket, out: dict | BaseException, started: float) -> TriageResult:
    latency = (time.perf_counter() - started) * 1000
    if isinstance(out, BaseException):
      failure = classify(out)
      return TriageResult(ticket.id, None, None, latency, failure.reason, failure.action)
    if out["parsing_error"] is not None or out["parsed"] is None:
      error = out["parsing_error"]
      reason = (
        f"reply didn't match the schema: {error.errors()[0]['msg']}"
        if isinstance(error, ValidationError)
        else (f"reply didn't match the schema: {error}" if error else "no structured reply")
      )
      return TriageResult(ticket.id, None, out["raw"], latency, reason[:300], "fix")
    return TriageResult(ticket.id, out["parsed"], out["raw"], latency)

  def triage(self, ticket: Ticket) -> TriageResult:
    started = time.perf_counter()
    try:
      out = self.structured.invoke(self.messages(ticket))
    except Exception as error:  # noqa: BLE001 — one bad ticket must not stop the queue
      out = error
    return self._result(ticket, out, started)

  def triage_many(self, tickets: list[Ticket], concurrency: int = 4) -> list[TriageResult]:
    """Run tickets concurrently; failures come back as results, in input order.

    `self.structured.batch(...)` runs calls in a thread pool too, but reports one wall time for the whole batch;
    mapping `triage` over a pool keeps each ticket's own latency for the scorecard.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
      return list(pool.map(self.triage, tickets))

  async def atriage(self, ticket: Ticket) -> TriageResult:
    started = time.perf_counter()
    try:
      out = await self.structured.ainvoke(self.messages(ticket))
    except Exception as error:  # noqa: BLE001
      out = error
    return self._result(ticket, out, started)

  async def atriage_many(self, tickets: list[Ticket], concurrency: int = 4) -> list[TriageResult]:
    """The async way to run many: a semaphore bounds how many calls are in flight at once."""
    gate = asyncio.Semaphore(concurrency)

    async def one(ticket: Ticket) -> TriageResult:
      async with gate:
        return await self.atriage(ticket)

    return await asyncio.gather(*(one(t) for t in tickets))


def reply_messages(ticket: Ticket, triage: TicketTriage, history: list[BaseMessage] | None = None) -> list[BaseMessage]:
  return REPLY_PROMPT.invoke(
    {
      "ticket": ticket.text,
      "language": triage.language,
      "next_step": NEXT_STEPS[triage.category],
      "history": history or [],
    }
  ).to_messages()


def stream_reply(model: BaseChatModel, ticket: Ticket, triage: TicketTriage, history=None) -> Iterator[str]:
  """The reply draft, token by token: the agent sees it start in under a second instead of waiting for all of it."""
  for chunk in model.stream(reply_messages(ticket, triage, history)):
    if chunk.text:
      yield chunk.text


async def astream_reply(model: BaseChatModel, ticket: Ticket, triage: TicketTriage, history=None) -> AsyncIterator[str]:
  async for chunk in model.astream(reply_messages(ticket, triage, history)):
    if chunk.text:
      yield chunk.text
