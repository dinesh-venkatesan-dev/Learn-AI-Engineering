"""Support pipeline v2: one LCEL chain from a customer message to a routed reply.

  {"ticket": "..."}
    → triage     prompt | model.with_structured_output(TicketTriage)      (retried once if the reply doesn't parse)
    → route      safety tickets get a fixed, approved message and a human; everything else gets a drafted reply
    → reply      prompt | model | StrOutputParser()
  = {"ticket": ..., "triage": TicketTriage, "reply": str}

Each model step falls back to a second provider when the first one is down: the client already retries temporary
failures (timeouts, 429, 5xx); when it gives up, `with_fallbacks` sends the same step to the backup model.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough
from pydantic import ValidationError

from nimbus.llm import classify
from nimbus.llm.usage import ModelCall, UsageRecorder
from nimbus.support.prompts import NEXT_STEPS, REPLY_PROMPT, TRIAGE_PROMPTS
from nimbus.support.schema import TicketTriage

# Safety tickets never get a drafted reply: a fixed message a human approved, and a person calls the customer.
FIXED_REPLIES = {
  "allergy_safety": "We're very sorry. A member of our team will call you right away. If you feel unwell, please "
  "seek medical help now.",
}


def triage_step(model: BaseChatModel, prompt_version: str = "v2", method: str = "json_schema") -> Runnable:
  """Ticket → TicketTriage. A reply that doesn't match the schema is asked for once more."""
  structured = model.with_structured_output(TicketTriage, method=method)
  chain = (TRIAGE_PROMPTS[prompt_version] | structured).with_retry(
    retry_if_exception_type=(OutputParserException, ValidationError), stop_after_attempt=2
  )

  # When the pipeline is streamed, structured output would emit partial TicketTriage objects as the JSON arrives,
  # and LCEL can't merge partial objects. A routing decision is only useful complete: run this step as one call.
  # Passing `config` on keeps callbacks, tags and traces for the inner runs.
  def complete(x: dict, config: RunnableConfig) -> TicketTriage:
    return chain.invoke(x, config=config)

  async def acomplete(x: dict, config: RunnableConfig) -> TicketTriage:
    return await chain.ainvoke(x, config=config)

  return RunnableLambda(complete, afunc=acomplete, name="triage_once")


def reply_step(model: BaseChatModel) -> Runnable:
  """{ticket, triage, history?} → the reply draft as a string."""

  def reply_inputs(x: dict) -> dict:
    triage: TicketTriage = x["triage"]
    return {
      "ticket": x["ticket"],
      "language": triage.language,
      "next_step": NEXT_STEPS[triage.category],
      "history": x.get("history", []),
    }

  return RunnableLambda(reply_inputs) | REPLY_PROMPT | model | StrOutputParser()


def build_pipeline(
  model: BaseChatModel, fallback: BaseChatModel | None = None, prompt_version: str = "v2", method: str = "json_schema"
) -> Runnable:
  triage = triage_step(model, prompt_version, method)
  reply = reply_step(model)
  if fallback is not None:  # the same step on the backup model, tried only if the primary fails
    triage = triage.with_fallbacks([triage_step(fallback, prompt_version, method)])
    reply = reply.with_fallbacks([reply_step(fallback)])

  def route(x: dict) -> Runnable:
    """Returning a Runnable from a function makes LCEL run it on the same input."""
    fixed = FIXED_REPLIES.get(x["triage"].category)
    return RunnableLambda(lambda _: fixed, name="fixed_reply") if fixed else reply

  return (
    RunnablePassthrough.assign(triage=triage.with_config(run_name="triage", tags=["triage"]))
    | RunnablePassthrough.assign(reply=RunnableLambda(route, name="route").with_config(tags=["reply"]))
  ).with_config(run_name="support_pipeline")


@dataclass
class PipelineResult:
  ticket_id: str
  triage: TicketTriage | None
  reply: str | None
  ms: float
  calls: list[ModelCall] = field(default_factory=list)  # every model call, with the model that actually answered
  error: str | None = None
  action: str | None = None


def run(pipeline: Runnable, ticket_id: str, text: str, history: list | None = None) -> PipelineResult:
  """Run one ticket with a UsageRecorder attached, tagged with the ticket id for tracing."""
  recorder = UsageRecorder()
  config: RunnableConfig = {"callbacks": [recorder], "tags": ["support"], "metadata": {"ticket_id": ticket_id}}
  started = time.perf_counter()
  try:
    out = pipeline.invoke({"ticket": text, "history": history or []}, config=config)
  except Exception as error:  # noqa: BLE001 — a failed ticket is a result, not a crash
    failure = classify(error)
    return PipelineResult(
      ticket_id, None, None, (time.perf_counter() - started) * 1000, recorder.calls, failure.reason, failure.action
    )
  return PipelineResult(ticket_id, out["triage"], out["reply"], (time.perf_counter() - started) * 1000, recorder.calls)


async def stream_events(pipeline: Runnable, text: str) -> AsyncIterator[tuple[str, object]]:
  """For a live UI: the triage as soon as it's known, then the reply token by token (astream_events)."""
  async for event in pipeline.astream_events({"ticket": text, "history": []}, version="v2"):
    output = event["data"].get("output")
    if event["event"] == "on_chain_end" and event["name"] == "triage" and isinstance(output, TicketTriage):
      yield "triage", output
    elif event["event"] == "on_chat_model_stream" and "reply" in event.get("tags", []):
      if text_piece := event["data"]["chunk"].text:
        yield "token", text_piece
    elif event["event"] == "on_chain_end" and event["name"] == "fixed_reply":
      yield "token", output


def support_pipeline(settings=None) -> Runnable:
  """The production pipeline: the default model, the fallback provider behind it, the configured prompt and method."""
  from nimbus.config import get_settings
  from nimbus.llm import get_chat_model

  s = settings or get_settings()
  return build_pipeline(get_chat_model(s), get_chat_model(s, role="fallback"), s.triage_prompt_version, s.triage_method)
