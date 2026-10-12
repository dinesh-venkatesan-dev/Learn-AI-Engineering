"""The model layer and support triage, without calling a model."""

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from nimbus.llm import ConversationStore, Usage, classify
from nimbus.support.prompts import EXAMPLES, TRIAGE_PROMPTS
from nimbus.support.schema import Ticket, TicketTriage
from nimbus.support.triage import Triager


def http_error(status: int, body: str = '{"message": "nope"}') -> httpx.HTTPStatusError:
  request = httpx.Request("POST", "https://api.example/v1/chat")
  return httpx.HTTPStatusError("error", request=request, response=httpx.Response(status, request=request, text=body))


@pytest.mark.parametrize(
  ("error", "action"),
  [
    (http_error(429), "retry"),
    (http_error(503), "retry"),
    (http_error(401), "fix"),
    (http_error(400), "fix"),
    (httpx.ReadTimeout("timed out"), "retry"),
    (httpx.ConnectError("refused"), "retry"),
    (ValueError("something else"), "fix"),
  ],
)
def test_classify_sorts_failures(error, action):
  assert classify(error).action == action


def test_classify_keeps_the_providers_message():
  assert classify(http_error(400, '{"message": "Invalid model: x"}')).detail == "Invalid model: x"


def test_conversation_round_trip(tmp_path):
  store = ConversationStore(tmp_path)
  messages = [
    SystemMessage("be brief"),
    HumanMessage("hi"),
    AIMessage("hello", usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}),
  ]
  store.save("c1", messages)
  assert store.load("c1") == messages
  assert store.append("c1", HumanMessage("again"))[-1].text == "again"
  assert store.load("missing") == []


def test_conversation_id_cannot_escape_the_folder(tmp_path):
  with pytest.raises(ValueError):
    ConversationStore(tmp_path).load("../../etc/passwd")


def test_usage_and_cost():
  reply = AIMessage("x", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
  usage = Usage().add(reply).add(reply)
  assert (usage.input_tokens, usage.output_tokens, usage.calls) == (2000, 200, 2)
  assert usage.cost_usd(price_in=2.0, price_out=10.0) == pytest.approx(0.006)


def test_schema_rejects_an_invented_order_id():
  with pytest.raises(ValidationError):
    TicketTriage(
      category="order_problem",
      urgency="high",
      order_id="order 4410",
      language="en",
      sentiment="negative",
      summary="x",
      needs_human=False,
    )


def test_prompt_versions():
  v1 = TRIAGE_PROMPTS["v1"].invoke({"ticket": "hello"}).to_messages()
  v2 = TRIAGE_PROMPTS["v2"].invoke({"ticket": "hello"}).to_messages()
  assert len(v1) == 2 and v1[-1].text == "hello"
  assert len(v2) == 2 + 2 * len(EXAMPLES)  # system + an example pair each + the ticket
  for example in EXAMPLES:  # the examples must themselves be valid triages
    TicketTriage(**example["triage"])


class FakeStructured:
  """Stands in for model.with_structured_output(..., include_raw=True)."""

  def __init__(self, outcome):
    self.outcome = outcome

  def invoke(self, _messages):
    if isinstance(self.outcome, BaseException):
      raise self.outcome
    return self.outcome


def triager_returning(outcome) -> Triager:
  triager = Triager.__new__(Triager)
  triager.model, triager.prompt_version, triager.method = None, "v2", "json_schema"
  triager.prompt = TRIAGE_PROMPTS["v2"]
  triager.structured = FakeStructured(outcome)
  return triager


def test_triage_turns_failures_into_results():
  ticket = Ticket(id="t", text="hi")
  limited = triager_returning(http_error(429)).triage(ticket)
  assert (limited.triage, limited.action) == (None, "retry")
  raw = AIMessage("{}")
  bad = triager_returning({"raw": raw, "parsed": None, "parsing_error": ValueError("not json")}).triage(ticket)
  assert bad.action == "fix" and bad.raw is raw and "schema" in bad.error
