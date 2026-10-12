"""The support reply's quality as pytest tests, logged to LangSmith when test tracking is on (LANGSMITH_TEST_SUITE
names the suite; every run is an experiment). They call the real model: marked `integration`.

  uv run pytest -m integration tests/test_reply_quality.py
"""

import re

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import testing as t

PROMISE = (
  r"(?i)\bwe(?:'ll| will) (?:refund|send|deliver|replace)|refund(?:ed)? (?:has been|is) (?:issued|processed)|today"
)


def draft(message: str) -> str:
  from nimbus.api.routers.support import system_prompt
  from nimbus.api.schemas import ReplyRequest
  from nimbus.llm import get_chat_model

  body = ReplyRequest(message=message)
  return get_chat_model().invoke([SystemMessage(system_prompt(body)), HumanMessage(message)]).text


@pytest.mark.integration
@pytest.mark.langsmith
def test_no_promises_to_an_angry_customer():
  message = "I want my refund TODAY or I'll complain everywhere!"
  t.log_inputs({"message": message})
  reply = draft(message)
  t.log_outputs({"answer": reply})
  t.log_reference_outputs({"must_not": PROMISE})
  promised = re.search(PROMISE, reply)
  t.log_feedback(key="no_promises", score=int(promised is None))
  assert promised is None, promised.group() if promised else ""


@pytest.mark.integration
@pytest.mark.langsmith
def test_a_judge_finds_the_reply_polite():
  from openevals.llm import create_llm_as_judge

  from nimbus.llm.judges import judge_chat_model

  message = "Your delivery was late again. This is the third time."
  t.log_inputs({"message": message})
  reply = draft(message)
  t.log_outputs({"answer": reply})
  judge = create_llm_as_judge(
    prompt="Is this support reply polite and does it acknowledge the problem? Customer: {inputs} Reply: {outputs}",
    judge=judge_chat_model(),
    feedback_key="polite",
  )
  with t.trace_feedback():  # the judge's own call is traced apart from the app's
    verdict = judge(inputs=message, outputs=reply)
  t.log_feedback(key="polite", score=int(verdict["score"]))
  assert verdict["score"], verdict["comment"]
