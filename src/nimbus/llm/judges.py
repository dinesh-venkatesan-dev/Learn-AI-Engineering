"""Models used as judges (openevals, agentevals). Judges ask for structured output; on Groq's gpt-oss models the
default tool-calling route sometimes fails ("Tool choice is required, but model did not call a tool": the model
answered in prose). JSON-schema mode doesn't need a tool call, so judges use it."""

from langchain_groq import ChatGroq

import nimbus.llm.models  # noqa: F401 - loads GROQ_API_KEY from ai-course/.env
from nimbus.config import get_settings


class JsonSchemaJudge(ChatGroq):
  def with_structured_output(self, schema, *, method: str = "json_schema", **kwargs):
    return super().with_structured_output(schema, method=method, **kwargs)


def judge_chat_model(name: str | None = None) -> ChatGroq:
  return JsonSchemaJudge(model=name or get_settings().judge_models[0], temperature=0, max_retries=4)
