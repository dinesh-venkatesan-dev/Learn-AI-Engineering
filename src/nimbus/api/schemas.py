"""What the Nimbus API accepts and returns. One model per direction: what a client sends, what we keep, what a client
may see, so internal fields (the system prompt, the model, costs) never leak by accident."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
  role: Literal["user", "assistant"]
  content: str = Field(min_length=1, max_length=2000)


class ReplyRequest(BaseModel):
  """A customer's message, the conversation so far, and how to answer."""

  model_config = {
    "extra": "forbid",
    "json_schema_extra": {"examples": [{"message": "My cold brew arrived warm. What can I do?", "tone": "warm"}]},
  }

  message: str = Field(min_length=1, max_length=2000)
  history: list[ChatMessage] = Field(default_factory=list, max_length=20, description="Earlier turns, oldest first")
  tone: Literal["warm", "formal"] = "warm"
  max_words: int = Field(60, ge=10, le=200)


class Draft(BaseModel):
  """What the service keeps about a reply: internal, never sent as is."""

  answer: str
  system_prompt: str
  model: str
  input_tokens: int = 0
  output_tokens: int = 0


class ReplyPublic(BaseModel):
  """What the client sees."""

  answer: str
  words: int
