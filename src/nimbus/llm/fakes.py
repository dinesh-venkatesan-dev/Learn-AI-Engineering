"""A stand-in model for load tests: it answers after `delay` seconds without calling any provider, so a load test
measures the service, not the provider's rate limit (NIMBUS_FAKE_MODEL_DELAY)."""

import asyncio
import itertools
import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class SlowFakeModel(GenericFakeChatModel):
  delay: float = 1.0

  def bind_tools(self, tools, **kwargs):
    return self

  def _generate(self, *args, **kwargs):
    time.sleep(self.delay)
    return super()._generate(*args, **kwargs)

  async def _agenerate(self, *args, **kwargs):
    await asyncio.sleep(self.delay)
    return super()._generate(*args, **kwargs)


def slow_fake(delay: float) -> SlowFakeModel:
  reply = AIMessage("Thanks for reaching out! A teammate will check and get back to you.")
  return SlowFakeModel(messages=itertools.repeat(reply), delay=delay)
