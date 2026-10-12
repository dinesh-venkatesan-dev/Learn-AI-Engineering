"""When the provider fails: a fallback on another provider, and a circuit breaker that stops calling a provider that
keeps failing.

  resilient = with_fallback(primary, backup)           primary.with_fallbacks([backup]): every call tries both
  breaker = CircuitBreaker(primary, backup)             after N failures in a row, skip the primary for a while
  reply = await breaker.ainvoke(messages)               breaker.last_route says who answered

Timeouts and retries live on the clients themselves (get_chat_model: timeout_s, max_retries).
"""

import logging
import time

from langchain_core.language_models import BaseChatModel

from nimbus.llm.errors import classify

log = logging.getLogger(__name__)


def with_fallback(primary: BaseChatModel, backup: BaseChatModel):
  return primary.with_fallbacks([backup])


class CircuitBreaker:
  """Closed: call the primary. After `threshold` failures in a row: open, go straight to the backup for `cooldown`
  seconds. Then half-open: try the primary once more."""

  def __init__(self, primary, backup, threshold: int = 2, cooldown_s: float = 30, clock=time.monotonic):
    self.primary, self.backup = primary, backup
    self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
    self.failures, self.opened_at, self.last_route = 0, None, ""

  @property
  def state(self) -> str:
    if self.opened_at is None:
      return "closed"
    return "open" if self.clock() - self.opened_at < self.cooldown_s else "half-open"

  def _failed(self, error: Exception) -> None:
    self.failures += 1
    if self.failures >= self.threshold and self.opened_at is None:
      log.warning("circuit opened after %d failures: %s", self.failures, classify(error).reason)
    if self.failures >= self.threshold:
      self.opened_at = self.clock()

  async def ainvoke(self, messages, config=None):
    if self.state == "open":
      self.last_route = "backup (circuit open: primary skipped)"
      return await self.backup.ainvoke(messages, config)
    try:
      reply = await self.primary.ainvoke(messages, config)
    except Exception as error:  # noqa: BLE001 - any primary failure counts; the backup answers
      self._failed(error)
      self.last_route = f"backup (primary failed, {self.failures} in a row)"
      return await self.backup.ainvoke(messages, config)
    self.failures, self.opened_at, self.last_route = 0, None, "primary"
    return reply

  def invoke(self, messages, config=None):
    if self.state == "open":
      self.last_route = "backup (circuit open: primary skipped)"
      return self.backup.invoke(messages, config)
    try:
      reply = self.primary.invoke(messages, config)
    except Exception as error:  # noqa: BLE001
      self._failed(error)
      self.last_route = f"backup (primary failed, {self.failures} in a row)"
      return self.backup.invoke(messages, config)
    self.failures, self.opened_at, self.last_route = 0, None, "primary"
    return reply
