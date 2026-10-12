"""Spend on the right model: a cheap rule sends simple questions to the small model, and every request's tokens are
priced from config.

  role = route(question)                      "small" or "default"
  usd = cost(usage_by_model, settings)        get_usage_metadata_callback() collects usage_by_model
"""

import re
from typing import Literal

from nimbus.config import Settings


def route(question: str) -> Literal["small", "default"]:
  """Several parts ("and", commas, more than one question) or a long question → the bigger model."""
  parts = len(re.findall(r"\band\b|,|\?", question))
  return "default" if parts > 1 or len(question) > 80 else "small"


def cost(usage_by_model: dict[str, dict], settings: Settings) -> float:
  """US$ for a request: each model's input and output tokens at that model's price (per million tokens)."""
  total = 0.0
  for model, usage in usage_by_model.items():
    price_in, price_out = settings.model_prices.get(model, (0.0, 0.0))
    total += (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1_000_000
  return total


class RoutedModel:
  """The production setup: each question goes to the small or the default model by `route`, and each of those sits
  behind a circuit breaker with the backup provider.

    models = RoutedModel.from_settings(settings)
    reply = await models.ainvoke(messages, question)      models.last_role, models.last_route say what happened
  """

  def __init__(self, small, default, backup, threshold: int = 2, cooldown_s: float = 30):
    from nimbus.llm.resilience import CircuitBreaker

    self.breakers = {
      "small": CircuitBreaker(small, backup, threshold, cooldown_s),
      "default": CircuitBreaker(default, backup, threshold, cooldown_s),
    }
    self.last_role, self.last_route = "", ""

  @classmethod
  def from_settings(cls, settings: Settings, **overrides) -> "RoutedModel":
    from nimbus.llm import get_chat_model

    return cls(
      get_chat_model(settings, role="small", **overrides),
      get_chat_model(settings, **overrides),
      get_chat_model(settings, role="fallback"),
      settings.circuit_threshold,
      settings.circuit_cooldown_s,
    )

  async def ainvoke(self, messages, question: str, config=None):
    self.last_role = route(question)
    breaker = self.breakers[self.last_role]
    reply = await breaker.ainvoke(messages, config)
    self.last_route = breaker.last_route
    return reply
