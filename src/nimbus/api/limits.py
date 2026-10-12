"""Keeping abuse out, and spend in check: plans per API key, a Redis rate limiter shared by every server process, and
per-caller daily token budgets.

  RateLimited   dependency: INCR a per-minute counter (fixed window); over the plan's limit → 429 + Retry-After
  TokenBudget   dependency: over today's budget → 429 BEFORE any model call; .spend(tokens) after the call
Without Redis the API still serves (fail open, logged): a limiter outage shouldn't become an API outage.
"""

import hashlib
import logging
import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Header, Request

from nimbus.api.deps import SettingsDep
from nimbus.api.errors import ApiError

log = logging.getLogger("nimbus.api.limits")


class TooManyRequests(ApiError):
  status, code = 429, "rate_limited"


class BudgetExceeded(ApiError):
  status, code = 429, "budget_exceeded"


def key_hash(key: str) -> str:
  return hashlib.sha256(key.encode()).hexdigest()


def caller_plan(settings: SettingsDep, x_api_key: Annotated[str | None, Header()] = None) -> tuple[str, str]:
  """(caller id, plan): a key's plan comes from its HASH; plain keys (api_keys) are pro. With no keys configured the
  API is open for local development: plan "dev", no limits."""
  if not settings.api_key_hashes and not settings.api_keys:
    return "dev", "dev"
  if x_api_key:
    digest = key_hash(x_api_key)
    if digest in settings.api_key_hashes:
      return digest[:12], settings.api_key_hashes[digest]
    if any(x_api_key == k.get_secret_value() for k in settings.api_keys):
      return digest[:12], "pro"
  return "anonymous", "free"


CallerPlan = Annotated[tuple[str, str], Depends(caller_plan)]


def redis_of(request: Request):
  return getattr(request.app.state, "redis", None)


def rate_limited(request: Request, caller: CallerPlan, settings: SettingsDep) -> str:
  caller_id, plan = caller
  redis = redis_of(request)
  if plan == "dev":
    return caller_id
  if redis is None:
    log.warning("no Redis: rate limits off")
    return caller_id
  window = int(time.time() // 60)
  key = f"ratelimit:{caller_id}:{window}"
  count = redis.incr(key)
  if count == 1:
    redis.expire(key, 60)
  limit = settings.plan_limits.get(plan, 0)
  if count > limit:
    retry_after = 60 - int(time.time() % 60)
    raise TooManyRequests(
      f"Rate limit: {limit} requests/minute on the {plan} plan.", headers={"Retry-After": str(retry_after)}
    )
  return caller_id


RateLimited = Annotated[str, Depends(rate_limited)]


class TokenBudget:
  def __init__(self, request: Request, caller: CallerPlan, settings: SettingsDep):
    self.redis = redis_of(request) if caller[1] != "dev" else None  # dev: no budget
    self.limit = settings.daily_token_budget
    self.key = f"tokens:{caller[0]}:{datetime.now(UTC).date()}"
    if self.redis is not None and int(self.redis.get(self.key) or 0) >= self.limit:  # before spending anything
      raise BudgetExceeded(f"Today's budget of {self.limit:,} tokens is used. It resets at midnight UTC.")

  def spend(self, tokens: int) -> int:
    if self.redis is None:
      return 0
    pipe = self.redis.pipeline()
    pipe.incrby(self.key, tokens).expire(self.key, 2 * 24 * 3600)
    return pipe.execute()[0]


BudgetDep = Annotated[TokenBudget, Depends(TokenBudget)]
