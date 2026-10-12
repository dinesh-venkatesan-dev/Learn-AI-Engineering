"""Chat models: one configured client per role, so retries, timeouts, rate limits and capabilities live in one place.

Roles:
  default    the everyday model (Mistral's free plan by default)
  small      a smaller, cheaper model on the same provider, for simple questions (routing)
  reasoning  a model that thinks before answering and returns its reasoning (Groq's free plan by default)
  fallback   a backup on a different provider, for when the default provider is down
  local      a model on this computer through Ollama: no key, no rate limit, works offline

The provider and model of each role are configuration (NIMBUS_CHAT_MODEL, NIMBUS_REASONING_MODEL, ...), not code.
"""

import inspect
import logging
import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from nimbus.config import PROJECT_DIR, Settings, get_settings

load_dotenv(PROJECT_DIR / ".env")  # provider API keys (MISTRAL_API_KEY, GROQ_API_KEY) live in .env, never in code

log = logging.getLogger(__name__)

Role = Literal["default", "small", "reasoning", "fallback", "local"]
API_KEYS = {"mistralai": "MISTRAL_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}

# What a model can do, for models the installed provider package has no profile for (model.profile is None).
# Checked against the provider's model card; keys follow LangChain's model profile format.
KNOWN_PROFILES: dict[str, dict] = {
  "ministral-14b-latest": {
    "max_input_tokens": 262_144,
    "image_inputs": True,
    "pdf_inputs": False,
    "reasoning_output": False,
    "tool_calling": True,
    "structured_output": True,
  },
  "qwen3:1.7b": {
    "max_input_tokens": 40_960,
    "image_inputs": False,
    "reasoning_output": True,
    "tool_calling": True,
    "structured_output": True,
  },
}


@lru_cache
def rate_limiter(provider: str, requests_per_second: float) -> InMemoryRateLimiter:
  """ONE limiter per provider: provider limits count every request from your key, whichever model it names."""
  return InMemoryRateLimiter(requests_per_second=requests_per_second, check_every_n_seconds=0.1, max_bucket_size=5)


_OPENED: list[BaseChatModel] = []  # every cached client, so shutdown can close their HTTP connections


@lru_cache
def _client(
  provider: str, model: str, requests_per_second: float | None, timeout_s: float, max_retries: int, extra: tuple = ()
) -> BaseChatModel:
  # One client per configuration: creating a client per request would leak HTTP connections.
  options = dict(extra)
  if requests_per_second:
    options["rate_limiter"] = rate_limiter(provider, requests_per_second)
  chat = init_chat_model(model, model_provider=provider, timeout=timeout_s, max_retries=max_retries, **options)
  if chat.profile is None and model in KNOWN_PROFILES:
    chat = chat.model_copy(update={"profile": KNOWN_PROFILES[model]})
  _OPENED.append(chat)
  return chat


async def aclose_clients() -> None:
  """Close every cached client's HTTP connections: call once at shutdown (the web app's lifespan does)."""
  for chat in _OPENED:
    for name in ("client", "async_client"):
      client = getattr(chat, name, None)
      client = getattr(client, "_client", client)  # Groq/OpenAI wrap the real client (Completions → Groq)
      close = getattr(client, "aclose", None) or getattr(client, "close", None)
      if close is None:
        continue
      try:
        result = close()
        if inspect.isawaitable(result):
          await result
      except RuntimeError:  # async connections opened in an event loop that has already ended can't be closed here
        log.debug("skipped closing %s of %s: its event loop is closed", name, type(chat).__name__)
  _OPENED.clear()
  _client.cache_clear()


def get_chat_model(settings: Settings | None = None, role: Role = "default", **overrides) -> BaseChatModel:
  """The chat model for a role. `overrides` (temperature=..., timeout_s=...) make a separately cached client."""
  s = settings or get_settings()
  if role == "local":
    spec = {"provider": "ollama", "model": s.local_model, "requests_per_second": None, "base_url": s.ollama_url}
  elif role == "fallback":
    spec = {
      "provider": s.fallback_provider,
      "model": s.fallback_model,
      "requests_per_second": s.fallback_requests_per_second,
    }
  elif role == "small":
    spec = {"provider": s.chat_provider, "model": s.small_model, "requests_per_second": s.chat_requests_per_second}
  elif role == "reasoning":
    spec = {
      "provider": s.reasoning_provider,
      "model": s.reasoning_model,
      "requests_per_second": s.reasoning_requests_per_second,
    }
  else:
    spec = {"provider": s.chat_provider, "model": s.chat_model, "requests_per_second": s.chat_requests_per_second}
  spec |= {"temperature": 0, "timeout_s": s.chat_timeout_s, "max_retries": s.chat_max_retries} | overrides
  key = API_KEYS.get(spec["provider"])
  if key and not os.environ.get(key):
    raise RuntimeError(f"{key} is not set: add it to ai-course/.env")
  extra = tuple(
    sorted(
      (k, v)
      for k, v in spec.items()
      if k not in {"provider", "model", "requests_per_second", "timeout_s", "max_retries"}
    )
  )
  return _client(
    spec["provider"], spec["model"], spec["requests_per_second"], spec["timeout_s"], spec["max_retries"], extra
  )


def supports(model: BaseChatModel, capability: str) -> bool:
  """Check the model's profile before sending something it can't handle (an image to a text-only model)."""
  return bool((model.profile or {}).get(capability))


def model_name(model: BaseChatModel) -> str:
  return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__
