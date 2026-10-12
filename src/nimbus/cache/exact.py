"""An exact cache for model calls, shared by every server process (Redis), plugged into LangChain with
set_llm_cache(). Same messages + same model settings → the stored response, no API call.

The key is a hash of the prompt and the model's settings (llm_string): change the temperature or the model and it's
a miss. normalize=True also folds case, whitespace and trailing punctuation, which raises the hit rate without
changing any answer ("how much is a cold brew" == "How much is a cold brew?").
"""

import hashlib
import re

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation

ALLOWED = [Generation, ChatGeneration, ChatGenerationChunk, AIMessage, AIMessageChunk]


def normalize(text: str) -> str:
  text = re.sub(r"\s+", " ", text.lower())
  return re.sub(r"[?!.,\s]+(?=[\"'\]}])", "", text).strip(" ?!.")


def cache_key(prompt: str, llm_string: str, namespace: str, normalized: bool) -> str:
  text = normalize(prompt) if normalized else prompt
  return f"{namespace}:{hashlib.sha256((llm_string + '|' + text).encode()).hexdigest()}"


class RedisLLMCache(BaseCache):
  def __init__(self, redis, ttl_s: int = 3600, namespace: str = "llmcache", normalized: bool = True):
    self.redis, self.ttl_s, self.namespace, self.normalized = redis, ttl_s, namespace, normalized
    self.hits = self.misses = 0

  def lookup(self, prompt: str, llm_string: str):
    raw = self.redis.get(cache_key(prompt, llm_string, self.namespace, self.normalized))
    if raw is None:
      self.misses += 1
      return None
    self.hits += 1
    return loads(raw, allowed_objects=ALLOWED)  # only what a model call returns: a cache is not a place to run code

  def update(self, prompt: str, llm_string: str, return_val) -> None:
    key = cache_key(prompt, llm_string, self.namespace, self.normalized)
    self.redis.set(key, dumps(return_val), ex=self.ttl_s)  # expires: price changes show up within the TTL

  def clear(self, **kwargs) -> None:
    keys = list(self.redis.scan_iter(f"{self.namespace}:*"))
    if keys:
      self.redis.delete(*keys)
