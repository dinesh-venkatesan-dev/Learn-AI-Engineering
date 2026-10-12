"""A semantic cache: reuse an answer whose question MEANS the same, safely.

lookup     embed the question, find the nearest cached question IN THE SAME SCOPE, reuse its answer when
           similarity ≥ threshold, the entry's versions match (model, prompt, index), it hasn't expired, and the
           ENTITY GUARD passes: numbers, order ids, dates and names in both questions must be the same
           ("cancel NC-1002?" and "cancel NC-1003?" are 0.98 similar and have different answers)
store      only answers that are safe to reuse: never personal or permission-dependent ones, time-sensitive ones,
           refusals or errors
invalidate a version bump (a new prompt, re-ingested documents) retires old entries; purge(sources) on a change
get_or_compute   concurrent identical questions wait for ONE model call (no cache stampede)

cache = SemanticCache(MemoryStore(), embeddings, threshold=0.9, versions={"model": "...", "prompt": "v3"})
"""

import asyncio
import re
from dataclasses import dataclass, field
from functools import cache

from nimbus.cache.stores import Entry

ENTITY = re.compile(
  r"\bNC-\d+\b|\bc-\d{4}\b|\d{4}-\d{2}-\d{2}|₹?\d[\d,]*(?:\.\d+)?|\b(?:Mumbai|Pune|Delhi|Chennai|Bengaluru|Kolkata)\b",
  re.IGNORECASE,
)
PERSONAL = re.compile(r"\bNC-\d+\b|\bc-\d{4}\b|\bmy (?:order|account|subscription|refund|points|bag)\b", re.IGNORECASE)
TIME_SENSITIVE = re.compile(r"\b(?:today|now|tonight|this week|right now|currently|latest|open now)\b", re.IGNORECASE)
NOT_AN_ANSWER = re.compile(r"\b(?:I don't know|I can't|I cannot|sorry|error|unavailable|try again)\b", re.IGNORECASE)


@cache
def known_names() -> re.Pattern:
  """Nimbus's own names are entities too: menu items and cafés ("cold brew" vs "cappuccino", "Bandra" vs "Colaba")."""
  from nimbus.data.warehouse import CAFES, PRODUCTS
  from nimbus.support.tools import SupportStore

  names = {row["item"] for row in SupportStore().menu.values()} | {p[1] for p in PRODUCTS}
  names |= {c[1].removeprefix("Nimbus ") for c in CAFES}
  return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b", re.I)


def entities(text: str) -> set[str]:
  found = {m.group().lower().replace(",", "").lstrip("₹") for m in ENTITY.finditer(text)}
  return found | {m.group().lower() for m in known_names().finditer(text)}


def cacheable(question: str, answer: str) -> tuple[bool, str]:
  """Is this answer safe to give to someone else who asks the same?"""
  if PERSONAL.search(question):
    return False, "personal: depends on whose order or account"
  if TIME_SENSITIVE.search(question):
    return False, "time-sensitive"
  if not answer.strip() or NOT_AN_ANSWER.search(answer):
    return False, "a refusal or an error, not an answer"
  return True, ""


@dataclass
class Lookup:
  answer: str | None
  similarity: float = 0.0
  matched: str = ""
  reason: str = ""

  @property
  def hit(self) -> bool:
    return self.answer is not None


@dataclass
class CacheStats:
  hits: int = 0
  misses: int = 0
  stored: int = 0
  skipped: int = 0
  guarded: int = 0  # near-misses the entity guard refused
  saved_tokens: int = 0
  log: list[str] = field(default_factory=list)

  @property
  def hit_rate(self) -> float:
    return self.hits / max(self.hits + self.misses, 1)


class SemanticCache:
  def __init__(
    self,
    store,
    embeddings,
    threshold: float = 0.9,
    ttl_s: float = 3600,
    versions: dict[str, str] | None = None,
    entity_guard: bool = True,
    rules: bool = True,
  ):
    self.store, self.embeddings, self.threshold, self.ttl_s = store, embeddings, threshold, ttl_s
    self.versions = {"model": "", "prompt": "", "index": ""} | (versions or {})
    self.entity_guard, self.rules = entity_guard, rules
    self.stats = CacheStats()
    self._inflight: dict[tuple[str, str], asyncio.Future] = {}

  def lookup(self, question: str, scope: str = "public:en") -> Lookup:
    if self.rules and (PERSONAL.search(question) or TIME_SENSITIVE.search(question)):
      self.stats.misses += 1
      return Lookup(None, reason="never cached: personal or time-sensitive")
    found = self.store.nearest(self.embeddings.embed_query(question), scope)
    if found is None:
      self.stats.misses += 1
      return Lookup(None, reason="empty")
    entry, similarity = found
    if similarity < self.threshold:
      self.stats.misses += 1
      return Lookup(None, similarity, entry.question, "not similar enough")
    stale = (entry.model, entry.prompt_version, entry.index_version) != tuple(self.versions.values())
    if stale or entry.expired():
      self.store.remove(entry.id)  # retired: a new model, prompt or index version, or past its TTL
      self.stats.misses += 1
      return Lookup(None, similarity, entry.question, "stale version" if stale else "expired")
    if self.entity_guard and entities(question) != entities(entry.question):
      self.stats.guarded += 1
      self.stats.misses += 1
      return Lookup(None, similarity, entry.question, "entity guard: a number, id, date or place differs")
    self.store.hit(entry)
    self.stats.hits += 1
    return Lookup(entry.answer, similarity, entry.question, "hit")

  def save(self, question: str, answer: str, scope: str = "public:en", sources=(), tokens: int = 0) -> bool:
    ok, why = cacheable(question, answer) if self.rules else (True, "")
    if not ok:
      self.stats.skipped += 1
      self.stats.log.append(f"not stored ({why}): {question}")
      return False
    entry = Entry(
      question=question,
      answer=answer,
      vector=self.embeddings.embed_query(question),
      scope=scope,
      model=self.versions["model"],
      prompt_version=self.versions["prompt"],
      index_version=self.versions["index"],
      sources=list(sources),
      ttl_s=self.ttl_s,
    )
    self.store.add(entry)
    self.stats.stored += 1
    return True

  async def get_or_compute(self, question: str, compute, scope: str = "public:en") -> tuple[str, bool]:
    """(answer, from_cache). Identical questions arriving together share ONE call: the first computes, the rest
    await its future (request coalescing)."""
    found = await asyncio.to_thread(self.lookup, question, scope)
    if found.hit:
      return found.answer, True
    key = (scope, question.strip().lower())
    if key in self._inflight:
      return await self._inflight[key], True
    future = asyncio.get_running_loop().create_future()
    self._inflight[key] = future
    try:
      answer = await compute(question)
      future.set_result(answer)
      await asyncio.to_thread(self.save, question, answer, scope)
      return answer, False
    except Exception as error:
      future.set_exception(error)
      raise
    finally:
      self._inflight.pop(key, None)


def tune_threshold(pairs: list[dict], embeddings, thresholds=(0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94)) -> list[dict]:
  """From labelled pairs (should_hit: paraphrases; must_not: near-misses): at each threshold, how many paraphrases
  hit and how many near-misses would wrongly hit."""
  import numpy as np

  def sim(a: str, b: str) -> float:
    va, vb = (np.asarray(v) for v in embeddings.embed_documents([a, b]))
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

  scored = [(p["kind"], sim(p["a"], p["b"]), entities(p["a"]) == entities(p["b"])) for p in pairs]
  rows = []
  for t in thresholds:
    rows.append(
      {
        "threshold": t,
        "paraphrases_hit": sum(kind == "should_hit" and s >= t for kind, s, _ in scored),
        "paraphrases": sum(kind == "should_hit" for kind, _, _ in scored),
        "false_hits": sum(kind == "must_not" and s >= t for kind, s, _ in scored),
        "false_hits_with_guard": sum(kind == "must_not" and s >= t and same for kind, s, same in scored),
        "near_misses": sum(kind == "must_not" for kind, _, _ in scored),
      }
    )
  return rows
