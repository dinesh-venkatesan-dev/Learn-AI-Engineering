"""Production · Module 2 labs: caching. An exact cache for model calls (in memory, then in Redis, normalized), a
semantic cache with its dangers and guards, choosing its threshold from data, the stores it can live in, and what
(not) to cache. Then the caches replayed on support traffic."""

import asyncio
import json
import time

import redis as redis_lib
from langchain_core.messages import HumanMessage, SystemMessage

from academy.labs.runtime import Metric, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model

POLICY = (
  "Nimbus policy: a cold brew costs ₹200, a cappuccino ₹180; the Oat Latte is vegan (oat milk), the Mocha is not "
  "(dairy); refunds within 14 days of delivery for unopened bags; loyalty: 1 point per ₹10, points expire after 12 "
  "months; deliveries take 2 days in Mumbai, 3 to 5 days elsewhere. Order questions need a teammate."
)


def _redis():
  return redis_lib.Redis.from_url(get_settings().redis_url, socket_connect_timeout=0.5)


def _embeddings():
  from nimbus.rag.ingestion.embeddings import get_embeddings

  return get_embeddings(get_settings())


def ask(model, question: str):
  return model.invoke([SystemMessage(f"{POLICY} Answer in one sentence."), HumanMessage(question)])


# ── Lab 1: an exact cache ──────────────────────────────────────────────────────
def in_memory_cache(model, question: str):
  from langchain_core.caches import InMemoryCache
  from langchain_core.globals import set_llm_cache

  set_llm_cache(InMemoryCache())  # every model call in this process now checks the cache first
  try:
    timings = []
    for _ in range(2):
      started = time.perf_counter()
      reply = model.invoke(question)
      timings.append((time.perf_counter() - started, reply.usage_metadata))
    return timings
  finally:
    set_llm_cache(None)


def the_key_is_exact(model):
  from langchain_core.caches import InMemoryCache
  from langchain_core.globals import set_llm_cache

  cache = InMemoryCache()
  set_llm_cache(cache)
  try:
    model.invoke("In one sentence: what is a flat white?")
    warmer = model.bind(temperature=0.7)  # different settings → a different key → a miss
    calls = {}
    for label, runnable, prompt in (
      ("same prompt, same settings", model, "In one sentence: what is a flat white?"),
      ("same prompt, temperature 0.7", warmer, "In one sentence: what is a flat white?"),
      ("a paraphrase", model, "In one sentence: what's a flat white?"),
    ):
      started = time.perf_counter()
      runnable.invoke(prompt)
      calls[label] = time.perf_counter() - started
    return calls, len(cache._cache)
  finally:
    set_llm_cache(None)


def shared_in_redis(question: str):
  from langchain_core.globals import set_llm_cache

  from nimbus.cache.exact import RedisLLMCache

  with _redis() as client:
    cache = RedisLLMCache(client, ttl_s=3600, namespace="lab-llmcache", normalized=True)
    cache.clear()
    set_llm_cache(cache)  # in Redis: every server process shares it
    try:
      first = get_chat_model()
      second = first.model_copy()  # another client with the same settings, as a second process would have
      rows = []
      for label, model, q in (
        ("process A asks", first, question),
        ("process B asks the same", second, question),
        ("process B, different case and no '?'", second, question.lower().rstrip("?")),
      ):
        started = time.perf_counter()
        model.invoke(q)
        rows.append([label, f"{time.perf_counter() - started:.2f} s"])
      ttl = client.ttl(next(client.scan_iter("lab-llmcache:*")))
    finally:
      set_llm_cache(None)
  return rows, cache.hits, cache.misses, ttl


@lab(
  "prod.exact-cache",
  "An exact cache: identical requests answered for free",
  uses_model=True,
  calls="~5 calls to Mistral",
)
def exact_cache():
  """set_llm_cache with an in-memory cache, what changes the key, and the same cache in Redis shared by two clients,
  normalized so case and punctuation don't matter. Needs Redis for the last part (docker compose up -d)."""
  model = get_chat_model()
  timings = in_memory_cache(model, "In one sentence: what is a cortado?")
  keys, entries = the_key_is_exact(model)
  memory = table(
    ["call", "time", "tokens"],
    [[f"call {i + 1}", f"{t:.3f} s", json.dumps(u)] for i, (t, u) in enumerate(timings)],
  )
  key = table(
    ["request", "time", "from the cache?"], [[k, f"{v:.3f} s", "yes" if v < 0.05 else "no"] for k, v in keys.items()]
  )
  try:
    rows, hits, misses, ttl = shared_in_redis("How much is a cold brew?")
    shared = table(["", "time"], rows + [["hits / misses", f"{hits} / {misses}"], ["time to live", f"{ttl} s"]])
  except redis_lib.RedisError as error:
    shared, hits = table(["Redis", "unavailable"], [[str(error)[:80], ""]]), 0
  return [
    memory,
    key,
    shared,
    callout(
      f"The second identical call took {timings[1][0] * 1000:.0f} ms instead of {timings[0][0]:.2f} s: no API call, "
      "no tokens. The key is the exact prompt plus the model's settings, so a new temperature or a paraphrase is a "
      f"miss. In Redis the cache is shared by every process ({hits} hits), expires, and normalizing case and "
      "punctuation turns more requests into hits without changing any answer.",
      "decision",
      "Exact cache in Redis, normalized",
    ),
    example("lc.l58.1", "set_llm_cache()", in_memory_cache, "", [memory]),
    example("lc.l58.2", "The key: prompt + settings", the_key_is_exact, "", [key]),
    example("rag.l53.3", "Identical questions, instantly and free", in_memory_cache, "", [memory]),
    example(
      "prod.l07.1", "Shared in Redis, with a TTL", shared_in_redis, "nimbus/cache/exact.py RedisLLMCache", [shared]
    ),
    example("prod.l07.2", "Paraphrases miss", the_key_is_exact, "", [key]),
    example("new.l01.2", "A normalized key", shared_in_redis, "nimbus/cache/exact.py normalize", [shared]),
  ]


# ── Lab 2: a semantic cache ────────────────────────────────────────────────────
def semantic_basics(cache):
  cache.save("How much is a cold brew?", "A cold brew is ₹200.")
  cache.save("Can I cancel order NC-1002?", "NC-1002 has shipped, so it can't be cancelled.")  # refused: personal
  cache.save("Is the café open right now?", "Yes, until 10 pm.")  # refused: time-sensitive
  return [
    [q, *(lambda r: [r.answer or "—", f"{r.similarity:.3f}", r.reason])(cache.lookup(q))]
    for q in (
      "What does a cold brew cost?",
      "cold brew price?",
      "How much is a cappuccino?",
      "Can I cancel order NC-1003?",
    )
  ]


def the_trap(embeddings):
  import numpy as np

  a, b = (
    np.asarray(v) for v in embeddings.embed_documents(["Can I cancel order NC-1002?", "Can I cancel order NC-1003?"])
  )
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # nearly identical to an embedding model


def versions_and_ttl(store, embeddings):
  from nimbus.cache.semantic import SemanticCache

  old = SemanticCache(store, embeddings, threshold=0.75, versions={"prompt": "v1"})
  old.save("When do loyalty points expire?", "After 12 months.")
  new = SemanticCache(store, embeddings, threshold=0.75, versions={"prompt": "v2"})  # a new prompt version
  stale = new.lookup("When do loyalty points expire?")
  short = SemanticCache(store, embeddings, threshold=0.75, ttl_s=0.2, versions={"prompt": "v2"})
  short.save("Do you have decaf?", "Yes: decaf Nilgiri Sunrise.")
  time.sleep(0.3)
  expired = short.lookup("Do you have decaf?")
  return stale.reason, expired.reason


def scoped(store, embeddings):
  from nimbus.cache.semantic import SemanticCache

  cache = SemanticCache(store, embeddings, threshold=0.75)
  cache.save("How much is a cold brew?", "A cold brew is ₹200.", scope="public:en")
  cache.save("How much is a cold brew?", "ஒரு கோல்ட் ப்ரூ ₹200.", scope="public:ta")  # the Tamil site
  return cache.lookup("How much is a cold brew?", "public:ta").answer, cache.lookup(
    "How much is a cold brew?", "staff:en"
  ).reason


@lab("prod.semantic-cache", "A semantic cache: paraphrases hit, near-misses must not", calls="No model calls")
def semantic_cache():
  """A semantic cache in memory: paraphrases reuse an answer, a near-miss on an order id would too without the
  guards, personal and time-sensitive answers are never stored, version bumps and TTLs retire entries, and scopes
  keep languages and audiences apart."""
  from nimbus.cache.semantic import SemanticCache
  from nimbus.cache.stores import MemoryStore

  embeddings = _embeddings()
  cache = SemanticCache(MemoryStore(), embeddings, threshold=0.75, versions={"prompt": "v1"})
  rows = semantic_basics(cache)
  similarity = the_trap(embeddings)
  unguarded = SemanticCache(MemoryStore(), embeddings, threshold=0.75, entity_guard=False, rules=False)
  unguarded.save("Can I cancel order NC-1002?", "NC-1002 has shipped, so it can't be cancelled.")
  wrong = unguarded.lookup("Can I cancel order NC-1003?")
  stale, expired = versions_and_ttl(MemoryStore(), embeddings)
  tamil, staff = scoped(MemoryStore(), embeddings)
  lookups = table(["question", "answer", "similarity", "why"], rows)
  trap = table(
    ["", "value"],
    [
      ["similarity of 'cancel NC-1002?' and 'cancel NC-1003?'", f"{similarity:.3f}"],
      ["without guards, NC-1003 gets", wrong.answer or "—"],
      ["with the guards", "never stored, never served (an order id: personal)"],
    ],
  )
  lifecycle = table(
    ["", "result"],
    [
      ["prompt v1 entry, asked with prompt v2", stale],
      ["an entry past its TTL", expired],
      ["Tamil scope", tamil],
      ["staff scope", staff],
    ],
  )
  return [
    lookups,
    trap,
    lifecycle,
    table(["not stored"], [[line] for line in cache.stats.log]),
    callout(
      f"Paraphrases hit. 'Cancel NC-1002?' and 'cancel NC-1003?' are {similarity:.2f} similar, so an unguarded cache "
      "answered NC-1003 with NC-1002's answer. The guards stop it twice: personal questions are never stored, and the "
      "entity guard requires the same numbers, ids, places and product names. Versions and TTLs retire stale "
      "answers; scopes keep languages and audiences apart.",
      "decision",
      "A guarded semantic cache",
    ),
    example(
      "prod.l07.3",
      "Embed, find the nearest, reuse above a threshold",
      semantic_basics,
      "nimbus/cache/semantic.py",
      [lookups],
    ),
    example("prod.l07.4", "The danger: one detail", the_trap, "", [trap]),
    example("prod.l07.5", "TTLs and scopes", scoped, "", [lifecycle]),
    example("new.l01.3", "Lookup", semantic_basics, "SemanticCache.lookup", [lookups]),
    example(
      "new.l01.4",
      "What an entry stores",
      semantic_basics,
      "nimbus/cache/stores.py Entry",
      [code(json.dumps(list(cache.store.entries[0].payload())), "json")],
    ),
    example(
      "new.l01.6",
      "The entity guard",
      the_trap,
      "semantic.entities: ids, numbers, dates, places, menu items, cafés",
      [trap],
    ),
    example(
      "new.l01.10",
      "What never to cache",
      semantic_basics,
      "semantic.cacheable",
      [table(["not stored"], [[line] for line in cache.stats.log])],
    ),
    example("new.l01.11", "Invalidation: versions and TTLs", versions_and_ttl, "", [lifecycle]),
    example("new.l01.12", "Scopes", scoped, "", [lifecycle]),
  ]


# ── Lab 3: choosing the threshold ──────────────────────────────────────────────
@lab("prod.cache-threshold", "Choosing the similarity threshold from data", calls="No model calls")
def cache_threshold():
  """Ten paraphrase pairs that should hit and eight near-miss pairs that must not (data/eval/cache_pairs.json), at
  seven thresholds, without and with the entity guard."""
  from nimbus.cache.semantic import tune_threshold

  pairs = json.loads((PROJECT_DIR / "data" / "eval" / "cache_pairs.json").read_text())["pairs"]
  rows = tune_threshold(pairs, _embeddings())
  safe = [r for r in rows if r["false_hits_with_guard"] == 0]
  best = max(safe, key=lambda r: r["paraphrases_hit"]) if safe else None
  view = table(
    ["threshold", "paraphrases that hit", "near-misses that hit", "… with the entity guard"],
    [
      [
        f"{r['threshold']:.2f}",
        f"{r['paraphrases_hit']}/{r['paraphrases']}",
        f"{r['false_hits']}/{r['near_misses']}",
        f"{r['false_hits_with_guard']}/{r['near_misses']}",
      ]
      for r in rows
    ],
  )
  return [
    view,
    callout(
      "No threshold separates paraphrases from near-misses on its own with this embedding model: at "
      f"{rows[2]['threshold']:.2f}, {rows[2]['paraphrases_hit']} paraphrases hit and so did {rows[2]['false_hits']} "
      "near-misses. With the entity guard the near-misses stop hitting"
      + (
        f", so the threshold can come down to {best['threshold']:.2f} ({best['paraphrases_hit']}/{best['paraphrases']}"
        " paraphrases). Nimbus uses 0.75: a margin above the lowest safe value on 18 pairs."
        if best
        else "."
      ),
      "decision",
      "Threshold 0.75, with the guard",
    ),
    example("new.l01.5", "A threshold from labelled pairs", tune_threshold, "", [view]),
  ]


# ── Lab 4: stores ──────────────────────────────────────────────────────────────
def benchmark(store, vectors, query) -> dict:
  from nimbus.cache.stores import Entry

  started = time.perf_counter()
  for i, v in enumerate(vectors):
    store.add(Entry(question=f"q{i}", answer=f"a{i}", vector=v))
  added = (time.perf_counter() - started) / len(vectors) * 1000
  timings = []
  for _ in range(20):
    started = time.perf_counter()
    store.nearest(query, "public:en")
    timings.append((time.perf_counter() - started) * 1000)
  return {"add_ms": round(added, 3), "lookup_ms": round(sorted(timings)[10], 3)}


def compare_stores(n: int = 500) -> list[dict]:
  import tempfile
  from pathlib import Path

  import numpy as np

  from nimbus.cache.stores import MemoryStore, PgVectorStore, RedisVectorStore, SQLiteStore

  rng = np.random.default_rng(7)
  vectors = rng.normal(size=(n, 384)).astype(np.float32)
  query = vectors[42] + rng.normal(scale=0.05, size=384)
  rows = []
  folder = Path(tempfile.mkdtemp(prefix="nimbus-cache-"))
  stores = [
    ("memory", lambda: MemoryStore(), "no", "in code"),
    ("SQLite", lambda: SQLiteStore(folder / "cache.db"), "yes (a file)", "in code"),
  ]
  client, redis_used = _redis(), False
  try:
    client.ping()
    client.delete("lab-semcache")
    stores.append(("Redis Vector Sets", lambda: RedisVectorStore(client, "lab-semcache"), "yes (shared)", "per entry"))
    redis_used = True
  except redis_lib.RedisError:
    rows.append(
      {"variant": "Redis Vector Sets", "add_ms": None, "lookup_ms": None, "persists": "Redis unavailable", "ttl": ""}
    )
  conninfo = get_settings().database_url.replace("+psycopg", "")
  try:
    pg = PgVectorStore(conninfo, 384, "lab_semantic_cache")
    pg.clear()
    stores.append(("Postgres + pgvector", lambda: pg, "yes (database)", "expires_at column"))
  except Exception as error:  # noqa: BLE001 - no course database, or no pgvector extension
    rows.append(
      {
        "variant": "Postgres + pgvector",
        "add_ms": None,
        "lookup_ms": None,
        "persists": f"unavailable: {str(error).splitlines()[0][:60]}",
        "ttl": "",
      }
    )
  try:
    for name, make, persists, ttl in stores:
      rows.append({"variant": name, **benchmark(make(), vectors, query), "persists": persists, "ttl": ttl})
  finally:
    if redis_used:
      RedisVectorStore(client, "lab-semcache").clear()
    client.close()
  return rows


OTHER_STORES = [
  ["Valkey", "Redis-compatible open-source fork; same commands, check vector support in your version"],
  ["MongoDB Atlas", "vector search next to documents you already keep there"],
  ["Elasticsearch / OpenSearch", "kNN search plus full-text and filters, if you run them for search already"],
  ["Cassandra / Astra DB", "vector search at very large scale, multi-region"],
  ["Managed caches", "provider gateways and API caches that do semantic caching for you (less control)"],
]


@lab("prod.cache-backends", "Where a semantic cache lives: memory, SQLite, Redis, pgvector", calls="No model calls")
def cache_backends():
  """500 cached questions in each store: time to add one and to find the nearest, whether it survives a restart, and
  how entries expire. pgvector needs the course database (docker compose up -d)."""
  rows = compare_stores()
  card = scorecard(
    rows,
    [
      Metric("lookup_ms", "lookup (ms, p50)", "lower", "text"),
      Metric("add_ms", "add (ms each)", "lower", "text"),
      Metric("persists", "survives a restart", "none", "text"),
      Metric("ttl", "expiry", "none", "text"),
    ],
    title="500 entries, 384 dimensions",
  )
  measured = [r for r in rows if r["lookup_ms"] is not None]
  fastest = min(measured, key=lambda r: r["lookup_ms"])
  return [
    card,
    table(["other stores", "when"], OTHER_STORES),
    callout(
      f"Fastest lookup: {fastest['variant']} ({fastest['lookup_ms']} ms); every store answered in about a "
      "millisecond, against seconds for a model call. Memory is per process and lost on restart; SQLite persists "
      "on one machine; Redis is shared by every process with expiry per entry; "
      "pgvector keeps the cache in the database you already back up. Nimbus uses Redis: already running for jobs "
      "and limits, shared, and fast enough.",
      "decision",
      "Redis Vector Sets for the shared cache",
    ),
    example("new.l01.7", "Four stores, measured", compare_stores, "nimbus/cache/stores.py", [card]),
    example("new.l01.8", "Other stores", compare_stores, "", [table(["store", "when"], OTHER_STORES)]),
  ]


# ── Lab 5: what to cache, and caching in production ────────────────────────────
def embedding_cache(texts: list[str]):
  import tempfile

  from langchain_classic.embeddings import CacheBackedEmbeddings
  from langchain_classic.storage import LocalFileStore

  store = LocalFileStore(tempfile.mkdtemp(prefix="nimbus-embeddings-"))
  cached = CacheBackedEmbeddings.from_bytes_store(  # key: model namespace + a hash of the text
    _embeddings(), store, namespace="bge-small", key_encoder="sha256"
  )
  timings = []
  for _ in range(2):
    started = time.perf_counter()
    cached.embed_documents(texts)  # the second time, every vector comes from the store
    timings.append(time.perf_counter() - started)
  return timings


def provider_prompt_cache(model):
  # The provider caches a long, repeated PREFIX (system prompt, documents) and bills it cheaper; the answer is
  # still generated. A response cache skips the call entirely. Different mechanisms, often used together.
  prefix = "You are Nimbus Coffee's support assistant. Reference: " + (POLICY + " ") * 40
  details = []
  for question in ("How much is a cold brew?", "Is the Mocha vegan?"):
    reply = model.invoke([SystemMessage(prefix), HumanMessage(question)])
    usage = reply.response_metadata.get("token_usage", {})
    details.append(
      (question, usage.get("prompt_tokens"), (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    )
  return details


async def stampede(coalesce: bool, n: int = 10) -> int:
  from nimbus.cache.semantic import SemanticCache
  from nimbus.cache.stores import MemoryStore

  calls = 0

  async def compute(question):
    nonlocal calls
    calls += 1
    await asyncio.sleep(0.5)  # a slow model call
    return "A cold brew is ₹200."

  cache = SemanticCache(MemoryStore(), _embeddings(), threshold=0.75)
  if coalesce:  # identical questions arriving together share one call
    await asyncio.gather(*(cache.get_or_compute("How much is a cold brew?", compute) for _ in range(n)))
  else:  # every request misses until the first answer is stored
    await asyncio.gather(*(compute("How much is a cold brew?") for _ in range(n)))
  return calls


def warm_up(cache, top_questions: dict[str, str]) -> int:
  for question, answer in top_questions.items():  # the most frequent questions, answered before traffic arrives
    cache.save(question, answer)
  return len(cache.store)


@lab("prod.cache-production", "What to cache, and caching in production", uses_model=True, calls="2 calls to Mistral")
def cache_production():
  """Why cache (the numbers to watch), an embedding cache (CacheBackedEmbeddings), the provider's prompt caching
  against a response cache, ten identical requests at once with and without coalescing, and a warm-up."""
  from nimbus.cache.semantic import SemanticCache
  from nimbus.cache.stores import MemoryStore

  texts = [f"Nimbus help article {i}: refunds, deliveries and points." for i in range(64)]
  first, second = embedding_cache(texts)
  prompt_cache = provider_prompt_cache(get_chat_model())
  without, with_coalescing = asyncio.run(stampede(False)), asyncio.run(stampede(True))
  warmed = warm_up(
    SemanticCache(MemoryStore(), _embeddings(), threshold=0.75),
    {"How much is a cold brew?": "₹200.", "Is the Oat Latte vegan?": "Yes."},
  )
  why = table(
    ["measure", "why it matters"],
    [
      ["hit rate", "share of requests answered from the cache"],
      ["false-hit rate", "cached answers given to a different question: the one that hurts"],
      ["latency saved", "a hit answers in milliseconds instead of seconds"],
      ["tokens / $ saved", "every hit is a model call not paid for, and rate-limit budget kept"],
    ],
  )
  what = table(
    ["cache", "examples"],
    [
      ["model answers", "FAQ-style questions with one right answer for everyone"],
      ["retrieval results", "the chunks for a question, keyed with the index version"],
      ["embeddings", f"CacheBackedEmbeddings: 64 texts took {first:.2f} s, then {second:.3f} s"],
      ["reranker scores, slow tools", "anything deterministic and expensive"],
    ],
  )
  prompt = table(
    ["question", "prompt tokens", "of which cached by the provider"], [[q, str(p), str(c)] for q, p, c in prompt_cache]
  )
  coalesce = table(
    ["10 identical requests at once", "model calls"],
    [["no coalescing", str(without)], ["get_or_compute (coalesced)", str(with_coalescing)]],
  )
  return [
    why,
    what,
    prompt,
    coalesce,
    callout(
      f"The embedding cache made the second pass {first / max(second, 1e-6):.0f}× faster. Ten identical requests "
      f"cost {without} model calls without coalescing and {with_coalescing} with it. Provider prompt caching "
      + (
        f"billed {prompt_cache[1][2]} of {prompt_cache[1][1]} prompt tokens as cached on the second call"
        if prompt_cache[1][2]
        else "reported no cached tokens here (the provider decides when it applies)"
      )
      + ", but the answer was still generated: only a response cache skips the call.",
      "info",
      "Caching in production",
    ),
    example("new.l01.1", "Why cache, and what to measure", stampede, "", [why]),
    example("new.l01.9", "What to cache", embedding_cache, "", [what]),
    example("new.l01.13", "Provider prompt caching vs response caching", provider_prompt_cache, "", [prompt]),
    example(
      "new.l01.14",
      "Stampedes, warm-up, size and metrics",
      stampede,
      "SemanticCache.get_or_compute; warm_up",
      [coalesce, code(f"warmed entries: {warmed}", "text")],
    ),
  ]


# ── Production path: replaying support traffic ────────────────────────────────
def replay(variant: str, requests: list[dict], model) -> dict:
  from nimbus.cache.exact import normalize
  from nimbus.cache.semantic import SemanticCache
  from nimbus.cache.stores import MemoryStore

  embeddings = _embeddings()
  semantic = None
  if variant.startswith("semantic"):
    guarded = variant == "semantic, guarded (production)"
    semantic = SemanticCache(
      MemoryStore(), embeddings, get_settings().semantic_cache_threshold, entity_guard=guarded, rules=guarded
    )
  exact: dict[str, tuple[str, str]] = {}
  group_of: dict[str, str] = {}
  calls = hits = false_hits = tokens = 0
  latencies = []
  for request in requests:
    question, group = request["question"], request["group"]
    started = time.perf_counter()
    answer_group = None
    if variant == "no cache":
      cached = None
    elif variant.startswith("exact"):
      key = normalize(question) if "normalized" in variant else question
      cached = exact.get(key)
      answer_group = cached[1] if cached else None
    else:
      found = semantic.lookup(question)
      cached = (found.answer, group_of.get(found.matched)) if found.hit else None
      answer_group = cached[1] if cached else None
    if cached:
      hits += 1
      false_hits += answer_group != group
    else:
      reply = ask(model, question)
      calls += 1
      tokens += (reply.usage_metadata or {}).get("total_tokens", 0)
      if variant.startswith("exact"):
        exact[normalize(question) if "normalized" in variant else question] = (reply.text, group)
      elif semantic is not None and semantic.save(question, reply.text):
        group_of[question] = group
    latencies.append((time.perf_counter() - started) * 1000)
  ordered = sorted(latencies)
  return {
    "variant": variant,
    "hit_rate": round(hits / len(requests), 3),
    "false_hits": false_hits,
    "model_calls": calls,
    "tokens": tokens,
    "latency_p50_ms": round(ordered[len(ordered) // 2]),
  }


def evaluate_caches() -> list[dict]:
  requests = json.loads((PROJECT_DIR / "data" / "eval" / "cache_traffic.json").read_text())["requests"]
  model = get_chat_model().with_retry(stop_after_attempt=4, wait_exponential_jitter=True)  # Mistral sometimes 429s
  variants = ["no cache", "exact", "exact, normalized", "semantic, unguarded", "semantic, guarded (production)"]
  return [replay(v, requests, model) for v in variants]


@lab(
  "prod.cache-eval",
  "Evaluate: caches on replayed support traffic",
  uses_model=True,
  calls="~90 calls to Mistral: 28 requests × 5 caches, minus the hits",
)
def cache_eval():
  """28 support requests (paraphrases, repeats and near-misses: another drink, another city, another number,
  another order id) replayed through no cache, an exact cache, a normalized exact cache, an unguarded semantic cache
  and the production semantic cache: hit rate, false hits, model calls, tokens and latency."""
  rows = evaluate_caches()
  none, exact, normalized, unguarded, production = rows
  return [
    scorecard(
      rows,
      [
        Metric("hit_rate", "hit rate"),
        Metric("false_hits", "false hits", "lower", "int"),
        Metric("model_calls", "model calls", "lower", "int"),
        Metric("tokens", "tokens", "lower", "int"),
        Metric("latency_p50_ms", "latency p50", "lower", "ms"),
      ],
      title="28 requests",
    ),
    callout(
      f"The production cache answered {production['hit_rate']:.0%} of requests from the cache with "
      f"{production['false_hits']} false hits, against {unguarded['hit_rate']:.0%} and {unguarded['false_hits']} "
      f"false hits unguarded, {normalized['hit_rate']:.0%} for a normalized exact cache and {exact['hit_rate']:.0%} "
      f"for a plain one. Model calls: {production['model_calls']} instead of {none['model_calls']}.",
      "decision",
      "Ship the guarded semantic cache",
    ),
  ]
