"""RAG · Module 6 labs, part 1: retrievers and search settings, filters the question sets itself, time-aware
retrieval, hybrid search, and learned sparse / late-interaction representations."""

import json
import time
from datetime import date

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from academy.labs import cache
from academy.labs.rag import corpus
from academy.labs.runtime import Input, Metric, answer, callout, code, columns, example, lab, scorecard, table
from nimbus.config import get_settings
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import is_relevant

DATA = get_settings().data_dir
CORPUS = ("v3-tesseract", "recursive", "headers")


def _chunks() -> list[Document]:
  return list(corpus.chunks(300, 50, *CORPUS))


class _CosineStore(InMemoryVectorStore):
  """The in-memory store's scores are already cosine similarities: use them as relevance scores (0–1 for these
  normalised vectors), which score-threshold search needs."""

  def _select_relevance_score_fn(self):
    return lambda score: score


_STORE: InMemoryVectorStore | None = None


def _store() -> InMemoryVectorStore:
  """Production's chunks in LangChain's in-memory store (vectors reused from the corpus cache)."""
  global _STORE
  if _STORE is None:
    chunks = _chunks()
    vectors = corpus.chunk_vectors(300, 50, *CORPUS)
    store = _CosineStore(corpus.embeddings())
    for i, (c, v) in enumerate(zip(chunks, vectors, strict=True)):
      store.store[str(i)] = {
        "id": str(i),
        "vector": list(map(float, v)),
        "text": c.page_content,
        "metadata": c.metadata,
      }
    _STORE = store
  return _STORE


def _rank(docs: list[Document], evidence_of) -> int | None:
  return next((i for i, d in enumerate(docs, start=1) if evidence_of(d.page_content)), None)


# ── Lab 1: retrievers and search settings ───────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def a_retriever(store, k):
  retriever = store.as_retriever(search_kwargs={"k": k})  # a vector store wrapped as a Retriever
  docs = retriever.invoke("How many days of sick leave do I get?")
  return retriever, docs


def retrievers_are_runnables(retriever):
  from langchain_core.runnables import RunnableLambda

  many = retriever.batch(["Who approves remote work outside India?", "How much does an oat latte cost?"])
  chain = retriever | RunnableLambda(lambda docs: [d.metadata["source"] for d in docs])  # piped like any Runnable
  sources = chain.invoke("What is the hotel limit per night?")
  return many, sources


def filtered_retriever(store):
  pdf_only = store.as_retriever(
    search_kwargs={"k": 3, "filter": lambda doc: doc.metadata["file_type"] == "pdf"}  # in-memory: a function
  )
  docs = pdf_only.invoke("revenue growth")
  return docs


def score_threshold(store, threshold):
  retriever = store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": threshold, "k": 5}
  )
  on_topic = retriever.invoke("How many days of sick leave do I get?")
  off_topic = retriever.invoke("What's the capital of Peru?")  # better nothing than something irrelevant
  return on_topic, off_topic


def mmr_retriever(store, lambda_mult):
  retriever = store.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": lambda_mult}
  )  # lambda_mult: 1 = pure relevance, 0 = pure diversity
  docs = retriever.invoke("What does the employee handbook say?")
  return docs


def k_trade_off(store, ks):
  results = {}
  for k in ks:
    precisions, recalls = [], []
    for q in corpus.questions():
      docs = store.similarity_search(q.question, k=k)
      relevant = [is_relevant(d.page_content, q) for d in docs]
      total = sum(is_relevant(c.page_content, q) for c in _chunks()) or 1
      precisions.append(sum(relevant) / k)
      recalls.append(sum(relevant) / total)
    results[k] = (float(np.mean(precisions)), float(np.mean(recalls)))
  return results  # k → (precision@k, recall@k): recall rises, precision falls


@lab(
  "rag.retrievers",
  "Retrievers: k, score thresholds, MMR and filters behind one interface",
  inputs=[
    Input("threshold", "Score threshold", "number", default=0.7, min=0.3, max=0.95),
    Input("lambda_mult", "MMR lambda_mult", "number", default=0.5, min=0.0, max=1.0),
  ],
)
def retrievers_lab(threshold: float, lambda_mult: float):
  """Production's chunks behind the Retriever interface, with each search setting measured."""
  store = _store()
  retriever, docs = a_retriever(store, 3)
  many, sources = retrievers_are_runnables(retriever)
  pdf_docs = filtered_retriever(store)
  on_topic, off_topic = score_threshold(store, threshold)
  mmr = mmr_retriever(store, lambda_mult)
  plain = store.similarity_search("What does the employee handbook say?", k=4)
  trade = k_trade_off(store, (1, 3, 5, 10))
  distinct = lambda ds: len({d.metadata.get("section") or d.page_content[:40] for d in ds})  # noqa: E731
  rows = [{"variant": f"k={k}", "precision": round(p, 3), "recall": round(r, 3)} for k, (p, r) in trade.items()]
  return [
    scorecard(
      rows,
      [Metric("precision", "precision@k", "higher", "ratio"), Metric("recall", "recall@k", "higher", "ratio")],
      title="k: recall rises, precision falls (45 questions)",
    ),
    table(
      ["Setting", "Result"],
      [
        [f"threshold {threshold}: on-topic question", f"{len(on_topic)} chunks"],
        [f"threshold {threshold}: off-topic question", f"{len(off_topic)} chunks"],
        ["plain top 4: distinct sections", distinct(plain)],
        [f"MMR (lambda_mult={lambda_mult}): distinct sections", distinct(mmr)],
      ],
    ),
    callout(
      "One interface, `retriever.invoke(question)`, hides the search settings: how many results (k), a minimum "
      "score (return nothing rather than noise), MMR for variety, and filters. Choose k for the next stage: a "
      "reranker wants high recall (more candidates); a model reading the results wants high precision.",
      "decision",
      "Search settings",
    ),
  ] + [
    example(
      "lc.l44.1",
      "as_retriever(): k sets how many",
      a_retriever,
      "",
      [table(["source", "chunk"], [[d.metadata["source"], d.page_content[:80]] for d in docs])],
    ),
    example(
      "lc.l44.2",
      "A retriever is a Runnable: invoke, batch, pipe",
      retrievers_are_runnables,
      "",
      [
        table(["batch()", "results"], [[i, len(r)] for i, r in enumerate(many)]),
        table(["piped into a chain → sources"], [[str(sources)]]),
      ],
    ),
    example(
      "lc.l44.3",
      "Metadata filtering",
      filtered_retriever,
      "Only chunks matching the condition are searched.",
      [table(["source", "chunk"], [[d.metadata["source"], d.page_content[:80]] for d in pdf_docs])],
    ),
    example(
      "lc.l44.4",
      "search_type switches the method",
      mmr_retriever,
      '`search_type="mmr"` (or `"similarity_score_threshold"`) behind the same interface.',
      [table(["MMR result"], [[d.page_content[:80]] for d in mmr])],
    ),
    example(
      "rag.l31.1",
      "k: recall vs precision",
      k_trade_off,
      "",
      [table(["k", "precision@k", "recall@k"], [[r["variant"], r["precision"], r["recall"]] for r in rows])],
    ),
    example(
      "rag.l31.2",
      "A score threshold: nothing rather than noise",
      score_threshold,
      "",
      [table(["question", "chunks returned"], [["on topic", len(on_topic)], ["off topic", len(off_topic)]])],
    ),
    example(
      "rag.l31.3",
      "MMR: relevance vs variety",
      mmr_retriever,
      "Plain top 4 vs MMR top 4 for a broad question.",
      [table(["plain", "MMR"], [[a.page_content[:60], b.page_content[:60]] for a, b in zip(plain, mmr, strict=False)])],
    ),
    example(
      "rag.l31.4",
      "One interface for all of it",
      a_retriever,
      "Production's Retriever (nimbus/rag/retriever.py) has the same shape: a question in, ranked chunks out.",
      [table(["type"], [[type(retriever).__name__]])],
    ),
  ]


# ── Lab 2: metadata filters and self-querying ───────────────────────────────────
SELF_QUERY_QUESTION = "Which vegan things on the menu cost less than 210 rupees?"


def _menu_and_orders() -> list[Document]:
  import csv

  docs = []
  with (DATA / "menu.csv").open() as f:
    for row in csv.DictReader(f):
      vegan = row["vegan"] == "yes"
      docs.append(
        Document(
          f"{row['item']}: {row['category']} menu item, {row['price_rupees']} rupees, "
          f"{'vegan' if vegan else 'not vegan'}.",
          metadata={"doc_type": "menu", "item": row["item"], "price": int(row["price_rupees"]), "vegan": vegan},
        )
      )
  for o in json.loads((DATA / "orders.json").read_text()):
    docs.append(
      Document(
        f"Order {o['order_id']} by {o['customer']}: {', '.join(o['items'])}, {o['status']}.",
        metadata={"doc_type": "order", "status": o["status"], "item": o["order_id"]},
      )
    )
  docs += [
    Document(c.page_content, metadata={"doc_type": "policy", "item": c.metadata["source"]})
    for c in _chunks()
    if c.metadata["source"] == "handbook.md"
  ]
  return docs


def semantic_search_ignores_constraints(store):
  hits = store.similarity_search(SELF_QUERY_QUESTION, k=4)  # "vegan" and "210" are just words to match
  return hits


def a_metadata_filter(store):
  hits = store.similarity_search(
    "menu item",
    k=8,
    filter=lambda d: d.metadata["doc_type"] == "menu" and d.metadata["vegan"] and d.metadata["price"] < 210,
  )  # enforced exactly, whatever the wording
  return hits


class SearchPlan(BaseModel):
  query: str = Field(description="What to search for semantically, without the filter conditions")
  doc_type: str | None = Field(None, description="One of: menu, order, policy")
  vegan: bool | None = Field(None, description="Only for menu items")
  max_price: int | None = Field(None, description="Maximum price in rupees (exclusive), only for menu items")
  status: str | None = Field(None, description="Order status: delivered, shipped, processing or cancelled")


def self_query(model, question):
  plan = cache.structured(
    model, SearchPlan, f"Turn this question into a search plan for a coffee company's documents:\n{question}"
  )

  def matches(doc):
    m = doc.metadata
    return (
      (plan.doc_type is None or m.get("doc_type") == plan.doc_type)
      and (plan.vegan is None or m.get("vegan") == plan.vegan)
      and (plan.max_price is None or m.get("price", 0) < plan.max_price)
      and (plan.status is None or m.get("status") == plan.status)
    )

  return plan, matches  # the model writes the plan; our code turns it into a filter


def guard_rails(model):
  plan, _ = self_query(model, "Show me gluten-free drinks")  # no gluten field exists: it can't be filtered on
  return plan


@lab(
  "rag.self-query",
  "Metadata filters and self-querying: the question sets the filters",
  uses_model=True,
  calls="4 calls to Mistral (cached after the first run)",
)
def self_query_lab():
  """Constraints in a question (vegan, under 210 rupees, cancelled orders) as plain search, a filter, and a plan the
  model writes."""
  docs = _menu_and_orders()
  store = InMemoryVectorStore.from_documents(docs, corpus.embeddings())
  model = get_chat_model()
  semantic = semantic_search_ignores_constraints(store)
  filtered = a_metadata_filter(store)
  correct = {
    d.metadata["item"]
    for d in docs
    if d.metadata["doc_type"] == "menu" and d.metadata["vegan"] and d.metadata["price"] < 210
  }
  rows, plans = [], {}
  for question, right in (
    (SELF_QUERY_QUESTION, correct),
    ("Which orders were cancelled?", {d.metadata["item"] for d in docs if d.metadata.get("status") == "cancelled"}),
  ):
    plan, matches = self_query(model, question)
    planned = store.similarity_search(plan.query, k=8, filter=matches)
    plain = store.similarity_search(question, k=max(len(right), 1))
    plans[question] = plan
    for label, hits in (("semantic search only", plain), ("self-query (plan → filter)", planned)):
      found = {d.metadata["item"] for d in hits}
      rows.append(
        {
          "variant": f"{label} · {question[:30]}…",
          "right": len(found & right),
          "wrong": len(found - right),
          "expected": len(right),
        }
      )
  gluten = guard_rails(model)
  return [
    scorecard(
      rows,
      [
        Metric("right", "right results", "higher", "int"),
        Metric("wrong", "wrong results", "lower", "int"),
        Metric("expected", "expected", "none", "int"),
      ],
      title="Hard constraints: search vs a plan",
    ),
    table(["Question", "Plan the model wrote"], [[q, p.model_dump_json(exclude_none=True)] for q, p in plans.items()]),
    callout(
      "Semantic search treats 'vegan' and '210 rupees' as words to resemble, so non-vegan and pricier items come "
      "back. A metadata filter enforces them exactly. Self-query lets the model write the filter from the question; "
      "the schema is the guard rail: it can only set fields you defined (a 'gluten-free' question can't invent a "
      "gluten field). Filter values still come from a model, so validate them before they reach a database.",
      "decision",
      "Filters from the question",
    ),
  ] + [
    example(
      "rag.l32.1",
      "Semantic search ignores hard constraints",
      semantic_search_ignores_constraints,
      "",
      [table(["result"], [[d.page_content] for d in semantic])],
    ),
    example(
      "rag.l32.2",
      "A metadata filter enforces them",
      a_metadata_filter,
      "",
      [table(["result"], [[d.page_content] for d in filtered])],
    ),
    example(
      "rag.l32.3",
      "Self-query: the model writes (search text + filter)",
      self_query,
      "",
      [table(["Question", "plan"], [[q, p.model_dump_json(exclude_none=True)] for q, p in plans.items()])],
    ),
    example(
      "rag.l32.4",
      "Guard rails: only the fields you describe",
      guard_rails,
      "Asked about gluten, the plan has no field to put it in.",
      [code(gluten.model_dump_json(indent=2), "json")],
    ),
  ]


# ── Lab 3: time-aware retrieval ─────────────────────────────────────────────────
TODAY = date(2026, 3, 5)  # fixed, so the lab gives the same answers whenever it runs
NOTICES = [
  Document(
    "Notice: the Bengaluru roastery will be closed for maintenance on 22 and 23 September.",
    metadata={"published": "2025-09-01"},
  ),
  Document(
    "Notice: the Bengaluru roastery will be closed for maintenance on 16 and 17 March.",
    metadata={"published": "2026-03-02"},
  ),
  Document(
    "Notice: new espresso machines are being installed in all Bengaluru cafés this month.",
    metadata={"published": "2026-02-10"},
  ),
]


def both_versions_come_back(store):
  hits = store.similarity_search("How many days of paid annual leave do I get?", k=6)
  versions = [
    (d.metadata["source"], d.page_content.split("\n")[2] if d.page_content.count("\n") > 1 else d.page_content)
    for d in hits
    if "annual leave" in d.page_content
  ]
  return versions  # the archived handbook (20 days) and the current one (24 days)


def in_force_on(day):
  def keep(doc):
    return doc.metadata["valid_from"] <= day.isoformat() <= doc.metadata["valid_to"]

  return keep  # production: registry dates on every chunk + a SQL filter (nimbus/rag/ingestion/registry.py)


class AsOf(BaseModel):
  as_of: date | None = Field(description="The date the question is about, or null if it asks about now")


def as_of_from_the_question(model, question):
  when = cache.structured(
    model,
    AsOf,
    [("system", f"Today is {TODAY.isoformat()}. Find the date the question is about."), ("human", question)],
  ).as_of
  return when or TODAY


def recency_boost(similarity, published, half_life_days=90):
  age = (TODAY - date.fromisoformat(published)).days
  return similarity * (0.5 + 0.5 * 0.5 ** (age / half_life_days))  # never below half: relevance still counts


def answer_with_dates(model, question, docs, dated):
  if dated:  # tell the model today's date and when each source was written
    context = "\n".join(f"[{d.metadata['published']}] {d.page_content}" for d in docs)
    system = (
      f"Today is {TODAY:%d %B %Y}. Answer from the context in one sentence. Dates in [] are when each item was written."
    )
  else:
    context = "\n".join(d.page_content for d in docs)
    system = "Answer from the context in one sentence."
  return cache.text(model, [("system", system), ("human", f"Context:\n{context}\n\nQuestion: {question}")])


def _with_registry(docs: list[Document]) -> list[Document]:
  from nimbus.rag.ingestion.registry import apply_registry, load_registry

  copies = [Document(d.page_content, metadata=dict(d.metadata)) for d in docs]
  return apply_registry(copies, load_registry(DATA))


@lab(
  "rag.time-aware",
  "Time-aware retrieval: the right answer depends on when",
  uses_model=True,
  calls="up to 7 calls to Mistral (cached after the first run)",
)
def time_aware():
  """The archived and current handbooks, 'as of' questions, and notices with a recency boost."""
  model = get_chat_model()
  docs = _with_registry([c for c in _chunks() if c.metadata["source"] in ("handbook.md", "handbook_2024_archived.md")])
  store = InMemoryVectorStore.from_documents(docs, corpus.embeddings())
  versions = both_versions_come_back(store)
  question = "How many days of paid annual leave do I get?"
  plain = store.similarity_search(question, k=2)
  current = store.similarity_search(question, k=2, filter=in_force_on(date(2026, 10, 1)))
  rows = []
  as_of_questions = {  # question → the word that marks the line answering it
    "How many days of annual leave did we get in mid-2024?": "annual leave",
    "Which days can head office staff work remotely?": "work remotely",
    "Which days could I work from home back in 2024?": "work remotely",
  }
  for q, keyword in as_of_questions.items():
    when = as_of_from_the_question(model, q)
    top = store.similarity_search(q, k=1, filter=in_force_on(when))
    rows.append(
      [
        q,
        when.isoformat(),
        top[0].metadata["source"] if top else "—",
        next((x for x in top[0].page_content.splitlines() if keyword in x), None)
        or f"(a miss: the top chunk doesn't say; it starts '{top[0].page_content.splitlines()[1][:60]}')"
        if top
        else "",
      ]
    )
  notices = InMemoryVectorStore.from_documents(NOTICES, corpus.embeddings())
  nq = "When will the roastery be closed for maintenance?"
  scored = notices.similarity_search_with_score(nq, k=3)
  by_similarity = sorted(scored, key=lambda r: -r[1])
  by_boost = sorted(scored, key=lambda r: -recency_boost(r[1], r[0].metadata["published"]))
  boosted = [recency_boost(sc, d.metadata["published"]) for d, sc in by_boost]
  top2 = [d for d, _ in by_boost[:2]]
  without = answer_with_dates(model, nq, top2, dated=False)
  with_dates = answer_with_dates(model, nq, top2, dated=True)
  return [
    table(["source", "line"], versions, title="Plain search: both handbooks answer 'annual leave'"),
    table(
      ["Filter", "Sources returned"],
      [
        ["none", ", ".join(d.metadata["source"] for d in plain)],
        ["in force on 2026-10-01", ", ".join(d.metadata["source"] for d in current)],
      ],
    ),
    table(["Question", "as of", "source", "found"], rows, title="'As of' questions: the date comes from the question"),
    table(
      ["", "similarity", "boosted", "published", "notice"],
      [
        [
          i + 1,
          round(s, 3),
          round(recency_boost(s, d.metadata["published"]), 3),
          d.metadata["published"],
          d.page_content[8:70],
        ]
        for i, (d, s) in enumerate(by_boost)
      ],
      title="Recency boost (half-life 90 days)",
    ),
    columns(
      [answer(without, "without dates in the prompt"), answer(with_dates, "with today's date and each notice's date")]
    ),
    callout(
      "Old and new versions of a policy are equally similar to the question, so search returns both and the model "
      "may quote the outdated one. Record validity dates with every chunk and filter to what is in force (or to the "
      "date the question asks about). For news-like content without an expiry, boost recent items smoothly. "
      "Production now does the first: the document registry stamps valid_from/valid_to on every chunk and the "
      "retriever filters to today unless asked otherwise.",
      "decision",
      "Time-aware",
    ),
  ] + [
    example(
      "rag.l33.1",
      "Old and new versions both come back",
      both_versions_come_back,
      "",
      [table(["source", "line"], versions)],
    ),
    example(
      "rag.l33.2",
      "Validity dates + a filter: only what's in force",
      in_force_on,
      "The registry (data/nimbus/.documents.json) gives the archived handbook valid_to 2024-12-31.",
      [
        table(
          ["filter", "sources"],
          [
            ["none", [d.metadata["source"] for d in plain]],
            ["in force today", [d.metadata["source"] for d in current]],
          ],
        )
      ],
    ),
    example(
      "rag.l33.3",
      "'As of' questions: the model extracts the date",
      as_of_from_the_question,
      "",
      [table(["question", "as of", "source"], [r[:3] for r in rows])],
    ),
    example(
      "rag.l33.4",
      "A recency boost with a half-life",
      recency_boost,
      f"Similarity alone: {by_similarity[0][1]:.3f} vs {by_similarity[1][1]:.3f}, nearly a coin toss between the "
      f"two roastery notices. With the boost: {boosted[0]:.3f} vs {boosted[1]:.3f}: the recent one wins clearly.",
      [
        table(
          ["by similarity", "by boosted score"],
          [
            [a[0].metadata["published"], b[0].metadata["published"]]
            for a, b in zip(by_similarity, by_boost, strict=True)
          ],
        )
      ],
    ),
    example(
      "rag.l33.5",
      "Dates in the prompt",
      answer_with_dates,
      "",
      [table(["prompt", "answer"], [["no dates", without], ["with dates", with_dates]])],
    ),
  ]


# ── Lab 4: hybrid search ────────────────────────────────────────────────────────
HYBRID_QUESTIONS = {  # each method alone fails on some
  "it-help@nimbus.example": "it-help@nimbus.example",
  "What is the status of order NC-1003?": "NC-1003",
  "Can I work from home?": "work remotely",
  "How much time off do I get?": "24 days of paid",
  "GSTIN 29ABCDE1234F1Z5": "29ABCDE1234F1Z5",
  "Which coffee suits espresso?": "best for espresso",
}


def keyword_retriever(chunks, k):
  from langchain_community.retrievers import BM25Retriever

  from academy.labs.rag.corpus import tokenize

  # The default tokenizer is str.split: "it-help@nimbus.example." would keep its full stop and never match
  bm25 = BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize)
  return bm25


def reciprocal_rank_fusion(rankings, k=60):
  scores = {}
  for ranking in rankings:  # only positions count, so BM25 and cosine scores never have to be comparable
    for position, doc in enumerate(ranking, start=1):
      key = doc.page_content
      scores[key] = scores.get(key, (0.0, doc))
      scores[key] = (scores[key][0] + 1 / (k + position), doc)
  return [doc for _, doc in sorted(scores.values(), key=lambda pair: -pair[0])]


def ensemble_retriever(bm25, dense, weights):
  from langchain_classic.retrievers import EnsembleRetriever

  ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=weights)  # weighted RRF, built in
  return ensemble


@lab("rag.hybrid", "Hybrid search: keywords + embeddings, fused by rank")
def hybrid_lab():
  """Six questions where one method or the other fails, through BM25, dense, RRF and the EnsembleRetriever; then
  all 45 questions."""
  chunks = _chunks()
  store = _store()
  dense = store.as_retriever(search_kwargs={"k": 10})
  bm25 = keyword_retriever(chunks, 10)
  ensemble = ensemble_retriever(bm25, dense, [0.5, 0.5])
  from langchain_community.retrievers import BM25Retriever

  default_bm25 = BM25Retriever.from_documents(chunks, k=10)  # for comparison: the default tokenizer
  per_question = []
  for q, phrase in HYBRID_QUESTIONS.items():
    has = lambda text, p=phrase: p.casefold() in " ".join(text.split()).casefold()  # noqa: E731
    b, d = bm25.invoke(q), dense.invoke(q)
    per_question.append(
      [
        q,
        _rank(b, has) or "✗",
        _rank(d, has) or "✗",
        _rank(reciprocal_rank_fusion([b, d]), has) or "✗",
        _rank(ensemble.invoke(q), has) or "✗",
      ]
    )
  rows = []
  for name, search in (
    ("BM25, default str.split tokens", default_bm25.invoke),
    ("BM25 (keywords)", bm25.invoke),
    ("dense (embeddings)", dense.invoke),
    ("RRF (both)", lambda q: reciprocal_rank_fusion([bm25.invoke(q), dense.invoke(q)])),
  ):
    hits = [_rank(search(q.question)[:5], lambda t, q=q: is_relevant(t, q)) for q in corpus.questions()]
    rows.append(
      {
        "variant": name,
        "hit5": round(sum(h is not None for h in hits) / len(hits), 3),
        "mrr": round(sum(1 / h for h in hits if h) / len(hits), 3),
      }
    )
  return [
    table(
      ["Question", "BM25", "dense", "RRF", "Ensemble"],
      per_question,
      title="Rank of the right chunk (✗ = not in the top 10)",
    ),
    scorecard(
      rows,
      [Metric("hit5", "hit@5 (45)", "higher", "ratio"), Metric("mrr", "MRR", "higher", "ratio")],
      title="All 45 questions",
    ),
    callout(
      "Keywords find exact strings (an email, an order id, a GSTIN) that embeddings blur; embeddings find "
      "paraphrases ('work from home' → 'work remotely') that keywords miss. Tokenization decides how good keyword "
      f"search is: BM25 with the default str.split scored hit@5 {rows[0]['hit5']:.2f}, with a real tokenizer "
      f"{rows[1]['hit5']:.2f}. Reciprocal Rank Fusion merges rankings by position only, so incomparable scores never "
      f"mix (fused: {rows[3]['hit5']:.2f}, dense alone {rows[2]['hit5']:.2f}); its limit is that it can't tell a "
      "strong first place from a weak one. Production fuses Postgres full-text search with vectors, then reranks; "
      "the evaluation lab measures that.",
      "decision",
      "Hybrid",
    ),
  ] + [
    example(
      "rag.l34.1",
      "BM25Retriever",
      keyword_retriever,
      "",
      [table(["question", "rank"], [[r[0], r[1]] for r in per_question])],
    ),
    example(
      "rag.l34.2",
      "Each method alone fails on some questions",
      keyword_retriever,
      "",
      [table(["Question", "BM25", "dense"], [r[:3] for r in per_question])],
    ),
    example(
      "rag.l34.3",
      "Reciprocal Rank Fusion",
      reciprocal_rank_fusion,
      "1 / (60 + position) from each ranking, added up.",
      [table(["Question", "RRF"], [[r[0], r[3]] for r in per_question])],
    ),
    example(
      "rag.l34.4",
      "EnsembleRetriever: the built-in version",
      ensemble_retriever,
      "Weighted RRF over any retrievers.",
      [table(["Question", "Ensemble"], [[r[0], r[4]] for r in per_question])],
    ),
  ]


# ── Lab 5: SPLADE and ColBERT ───────────────────────────────────────────────────
SPLADE = "prithivida/Splade_PP_en_v1"
COLBERT = "answerdotai/answerai-colbert-small-v1"


def a_splade_vector(splade, question):
  import numpy as np

  vector = next(splade.query_embed(question))
  top = np.argsort(-vector.values)[:12]
  words = [(splade.model.tokenizer.id_to_token(int(vector.indices[i])), round(float(vector.values[i]), 2)) for i in top]
  return words  # readable word weights, including related words that aren't in the question


def dense_sparse_hybrid_in_qdrant(chunks, dense, sparse):
  import uuid

  from langchain_qdrant import QdrantVectorStore, RetrievalMode

  ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, c.page_content)) for c in chunks]  # Qdrant wants UUID ids
  stores = {
    mode.value: QdrantVectorStore.from_documents(
      chunks,
      embedding=dense,
      sparse_embedding=sparse,
      ids=ids,
      location=":memory:",
      collection_name=mode.value,
      retrieval_mode=mode,
    )
    for mode in (RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID)
  }
  return stores


def maxsim(query_tokens, doc_tokens):
  # ColBERT: every query token finds its best-matching document token; the matches are added up
  return float((query_tokens @ doc_tokens.T).max(axis=1).sum())


def colbert_rerank(colbert, question, candidates):
  q = next(colbert.query_embed(question))
  scores = [maxsim(q, t) for t in colbert.embed([d.page_content for d in candidates])]
  return [d for _, d in sorted(zip(scores, candidates, strict=True), key=lambda p: -p[0])]


@lab("rag.sparse-colbert", "Learned sparse vectors (SPLADE) and late interaction (ColBERT)")
def sparse_colbert():
  """SPLADE word weights, dense vs sparse vs hybrid in Qdrant, ColBERT's MaxSim as a reranker, and what each stores."""
  from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
  from langchain_qdrant import FastEmbedSparse

  s = get_settings()
  splade = SparseTextEmbedding(SPLADE, cache_dir=str(s.model_cache_dir))
  colbert = LateInteractionTextEmbedding(COLBERT, cache_dir=str(s.model_cache_dir))
  chunks = _chunks()
  words = a_splade_vector(splade, "How much time off do I get?")
  started = time.perf_counter()
  stores = dense_sparse_hybrid_in_qdrant(
    chunks, corpus.embeddings(), FastEmbedSparse(model_name=SPLADE, cache_dir=str(s.model_cache_dir))
  )
  build = time.perf_counter() - started
  rows = []
  for name, store in stores.items():
    hits = [
      _rank(store.similarity_search(q.question, k=5), lambda t, q=q: is_relevant(t, q)) for q in corpus.questions()
    ]
    rows.append(
      {
        "variant": f"Qdrant {name}",
        "hit5": round(sum(h is not None for h in hits) / len(hits), 3),
        "mrr": round(sum(1 / h for h in hits if h) / len(hits), 3),
      }
    )
  hits = []
  for q in corpus.questions():
    candidates = stores["hybrid"].similarity_search(q.question, k=10)
    hits.append(_rank(colbert_rerank(colbert, q.question, candidates)[:5], lambda t, q=q: is_relevant(t, q)))
  rows.append(
    {
      "variant": "hybrid + ColBERT rerank",
      "hit5": round(sum(h is not None for h in hits) / len(hits), 3),
      "mrr": round(sum(1 / h for h in hits if h) / len(hits), 3),
    }
  )
  sample = [c.page_content for c in chunks[:40]]
  sparse_nonzero = float(np.mean([len(v.indices) for v in splade.embed(sample)]))
  colbert_tokens = float(np.mean([v.shape[0] for v in colbert.embed(sample)]))
  colbert_dims = next(colbert.embed(["x"])).shape[1]
  q = next(colbert.query_embed("How much time off do I get?"))
  candidates = [
    "Every employee gets 24 days of paid annual leave per year.",
    "Sick leave is 12 days per year.",
    "Cafés open at 7:00 and close at 21:00.",
  ]
  maxsims = [(c, round(maxsim(q, next(colbert.embed([c]))), 2)) for c in candidates]
  storage = [
    ["dense (bge-small)", "384 floats"],
    ["sparse (SPLADE)", f"~{sparse_nonzero:.0f} (word id, weight) pairs"],
    [
      "ColBERT",
      f"~{colbert_tokens:.0f} tokens × {colbert_dims} floats ≈ {colbert_tokens * colbert_dims / 384:.0f}× dense",
    ],
  ]
  return [
    scorecard(
      rows,
      [Metric("hit5", "hit@5 (45)", "higher", "ratio"), Metric("mrr", "MRR", "higher", "ratio")],
      title=f"Production chunks in Qdrant (in memory, built in {build:.1f} s)",
    ),
    table(["word", "weight"], words, title="SPLADE vector for 'How much time off do I get?'"),
    table(["Representation", "Stored per chunk"], storage),
    callout(
      "SPLADE is a sparse vector of word weights: readable, exact-term aware, and it adds related words the question "
      "didn't use. ColBERT keeps one small vector per token and scores by best matches (MaxSim): precise, but it "
      "stores dozens of vectors per chunk, so it's usually a reranker over a short candidate list. Compare with "
      "production's choice (Postgres hybrid + a cross-encoder) in the evaluation lab.",
      "decision",
      "Beyond one vector",
    ),
  ] + [
    example(
      "rag.l35.1",
      "SPLADE: a sparse vector of word weights",
      a_splade_vector,
      "Related words appear even though the question never used them.",
      [table(["word", "weight"], words)],
    ),
    example(
      "rag.l35.2",
      "Dense vs sparse vs hybrid in Qdrant",
      dense_sparse_hybrid_in_qdrant,
      "Three in-memory collections through LangChain's QdrantVectorStore.",
      [table(["mode", "hit@5", "MRR"], [[r["variant"], r["hit5"], r["mrr"]] for r in rows[:3]])],
    ),
    example(
      "rag.l35.3", "ColBERT: one vector per token, MaxSim", maxsim, "", [table(["candidate", "MaxSim"], maxsims)]
    ),
    example(
      "rag.l35.4",
      "ColBERT as a reranker on hybrid search",
      colbert_rerank,
      "",
      [table(["", "hit@5", "MRR"], [[r["variant"], r["hit5"], r["mrr"]] for r in rows[2:]])],
    ),
    example(
      "rag.l35.5",
      "The price: storage per chunk",
      a_splade_vector,
      "",
      [table(["Representation", "per chunk"], storage)],
    ),
  ]
