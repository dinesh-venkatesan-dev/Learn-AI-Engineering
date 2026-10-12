"""RAG · Module 5 labs: embeddings (what they are, which model, which language), vector stores (Chroma, FAISS,
pgvector), keeping an index in sync, and ingesting at scale. Then: embedding models and query translation measured
on all 45 questions plus a multilingual set."""

import asyncio
import hashlib
import json
import shutil
import time
from collections.abc import Iterator
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

from academy.labs.rag import corpus
from academy.labs.runtime import (
  Input,
  Metric,
  answer,
  bars,
  callout,
  code,
  columns,
  example,
  lab,
  scorecard,
  table,
)
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import evaluate
from nimbus.rag.ingestion.embeddings import LocalEmbeddings
from nimbus.rag.retriever import needs_translation, translate

DATA = get_settings().data_dir
CACHE = PROJECT_DIR / ".cache" / "labs" / "m05"
MULTILINGUAL = json.loads((PROJECT_DIR / "data" / "eval" / "multilingual.json").read_text())
CORPUS = ("v3-tesseract", "recursive", "headers")  # production's loaders and chunking (Modules 3 and 4)


def _chunks() -> tuple[Document, ...]:
  return corpus.chunks(300, 50, *CORPUS)


@lru_cache
def _local(model_name: str) -> LocalEmbeddings:
  s = get_settings()
  return LocalEmbeddings(model_name, s.model_cache_dir, s.embedding_batch_size)


def _mistral_vectors(texts: tuple[str, ...]) -> tuple[np.ndarray, float]:
  """mistral-embed vectors and the seconds the API took, cached on disk so re-running doesn't spend requests."""
  from langchain_mistralai import MistralAIEmbeddings

  import nimbus.llm.models  # noqa: F401  (loads MISTRAL_API_KEY)

  path = CACHE / f"mistral-{hashlib.sha256(chr(0).join(texts).encode()).hexdigest()[:20]}.npz"
  if path.exists():
    saved = np.load(path)
    return saved["vectors"], float(saved["seconds"])
  started = time.perf_counter()
  vectors = np.array(MistralAIEmbeddings(model="mistral-embed").embed_documents(list(texts)))
  seconds = time.perf_counter() - started
  CACHE.mkdir(parents=True, exist_ok=True)
  np.savez(path, vectors=vectors, seconds=seconds)
  return vectors, seconds


def _mistral_query_latency() -> float:
  """Seconds for one embed_query request to Mistral, measured once and cached."""
  from langchain_mistralai import MistralAIEmbeddings

  path = CACHE / "mistral-query-latency.json"
  if path.exists():
    return json.loads(path.read_text())["seconds"]
  started = time.perf_counter()
  MistralAIEmbeddings(model="mistral-embed").embed_query("How many days of annual leave do I get?")
  seconds = time.perf_counter() - started
  CACHE.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({"seconds": seconds}))
  return seconds


def _normalise(v: np.ndarray) -> np.ndarray:
  return v / np.linalg.norm(v, axis=-1, keepdims=True)


MODELS = {
  "bge-small (local, 384)": "BAAI/bge-small-en-v1.5",
  "bge-base (local, 768)": "BAAI/bge-base-en-v1.5",
  "multilingual MiniLM (local, 384)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "mistral-embed (API, 1024)": "mistral-embed",
}


def _vectors(model: str, texts: list[str], query: bool = False) -> tuple[np.ndarray, float]:
  started = time.perf_counter()
  if model == "mistral-embed":
    v, seconds = _mistral_vectors(tuple(texts))  # batched here to spare requests
    if query:  # in production each question is its own request: report that latency per question
      seconds = _mistral_query_latency() * len(texts)
    return _normalise(v), seconds
  if query:
    v = np.array([_local(model).embed_query(t) for t in texts])
  else:
    v = np.array(_local(model).embed_documents(texts))
  return _normalise(v), time.perf_counter() - started


# ── Lab 1: embeddings and choosing a model ──────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def embed_one_query(embeddings):
  vector = embeddings.embed_query("How many days of annual leave do I get?")  # one text → one vector
  return vector


def embed_many_documents(embeddings):
  texts = [
    "Every employee gets 24 days of paid annual leave per year.",
    "Hotels are reimbursed up to 6,000 rupees per night.",
    "Points expire 12 months after they are earned.",
  ]
  vectors = embeddings.embed_documents(texts)  # many texts → many vectors, in one call
  return texts, vectors


def cosine_similarity(query_vector, document_vectors):
  import numpy as np

  q = np.array(query_vector)
  similarities = [float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))) for d in document_vectors]
  return similarities  # 1 = same direction (same meaning), ~0 = unrelated


def query_vs_document_embedding(embeddings):
  import numpy as np

  text = "How many days of annual leave do I get?"
  as_query = np.array(embeddings.embed_query(text))  # bge adds an instruction: "Represent this sentence for searching…"
  as_document = np.array(embeddings.embed_documents([text])[0])
  same = float(as_query @ as_document)
  return same  # < 1: the same text is embedded differently as a question and as a passage


def compare_models(models, questions, chunks):
  rows = {}
  for name, model in models.items():
    doc_vectors, index_seconds = _vectors(model, [c.page_content for c in chunks])
    q_vectors, query_seconds = _vectors(model, [q.question for q in questions], query=True)
    by_question = {q.question: v for q, v in zip(questions, q_vectors, strict=True)}

    def retrieve(question, k, doc_vectors=doc_vectors, by_question=by_question):
      return [chunks[i] for i in np.argsort(-(doc_vectors @ by_question[question]))[:k]]

    report = evaluate(retrieve, questions, [c.page_content for c in chunks], k=5)
    rows[name] = (doc_vectors.shape[1], index_seconds, query_seconds / len(questions), report)
  return rows


@lab(
  "rag.embeddings",
  "Embeddings: vectors that capture meaning, and choosing a model",
  uses_model=True,
  calls="2 requests to Mistral's embedding API the first time (chunks, questions), then cached",
)
def embeddings_lab():
  """What embed_query and embed_documents return, cosine similarity, and four models measured on all 45 questions."""
  bge = _local(MODELS["bge-small (local, 384)"])
  vector = embed_one_query(bge)
  texts, vectors = embed_many_documents(bge)
  similarities = cosine_similarity(vector, vectors)
  same = query_vs_document_embedding(bge)
  chunks = list(_chunks())
  results = compare_models(MODELS, list(corpus.questions()), chunks)
  rows = []
  for name, (dims, _, query_s, report) in results.items():
    rows.append(
      {
        "variant": name,
        "dims": dims,
        "index_mb": round(dims * 4 * 1_000_000 / 1e6),
        "query_ms": round(query_s * 1000, 1),
        "hit1": report.at["hit"][1],
        "hit5": report.at["hit"][5],
        "mrr": round(report.mrr, 3),
        "ndcg": report.at["ndcg"][5],
      }
    )
  return [
    scorecard(
      rows,
      [
        Metric("dims", "dimensions", "none", "int"),
        Metric("index_mb", "MB per million chunks", "lower", "int", "dims × 4 bytes × 1,000,000 (vectors only)"),
        Metric(
          "query_ms", "ms per question", "lower", "ms", "Local: on this CPU. API: one real request, measured once"
        ),
        Metric("hit1", "hit@1", "higher", "ratio"),
        Metric("hit5", "hit@5", "higher", "ratio"),
        Metric("mrr", "MRR", "higher", "ratio"),
        Metric("ndcg", "nDCG@5", "higher", "ratio"),
      ],
      title=f"Four embedding models, production chunks ({len(chunks)}), 45 questions",
    ),
    callout(
      "Bigger isn't automatically better: bge-base has twice the dimensions of bge-small and didn't beat it here. "
      "The API model was best on quality, at a network request per question (and the provider's rate limit, and "
      "your text leaving the machine). Measure on your own questions: public leaderboards rank general text, "
      "not your documents.",
      "decision",
      "Which embedding model",
    ),
  ] + [
    example(
      "lc.l42.1",
      "embed_query(): one text → one vector",
      embed_one_query,
      "A list of floats; the same model always gives the same length.",
      [table(["len(vector)", "first 6 numbers"], [[len(vector), str([round(x, 4) for x in vector[:6]])]])],
    ),
    example(
      "lc.l42.2",
      "embed_documents(): many texts in one call",
      embed_many_documents,
      "One vector per text, in order. Batching is far faster than one call per text.",
      [
        table(
          ["text", "vector (first 4)"],
          [[t, str([round(x, 3) for x in v[:4]])] for t, v in zip(texts, vectors, strict=True)],
        )
      ],
    ),
    example(
      "lc.l42.3",
      "Cosine similarity: same meaning, close vectors",
      cosine_similarity,
      "The question about annual leave shares few words with the leave sentence, yet its vector is closest.",
      [table(["Document", "cosine"], [[t, round(s, 3)] for t, s in zip(texts, similarities, strict=True)])],
    ),
    example(
      "rag.l24.1",
      "Two very different models",
      compare_models,
      "Local models run on this machine; mistral-embed is an API (it replaces the old course's Gemini model: "
      "Mistral's free plan includes it).",
      [table(["Model", "dims"], [[r["variant"], r["dims"]] for r in rows])],
    ),
    example(
      "rag.l24.2",
      "Speed and storage",
      compare_models,
      "Vector size × number of chunks is your index size; a local model answers in milliseconds, an API in hundreds.",
      [
        table(
          ["Model", "MB per million chunks", "ms per question"],
          [[r["variant"], r["index_mb"], r["query_ms"]] for r in rows],
        )
      ],
    ),
    example(
      "rag.l24.3",
      "Quality on your own test set",
      compare_models,
      "The 45 evaluation questions, same chunks, only the model changes.",
      [table(["Model", "hit@5", "MRR"], [[r["variant"], r["hit5"], r["mrr"]] for r in rows])],
    ),
    example(
      "rag.l24.4",
      "Query vs document embeddings",
      query_vs_document_embedding,
      "bge models embed questions with an instruction prefix, so the same text gets two different vectors. Always "
      "use embed_query for questions.",
      [table(["cosine(query vector, document vector) of the same text"], [[round(same, 4)]])],
    ),
  ]


# ── Lab 2: multilingual retrieval ───────────────────────────────────────────────
def same_meaning_other_languages(model):
  import numpy as np

  sentences = {
    "English": "How many days of annual leave do I get?",
    "Hindi": "मुझे साल में कितनी छुट्टी मिलती है?",
    "Spanish": "¿Cuántos días de vacaciones tengo al año?",
    "Unrelated English": "Is the banana bread vegan?",
  }
  vectors = dict(zip(sentences, np.array(model.embed_documents(list(sentences.values()))), strict=True))
  similarity = {name: float(vectors["English"] @ v) for name, v in vectors.items() if name != "English"}
  return similarity


def translate_then_search(question, model):
  from nimbus.rag.retriever import translate

  english = translate(question, model)  # one chat call (cached); the index stays English
  return english


def answer_in_their_language(model, question, context):
  reply = model.invoke(
    [
      ("system", "Answer only from the context. Reply in the SAME language as the question, in one sentence."),
      ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
  )
  return reply.text


@lab(
  "rag.multilingual",
  "Multilingual retrieval: questions in Hindi, Tamil, Hinglish… documents in English",
  uses_model=True,
  calls="up to 11 calls to Mistral the first time (10 translations + 1 answer; translations cached)",
)
def multilingual():
  """Ten non-English questions against the English index, four ways."""
  chunks = list(_chunks())
  texts = [c.page_content for c in chunks]
  model = get_chat_model()
  ways = {}
  for name, model_name in (
    ("english-only bge-small", MODELS["bge-small (local, 384)"]),
    ("multilingual MiniLM", MODELS["multilingual MiniLM (local, 384)"]),
    ("mistral-embed (API)", "mistral-embed"),
  ):
    doc_vectors, _ = _vectors(model_name, texts)
    q_vectors, _ = _vectors(model_name, [m["question"] for m in MULTILINGUAL], query=True)
    ways[name] = [
      any(m["evidence"] in " ".join(chunks[i].page_content.split()) for i in np.argsort(-(doc_vectors @ q))[:3])
      for m, q in zip(MULTILINGUAL, q_vectors, strict=True)
    ]
  bge_docs, _ = _vectors(MODELS["bge-small (local, 384)"], texts)
  translated = {m["question"]: translate_then_search(m["question"], model) for m in MULTILINGUAL}
  t_vectors, _ = _vectors(MODELS["bge-small (local, 384)"], list(translated.values()), query=True)
  ways["translate, then bge-small"] = [
    any(m["evidence"] in " ".join(chunks[i].page_content.split()) for i in np.argsort(-(bge_docs @ q))[:3])
    for m, q in zip(MULTILINGUAL, t_vectors, strict=True)
  ]
  policy = [needs_translation(m["question"], "non_ascii") for m in MULTILINGUAL]
  ways["translate only non-ASCII (production)"] = [
    ways["translate, then bge-small"][i] if policy[i] else ways["english-only bge-small"][i]
    for i in range(len(MULTILINGUAL))
  ]
  hindi = MULTILINGUAL[0]["question"]
  q = _normalise(np.array(_local(MODELS["bge-small (local, 384)"]).embed_query(translated[hindi])))
  context = "\n\n".join(chunks[i].page_content for i in np.argsort(-(bge_docs @ q))[:3])
  reply = answer_in_their_language(model, hindi, context)
  english_sim = same_meaning_other_languages(_local(MODELS["bge-small (local, 384)"]))
  multi_sim = same_meaning_other_languages(_local(MODELS["multilingual MiniLM (local, 384)"]))
  costs = {
    "english-only bge-small": "free",
    "multilingual MiniLM": "free (re-embed the corpus once)",
    "mistral-embed (API)": "an API request per question",
    "translate, then bge-small": "a chat call per question",
    "translate only non-ASCII (production)": f"a chat call for {sum(policy)} of {len(policy)} questions",
  }
  rows = [
    {"variant": name, "found": f"{sum(h)}/{len(h)}", "share": round(sum(h) / len(h), 3), "cost": costs[name]}
    for name, h in ways.items()
  ]
  hinglish = [i for i, m in enumerate(MULTILINGUAL) if m["language"] == "Hinglish"]
  hinglish_found = sum(ways["english-only bge-small"][i] for i in hinglish)
  return [
    scorecard(
      rows,
      [
        Metric("share", "answer in the top 3", "higher", "ratio"),
        Metric("found", "found", "none", "text"),
        Metric("cost", "extra cost per question", "none", "text"),
      ],
      title="10 questions in 5 languages, English documents",
    ),
    table(
      ["Question", "Language", *ways],
      [
        [m["question"], m["language"], *["✓" if hits[i] else "✗" for hits in ways.values()]]
        for i, m in enumerate(MULTILINGUAL)
      ],
    ),
    answer(reply, title=f"Answering in the user's language: {hindi}"),
    callout(
      f"Without help, the English-only model found {rows[0]['found']}: a Hindi or Tamil question lands nowhere near "
      "the English answer. Two fixes: a multilingual embedding model (re-embed everything, and here it lost English "
      "quality: see the evaluation), or translate the question and keep the English index. Production translates "
      "questions with non-ASCII letters (one cached chat call). Hinglish is written in ASCII so it isn't "
      f"translated, but its English words ('hotel', 'sick leave', 'laptop') carried it: {hinglish_found} of "
      f"{len(hinglish)} found without translation. Answer in the language the user asked in.",
      "decision",
      "Multilingual",
    ),
  ] + [
    example(
      "rag.l25.1",
      "An English-only model: no clear gap",
      same_meaning_other_languages,
      "The Hindi and Spanish versions should score far above the unrelated English sentence.",
      [table(["Sentence", "similarity to the English question"], [[k, round(v, 3)] for k, v in english_sim.items()])],
    ),
    example(
      "rag.l25.2",
      "A multilingual model: same meaning, nearby vectors",
      same_meaning_other_languages,
      "The translations score well above the unrelated sentence.",
      [table(["Sentence", "similarity"], [[k, round(v, 3)] for k, v in multi_sim.items()])],
    ),
    example(
      "rag.l25.3",
      "Measured on our corpus",
      same_meaning_other_languages,
      "Ten questions in Hindi, Tamil, Hinglish, Spanish, French and German.",
      [table(["Way", "found"], [[r["variant"], r["found"]] for r in rows])],
    ),
    example(
      "rag.l25.4",
      "Translate the question, then search as usual",
      translate_then_search,
      "One chat call per question (cached); the index and the embedding model stay as they are.",
      [table(["Question", "In English"], [[q, e] for q, e in translated.items()])],
    ),
    example(
      "rag.l25.5",
      "Answer in the user's language",
      answer_in_their_language,
      "Search in English, answer in Hindi.",
      [answer(reply, title=hindi)],
    ),
  ]


# ── Lab 3: Chroma ───────────────────────────────────────────────────────────────
MENU_DOCS = [
  Document(
    "Filter Coffee: hot, 120 rupees, not vegan.",
    metadata={"item": "Filter Coffee", "category": "hot", "price": 120, "vegan": False},
    id="menu-1",
  ),
  Document(
    "Oat Latte: hot, 220 rupees, vegan.",
    metadata={"item": "Oat Latte", "category": "hot", "price": 220, "vegan": True},
    id="menu-2",
  ),
  Document(
    "Cold Brew: cold, 200 rupees, vegan.",
    metadata={"item": "Cold Brew", "category": "cold", "price": 200, "vegan": True},
    id="menu-3",
  ),
  Document(
    "Mango Frappe: cold, 240 rupees, not vegan.",
    metadata={"item": "Mango Frappe", "category": "cold", "price": 240, "vegan": False},
    id="menu-4",
  ),
  Document(
    "Banana Bread: food, 150 rupees, not vegan.",
    metadata={"item": "Banana Bread", "category": "food", "price": 150, "vegan": False},
    id="menu-5",
  ),
  Document(
    "Vegan Brownie: food, 170 rupees, vegan.",
    metadata={"item": "Vegan Brownie", "category": "food", "price": 170, "vegan": True},
    id="menu-6",
  ),
]


def create_a_persistent_collection(folder, embeddings, docs):
  from langchain_chroma import Chroma

  store = Chroma(collection_name="nimbus", embedding_function=embeddings, persist_directory=folder)  # saved to disk
  store.add_documents(docs, ids=[d.id for d in docs])  # embedded once, stored with their ids
  return store


def reopen_it(folder, embeddings):
  from langchain_chroma import Chroma

  store = Chroma(collection_name="nimbus", embedding_function=embeddings, persist_directory=folder)
  count = store._collection.count()  # the vectors are on disk: nothing is re-embedded
  return store, count


def search_with_scores(store):
  distances = store.similarity_search_with_score("something cold to drink", k=3)  # distance: lower = closer
  relevance = store.similarity_search_with_relevance_scores("something cold to drink", k=3)  # 0–1: higher = closer
  return distances, relevance


def metadata_filters(store):
  exact = store.similarity_search("a drink", k=6, filter={"category": "cold"})
  cheaper = store.similarity_search("a drink", k=6, filter={"price": {"$lt": 200}})
  both = store.similarity_search(
    "a treat", k=6, filter={"$and": [{"vegan": True}, {"category": {"$in": ["food", "cold"]}}]}
  )
  return exact, cheaper, both


def crud_without_searching(store):
  from langchain_core.documents import Document

  got = store.get(ids=["menu-2"])  # read by id
  store.update_document(
    "menu-2",
    Document(
      "Oat Latte: hot, 230 rupees, vegan.",
      metadata={"item": "Oat Latte", "category": "hot", "price": 230, "vegan": True},
    ),
  )
  store.delete(ids=["menu-4"])  # delete by id
  removed = store.get(where={"category": "food"})["ids"]
  store._collection.delete(where={"category": "food"})  # delete by filter
  left = store.get()["ids"]
  return got, removed, left


def vector_store_basics(embeddings, docs):
  from langchain_core.vectorstores import InMemoryVectorStore

  store = InMemoryVectorStore.from_documents(docs, embeddings)  # embed + index in one call
  hits = store.similarity_search("is there anything vegan to eat?", k=2)  # the k closest in meaning
  scored = store.similarity_search_with_score("is there anything vegan to eat?", k=2)  # with how close
  diverse = store.max_marginal_relevance_search("vegan", k=3, fetch_k=6)  # relevant AND different from each other
  store.delete(ids=[docs[0].id])
  return hits, scored, diverse, len(store.store)


@lab("rag.chroma", "Vector stores and Chroma: add, search, filter, persist, update")
def chroma_lab():
  """The café menu in LangChain's in-memory store and in a persistent Chroma collection."""
  embeddings = _local(MODELS["bge-small (local, 384)"])
  folder = CACHE / "chroma"
  shutil.rmtree(folder, ignore_errors=True)
  started = time.perf_counter()
  store = create_a_persistent_collection(str(folder), embeddings, MENU_DOCS)
  build_ms = (time.perf_counter() - started) * 1000
  started = time.perf_counter()
  reopened, count = reopen_it(str(folder), embeddings)
  reopen_ms = (time.perf_counter() - started) * 1000
  distances, relevance = search_with_scores(reopened)
  exact, cheaper, both = metadata_filters(reopened)
  got, removed, left = crud_without_searching(reopened)
  hits, scored, diverse, remaining = vector_store_basics(embeddings, MENU_DOCS)
  similar = embeddings.embed_documents(["Oat Latte: hot, 220 rupees, vegan.", "Oat Latte: hot, 220 rupees, vegan!"])
  near_duplicate = float(np.dot(_normalise(np.array(similar[0])), _normalise(np.array(similar[1]))))
  plain_top3 = [d.metadata["item"] for d in reopened.similarity_search("vegan", k=3)]
  for s in (store, reopened):
    s._client.clear_system_cache() if hasattr(s._client, "clear_system_cache") else None
  shutil.rmtree(folder, ignore_errors=True)
  rows = [
    {"variant": "create + embed 6 items", "ms": round(build_ms)},
    {"variant": "reopen from disk", "ms": round(reopen_ms)},
  ]
  return [
    scorecard(
      rows,
      [Metric("ms", "time", "lower", "ms")],
      title=f"A persistent collection ({count} items after reopening: nothing re-embedded)",
    ),
    table(
      ["Filter", "Items"],
      [
        ["category = cold", ", ".join(d.metadata["item"] for d in exact)],
        ["price < 200", ", ".join(d.metadata["item"] for d in cheaper)],
        ["vegan AND category in (food, cold)", ", ".join(d.metadata["item"] for d in both)],
      ],
      title="Metadata filters",
    ),
    columns(
      [
        table(
          ["item", "distance (↓)"],
          [[d.metadata["item"], round(s, 3)] for d, s in distances],
          title="similarity_search_with_score",
        ),
        table(
          ["item", "relevance (↑)"],
          [[d.metadata["item"], round(s, 3)] for d, s in relevance],
          title="…_with_relevance_scores",
        ),
      ]
    ),
    callout(
      "Chroma keeps vectors and metadata on disk, so a restart reopens the index instead of re-embedding it. "
      "Filters run inside the search, so a filtered query still returns k results that match. It suits a single "
      "machine; production Nimbus uses pgvector because the data already lives in Postgres (lab 5).",
      "decision",
      "Chroma",
    ),
  ] + [
    example(
      "lc.l43.1",
      "Building a store: from_documents / add_documents",
      vector_store_basics,
      "Embeds every Document and indexes it.",
      [table(["store", "items"], [["InMemoryVectorStore", len(MENU_DOCS)]])],
    ),
    example(
      "lc.l43.2",
      "similarity_search: the k closest",
      vector_store_basics,
      "",
      [table(["item"], [[d.metadata["item"]] for d in hits])],
    ),
    example(
      "lc.l43.3",
      "…_with_score: how close each one is",
      vector_store_basics,
      "The in-memory store returns cosine similarity (higher = closer).",
      [table(["item", "score"], [[d.metadata["item"], round(s, 3)] for d, s in scored])],
    ),
    example(
      "lc.l43.4",
      "MMR: relevant AND diverse",
      vector_store_basics,
      "Max marginal relevance fetches `fetch_k` candidates, then picks ones that are relevant but not similar to "
      "each other.",
      [
        table(
          ["plain top 3 (Chroma)", "MMR top 3"],
          [[a, d.metadata["item"]] for a, d in zip(plain_top3, diverse, strict=False)],
        ),
        table(["near-duplicate texts", "cosine"], [["'…vegan.' vs '…vegan!'", round(near_duplicate, 4)]]),
      ],
    ),
    example(
      "lc.l43.5",
      "Deleting by id",
      vector_store_basics,
      "Stable ids make updates and deletes possible.",
      [table(["items left"], [[remaining]])],
    ),
    example(
      "rag.l26.1",
      "A persistent collection with ids",
      create_a_persistent_collection,
      "",
      [table(["persist_directory", "items"], [[".cache/labs/m05/chroma", len(MENU_DOCS)]])],
    ),
    example(
      "rag.l26.2",
      "Reopen: nothing re-embedded",
      reopen_it,
      f"Reopened in {reopen_ms:.0f} ms vs {build_ms:.0f} ms to create.",
      [table(["count"], [[count]])],
    ),
    example(
      "rag.l26.3",
      "Distance vs relevance scores",
      search_with_scores,
      "Chroma's raw score is a distance (lower = closer); relevance scores are normalised to higher = closer.",
      [
        table(
          ["item", "distance", "relevance"],
          [[d.metadata["item"], round(s, 3), round(r, 3)] for (d, s), (_, r) in zip(distances, relevance, strict=True)],
        )
      ],
    ),
    example(
      "rag.l26.4",
      "Metadata filters: exact, comparisons, $and / $or / $in",
      metadata_filters,
      "",
      [
        table(
          ["filter", "items"],
          [
            ["cold", [d.metadata["item"] for d in exact]],
            ["< 200", [d.metadata["item"] for d in cheaper]],
            ["vegan and food/cold", [d.metadata["item"] for d in both]],
          ],
        )
      ],
    ),
    example(
      "rag.l26.5",
      "Read, update, delete without searching",
      crud_without_searching,
      "",
      [
        table(
          ["step", "result"],
          [["get menu-2", got["documents"][0]], ["deleted by filter (food)", str(removed)], ["ids left", str(left)]],
        )
      ],
    ),
  ]


# ── Lab 4: FAISS ────────────────────────────────────────────────────────────────
def build_and_search(chunks, embeddings):
  from langchain_community.vectorstores import FAISS

  store = FAISS.from_documents(chunks, embeddings, ids=[str(i) for i in range(len(chunks))])
  hits = store.similarity_search_with_score("hotel limit per night", k=2)  # L2 distance: lower = closer
  pdf_only = store.similarity_search("revenue", k=2, filter={"file_type": "pdf"})  # filtered AFTER the search
  return store, hits, pdf_only


def cosine_instead(chunks, embeddings):
  from langchain_community.vectorstores import FAISS
  from langchain_community.vectorstores.utils import DistanceStrategy

  # bge vectors have length 1, so inner product == cosine similarity (higher = closer)
  store = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
  hits = store.similarity_search_with_score("hotel limit per night", k=2)
  return hits


def save_and_load(store, folder, embeddings):
  from langchain_community.vectorstores import FAISS

  store.save_local(folder)  # index.faiss (vectors) + index.pkl (documents, pickled)
  # Pickle can run code when loaded: only load indexes you created yourself
  loaded = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
  return loaded


def merge_and_delete(store, embeddings):
  from langchain_community.vectorstores import FAISS

  extra = FAISS.from_texts(["Nimbus opens a café in Pune in 2027."], embeddings, ids=["news-1"])
  before = store.index.ntotal
  store.merge_from(extra)
  merged = store.index.ntotal
  store.delete(ids=["news-1"])
  return before, merged, store.index.ntotal


def exact_vs_approximate(m, ef_values, n=20_000, dims=384):
  import time

  import faiss
  import numpy as np

  rng = np.random.default_rng(0)  # clustered random vectors behave like real embeddings
  centers = rng.standard_normal((200, dims)).astype("float32")
  vectors = (centers[rng.integers(0, 200, n)] + 0.35 * rng.standard_normal((n, dims))).astype("float32")
  faiss.normalize_L2(vectors)
  queries = vectors[:200] + 0.05 * rng.standard_normal((200, dims)).astype("float32")
  faiss.normalize_L2(queries)

  def timed(index):
    start = time.perf_counter()
    _, ids = index.search(queries, 10)
    return ids, (time.perf_counter() - start) / len(queries) * 1000

  flat = faiss.IndexFlatL2(dims)  # compares the query with EVERY vector: exact
  flat.add(vectors)
  exact, flat_ms = timed(flat)
  hnsw = faiss.IndexHNSWFlat(dims, m)  # a graph walked towards the query: approximate
  hnsw.add(vectors)
  results = {"flat (exact)": (flat_ms, 1.0)}
  for ef in ef_values:
    hnsw.hnsw.efSearch = ef  # candidates explored per query: higher = more accurate, slower
    ids, ms = timed(hnsw)
    results[f"HNSW M={m} efSearch={ef}"] = (
      ms,
      float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact, strict=True)])),
    )
  return results


@lab(
  "rag.faiss",
  "FAISS: fast vector search, exact vs approximate",
  inputs=[
    Input("m", "HNSW links per node (M)", "number", default=32, min=4, max=64),
    Input("ef", "efSearch (second run)", "number", default=64, min=1, max=256),
  ],
)
def faiss_lab(m: int, ef: int):
  """FAISS through LangChain on production chunks, then raw FAISS on 20,000 vectors: exact search vs HNSW."""
  embeddings = _local(MODELS["bge-small (local, 384)"])
  chunks = list(_chunks())
  store, hits, pdf_only = build_and_search(chunks, embeddings)
  cosine_hits = cosine_instead(chunks, embeddings)
  folder = CACHE / "faiss"
  shutil.rmtree(folder, ignore_errors=True)
  loaded = save_and_load(store, str(folder), embeddings)
  files = sorted(p.name for p in folder.iterdir())
  before, merged, after = merge_and_delete(loaded, embeddings)
  shutil.rmtree(folder, ignore_errors=True)
  started = time.perf_counter()
  speed = exact_vs_approximate(m, (8, ef))
  seconds = time.perf_counter() - started
  rows = [{"variant": name, "ms": round(ms, 3), "recall": round(recall, 3)} for name, (ms, recall) in speed.items()]
  return [
    scorecard(
      rows,
      [Metric("ms", "ms per query", "lower", "text"), Metric("recall", "recall@10 vs exact", "higher", "ratio")],
      title=f"20,000 vectors, 200 queries ({seconds:.1f} s including building)",
    ),
    bars([{"label": r["variant"], "values": {"ms per query": r["ms"]}} for r in rows], title="Speed", unit="ms"),
    callout(
      "Exact search compares every vector: perfect recall, time grows with the index. HNSW walks a graph: far "
      "faster at scale, and `efSearch` trades recall for speed (watch recall drop at efSearch=8). pgvector's HNSW "
      "index (production) exposes the same knobs as `m`, `ef_construction` and `hnsw.ef_search`. FAISS is a "
      "library, not a database: no server, filters run after the search, and its save format uses pickle.",
      "decision",
      "Exact vs approximate",
    ),
  ] + [
    example(
      "rag.l27.1",
      "Build, search, filter",
      build_and_search,
      f"{store.index.ntotal} vectors in a flat (exact) index. The filter is applied after the search.",
      [
        table(["distance (L2, ↓)", "chunk"], [[round(s, 3), d.page_content[:80]] for d, s in hits]),
        table(["filter file_type=pdf"], [[d.page_content[:80]] for d in pdf_only]),
      ],
    ),
    example(
      "rag.l27.2",
      "L2 distance vs cosine",
      cosine_instead,
      "On unit-length vectors the order is the same; only the numbers change meaning.",
      [table(["cosine (↑)", "chunk"], [[round(s, 3), d.page_content[:80]] for d, s in cosine_hits])],
    ),
    example(
      "rag.l27.3",
      "save_local / load_local",
      save_and_load,
      "Loading needs `allow_dangerous_deserialization=True` because the documents are pickled.",
      [table(["files"], [[f] for f in files])],
    ),
    example(
      "rag.l27.4",
      "Merge indexes, delete by id",
      merge_and_delete,
      "",
      [table(["before", "after merge", "after delete"], [[before, merged, after]])],
    ),
    example(
      "rag.l27.5",
      "Exact (IndexFlat) vs approximate (HNSW)",
      exact_vs_approximate,
      "",
      [table(["index", "ms per query", "recall@10"], [[r["variant"], r["ms"], r["recall"]] for r in rows])],
    ),
  ]


# ── Lab 5: pgvector (production) ────────────────────────────────────────────────
def search_production_index(settings, embeddings):
  from nimbus.rag.ingestion.store import VectorIndex

  with VectorIndex(settings, embeddings) as index:  # production's table, written by ingestion
    hits = index.search("How many days of annual leave do I get?", k=3)  # cosine distance: lower = closer
    pdf_hits = index.store.similarity_search("revenue", k=2, filter={"file_type": {"$eq": "pdf"}})  # in SQL
  return hits, pdf_hits


def plain_sql(settings, embeddings):
  from sqlalchemy import create_engine, text

  vector = embeddings.embed_query("How many days of annual leave do I get?")
  engine = create_engine(settings.database_url)
  try:
    with engine.connect() as db:
      rows = db.execute(
        text(
          "select source, left(content, 70) as content, embedding <=> cast(:q as vector) as distance "
          "from rag_chunks order by embedding <=> cast(:q as vector) limit 3"
        ),  # <=> is cosine distance; <-> is L2; <#> is negative inner product
        {"q": str(vector)},
      ).all()
  finally:
    engine.dispose()
  return rows


def the_hnsw_index(settings):
  from sqlalchemy import create_engine, text

  engine = create_engine(settings.database_url)
  try:
    with engine.connect() as db:
      index = db.execute(text("select indexdef from pg_indexes where indexname = 'rag_chunks_hnsw'")).scalar()
      db.execute(text("set enable_seqscan = off"))  # 99 rows: Postgres would scan them all; force the index
      plan = (
        db.execute(
          text(
            "explain select langchain_id from rag_chunks order by embedding <=> (select embedding from rag_chunks "
            "limit 1) limit 5"
          )
        )
        .scalars()
        .all()
      )
  finally:
    engine.dispose()
  return index, plan


@lab("rag.pgvector", "pgvector: vector search inside PostgreSQL (production)")
def pgvector_lab():
  """Production's own index: search through LangChain, the same search in SQL, and the HNSW index."""
  settings = get_settings()
  embeddings = _local(settings.embedding_model)
  hits, pdf_hits = search_production_index(settings, embeddings)
  rows = plain_sql(settings, embeddings)
  index, plan = the_hnsw_index(settings)
  return [
    table(
      ["distance (↓)", "source", "chunk"],
      [[round(s, 4), d.metadata["source"], d.page_content[:90]] for d, s in hits],
      title="VectorIndex.search (production)",
    ),
    table(
      ["distance", "source", "content"],
      [[round(r.distance, 4), r.source, r.content] for r in rows],
      title="The same search in plain SQL",
    ),
    code("\n".join(plan), "text", title="EXPLAIN: the query walks the HNSW index"),
    callout(
      "pgvector stores vectors in a Postgres column next to the chunk text and metadata: one database, real "
      "transactions, backups and SQL filters, and the record manager's table beside it. Production creates an HNSW "
      "index once (VectorIndex.ensure_vector_index). Distances from LangChain and from SQL are the same numbers: "
      "it's just a query.",
      "decision",
      "pgvector",
    ),
  ] + [
    example(
      "rag.l28.1",
      "Postgres with pgvector, in Docker",
      search_production_index,
      "`docker compose up -d` starts `learn-ai-course-postgres` (the pgvector/pgvector:pg17 image) on port 5443.",
      [table(["database_url"], [[settings.database_url.replace("nimbus:nimbus@", "nimbus:•••@")]])],
    ),
    example(
      "rag.l28.2",
      "PGVectorStore: add, search, filter",
      search_production_index,
      "The same interface as every LangChain vector store; the filter runs in SQL.",
      [table(["source", "chunk"], [[d.metadata["source"], d.page_content[:80]] for d in pdf_hits])],
    ),
    example(
      "rag.l28.3",
      "It's just SQL: the <=> operator",
      plain_sql,
      "",
      [table(["distance", "source"], [[round(r.distance, 4), r.source] for r in rows])],
    ),
    example(
      "rag.l28.4",
      "An HNSW index",
      the_hnsw_index,
      "Created once by production; Postgres uses it for ORDER BY embedding <=> … LIMIT k.",
      [code(index or "(missing)", "sql")],
    ),
  ]


# ── Lab 6: keeping the index in sync ────────────────────────────────────────────
HANDBOOK = [
  Document("Hotels are reimbursed up to 6,000 rupees per night.", metadata={"source": "handbook.md"}),
  Document("Meals during travel are reimbursed up to 1,500 rupees per day.", metadata={"source": "handbook.md"}),
]
FAQ = [Document("Points expire 12 months after they are earned.", metadata={"source": "faq.html"})]


def without_a_record_manager(embeddings):
  from langchain_core.vectorstores import InMemoryVectorStore

  store = InMemoryVectorStore(embeddings)
  store.add_documents(HANDBOOK)
  store.add_documents(HANDBOOK)  # a second ingestion run
  return len(store.store)  # duplicates


def sync_with_a_record_manager(embeddings):
  from langchain_core.indexing import InMemoryRecordManager, index
  from langchain_core.vectorstores import InMemoryVectorStore

  store = InMemoryVectorStore(embeddings)
  records = InMemoryRecordManager(namespace="lab/sync")  # remembers a hash of every chunk written, per source
  records.create_schema()
  edited = [
    Document("Hotels are reimbursed up to 7,500 rupees per night.", metadata={"source": "handbook.md"}),
    HANDBOOK[1],
  ]
  runs = {}
  for label, docs, cleanup in (
    ("first run", HANDBOOK + FAQ, "incremental"),
    ("re-run, nothing changed", HANDBOOK + FAQ, "incremental"),
    ("handbook edited", edited, "incremental"),  # only sources in this batch are touched
    ("faq.html removed", edited, "full"),  # the batch IS the whole corpus: missing sources are deleted
  ):
    result = index(docs, records, store, cleanup=cleanup, source_id_key="source", key_encoder="sha256")
    runs[label] = (result, len(store.store))
  return runs


@lab("rag.index-sync", "Keeping the index in sync: re-index only what changed")
def index_sync():
  """Three re-runs with LangChain's index() and a record manager, against plain add_documents."""
  embeddings = _local(MODELS["bge-small (local, 384)"])
  naive = without_a_record_manager(embeddings)
  runs = sync_with_a_record_manager(embeddings)
  rows = [
    {"variant": label, "added": r["num_added"], "skipped": r["num_skipped"], "deleted": r["num_deleted"], "in_store": n}
    for label, (r, n) in runs.items()
  ]
  return [
    scorecard(
      rows,
      [
        Metric("added", "added", "none", "int"),
        Metric("skipped", "skipped (unchanged)", "none", "int"),
        Metric("deleted", "deleted", "none", "int"),
        Metric("in_store", "chunks in the store", "none", "int"),
      ],
      title="index() + a record manager over four runs",
    ),
    callout(
      f"Plain add_documents twice left {naive} chunks for 2 (duplicates). With a record manager, an unchanged run "
      "embeds nothing, an edit replaces the old chunk, and cleanup='full' deletes a removed source. Production does "
      "exactly this on Postgres (SQLRecordManager + cleanup='full'): every `nimbus rag ingest` is a sync.",
      "decision",
      "Sync, don't append",
    ),
  ] + [
    example(
      "rag.l29.1",
      "The problem: duplicates and stale chunks",
      without_a_record_manager,
      "Running ingestion twice adds everything twice.",
      [table(["chunks after two runs of 2"], [[naive]])],
    ),
    example(
      "rag.l29.2",
      "index() + a RecordManager",
      sync_with_a_record_manager,
      "The record manager stores a hash per chunk and which source it came from.",
      [
        table(
          ["run", "added", "skipped", "deleted"],
          [[r["variant"], r["added"], r["skipped"], r["deleted"]] for r in rows[:2]],
        )
      ],
    ),
    example(
      "rag.l29.3",
      "Unchanged skipped, edits replaced (incremental)",
      sync_with_a_record_manager,
      "",
      [
        table(
          ["run", "added", "deleted", "in store"],
          [[r["variant"], r["added"], r["deleted"], r["in_store"]] for r in rows[1:3]],
        )
      ],
    ),
    example(
      "rag.l29.4",
      "Removed sources deleted (full)",
      sync_with_a_record_manager,
      "",
      [table(["run", "deleted", "in store"], [[rows[3]["variant"], rows[3]["deleted"], rows[3]["in_store"]]])],
    ),
  ]


# ── Lab 7: ingestion at scale ───────────────────────────────────────────────────
ISSUES = ["arrived late", "was cold", "was damaged in transit", "was the wrong item", "had a missing lid"]
MAX_CHARS = 2_000


def tickets(n: int) -> Iterator[Document]:
  """Support tickets with the mess real exports have: empty rows, pasted logs, re-exported copies."""

  def one(i: int) -> str:
    return f"Ticket {i}: the customer says order NC-{1000 + i % 400} {ISSUES[i % 5]} at café {i % 64}."

  for i in range(n):
    text = one(i)
    if i % 50 == 7:
      text = "   "
    elif i % 97 == 3:
      text += " Log: " + "x" * 3_000
    elif i % 10 == 9:
      text = one(i - 1).upper().replace(":", " :")  # the previous ticket again
    yield Document(text, metadata={"ticket": i})


def in_batches(docs, size):
  batch = []
  for doc in docs:  # a generator in, a generator out: memory holds one batch
    batch.append(doc)
    if len(batch) == size:
      yield batch
      batch = []
  if batch:
    yield batch


async def embed_concurrently(embeddings, texts, batch_size, concurrency):
  import asyncio

  limit = asyncio.Semaphore(concurrency)  # at most N batches in flight: protects memory and API rate limits

  async def one(chunk):
    async with limit:
      return await embeddings.aembed_documents(chunk)

  results = await asyncio.gather(*[one(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)])
  return [v for chunk in results for v in chunk]


def dedupe(docs):
  import hashlib
  import re

  def key(doc):  # case and punctuation don't make a different ticket
    return hashlib.sha256(re.sub(r"[^a-z0-9]+", " ", doc.page_content.lower()).strip().encode()).hexdigest()[:16]

  unique = {}
  for doc in docs:
    unique.setdefault(key(doc), doc)
  return unique  # id → document: the id is stable, so re-writing it is harmless


def validate(doc):
  if not doc.page_content.strip():
    return "empty"
  if len(doc.page_content) > MAX_CHARS:
    return f"too long ({len(doc.page_content):,} chars)"
  return None  # fine


def resumable_ingest(store, n, batch_size, progress_file, crash_after=None):
  import json

  done = set(json.loads(progress_file.read_text())) if progress_file.exists() else set()
  written, dead_letter, seen = 0, [], set()
  for number, batch in enumerate(in_batches(tickets(n), batch_size)):
    if number in done:
      continue  # finished before the crash
    good = {}
    for doc in batch:
      if reason := validate(doc):
        dead_letter.append((doc.metadata["ticket"], reason))  # set aside with a reason, not fatal
        continue
      for doc_id, d in dedupe([doc]).items():
        if doc_id not in seen:
          seen.add(doc_id)
          good[doc_id] = d
    store.add_documents(list(good.values()), ids=list(good))
    written += len(good)
    done.add(number)
    progress_file.write_text(json.dumps(sorted(done)))  # recorded AFTER the batch is safely written
    if crash_after is not None and number == crash_after:
      raise RuntimeError(f"simulated crash after batch {number}")
  return written, dead_letter


@lab(
  "rag.ingestion-scale",
  "Ingestion at scale: batches, concurrency, duplicates, bad records, resuming",
  inputs=[
    Input("n", "Tickets", "number", default=1200, min=200, max=5000),
    Input("concurrency", "Concurrent batches", "number", default=4, min=1, max=8),
  ],
)
def ingestion_scale(n: int, concurrency: int):
  """A messy export of support tickets through a streaming, deduplicating, resumable pipeline."""
  from langchain_core.vectorstores import InMemoryVectorStore

  embeddings = _local(MODELS["bge-small (local, 384)"])
  clean = [d.page_content for d in tickets(n) if not validate(d)][:600]
  started = time.perf_counter()
  for t in clean[:60]:
    embeddings.embed_query(t)
  one_by_one = (time.perf_counter() - started) * len(clean) / 60
  started = time.perf_counter()
  embeddings.embed_documents(clean)
  batched = time.perf_counter() - started
  started = time.perf_counter()
  asyncio.run(embed_concurrently(embeddings, clean, 100, concurrency))
  concurrent = time.perf_counter() - started
  docs = [d for d in tickets(n) if not validate(d)]
  unique = dedupe(docs)
  store = InMemoryVectorStore(embeddings)
  progress = CACHE / "ingest_progress.json"
  CACHE.mkdir(parents=True, exist_ok=True)
  progress.unlink(missing_ok=True)
  try:
    resumable_ingest(store, n, 200, progress, crash_after=2)
    crash = "no crash"
  except RuntimeError as error:
    crash = f"{error}: {len(store.store)} written"
  written, dead_letter = resumable_ingest(store, n, 200, progress)
  after_resume = len(store.store)
  progress.unlink(missing_ok=True)
  resumable_ingest(store, n, 200, progress)  # a full re-run: every id exists already
  after_rerun = len(store.store)
  progress.unlink(missing_ok=True)
  rows = [
    {"variant": "one by one (estimated)", "seconds": round(one_by_one, 2)},
    {"variant": "one batched call", "seconds": round(batched, 2)},
    {"variant": f"{concurrency} concurrent batches of 100", "seconds": round(concurrent, 2)},
  ]
  return [
    scorecard(rows, [Metric("seconds", "seconds", "lower", "text")], title=f"Embedding {len(clean)} tickets"),
    table(
      ["Step", "Result"],
      [
        ["tickets in the export", n],
        ["valid", len(docs)],
        ["after removing exact copies", len(unique)],
        ["dead letter (empty / too long)", len(dead_letter)],
        ["first run", crash],
        ["after resuming", f"{after_resume} in the store (wrote {written} more)"],
        ["after a full re-run", f"{after_rerun} (same ids: nothing duplicated)"],
      ],
      title="The pipeline",
    ),
    callout(
      "Stream in batches so memory stays flat, embed in batches (one call per text is the slow way), and run "
      "batches concurrently with a limit. Records that can't be indexed go to a dead-letter list with a reason. "
      "Progress is recorded after each batch is written, and stable ids make writing a batch twice harmless, so a "
      "crash costs one batch. For a local model concurrency helps little (the CPU is the limit); for an API model "
      "each call waits on the network, so it helps a lot.",
      "decision",
      "At scale",
    ),
  ] + [
    example(
      "rag.l30.1",
      "Stream in batches",
      in_batches,
      "A generator of batches over a generator of documents.",
      [table(["batch sizes"], [[str([len(b) for b in in_batches(tickets(n), 200)])]])],
    ),
    example(
      "rag.l30.2",
      "Batched and concurrent embedding",
      embed_concurrently,
      "",
      [table(["way", "seconds"], [[r["variant"], r["seconds"]] for r in rows])],
    ),
    example(
      "rag.l30.3",
      "Deduplicate exact copies by a normalised hash",
      dedupe,
      "Upper-cased re-exports collapse onto the original. Near-copies need embedding similarity, and aren't always "
      "waste (similar tickets are different tickets).",
      [table(["valid", "unique"], [[len(docs), len(unique)]])],
    ),
    example(
      "rag.l30.4",
      "A dead-letter list instead of a crash",
      validate,
      "",
      [table(["ticket", "reason"], [[t, r] for t, r in dead_letter[:8]])],
    ),
    example(
      "rag.l30.5",
      "Resume after a crash",
      resumable_ingest,
      "",
      [
        table(
          ["moment", "chunks in the store"], [["crash", crash], ["resume", after_resume], ["full re-run", after_rerun]]
        )
      ],
    ),
  ]


# ── Production path: embeddings and query translation ──────────────────────────
@lab(
  "rag.embeddings-eval",
  "Evaluate: embedding models and query translation",
  uses_model=True,
  calls="cached after the first run (2 embedding requests, 10 translations)",
)
def embeddings_eval():
  """All 45 English questions and the 10 multilingual ones, per embedding model and query policy."""
  chunks = list(_chunks())
  texts = [c.page_content for c in chunks]
  model = get_chat_model()
  english = compare_models(MODELS, list(corpus.questions()), chunks)
  rows = []
  for name, model_name in MODELS.items():
    doc_vectors, _ = _vectors(model_name, texts)
    for policy in ("off", "non_ascii") if model_name == MODELS["bge-small (local, 384)"] else ("off",):
      queries = [
        translate(m["question"], model) if needs_translation(m["question"], policy) else m["question"]
        for m in MULTILINGUAL
      ]
      q_vectors, _ = _vectors(model_name, queries, query=True)
      found = sum(
        any(m["evidence"] in " ".join(chunks[i].page_content.split()) for i in np.argsort(-(doc_vectors @ q))[:3])
        for m, q in zip(MULTILINGUAL, q_vectors, strict=True)
      )
      report = english[name][3]
      rows.append(
        {
          "variant": name + (" + translate non-ASCII" if policy == "non_ascii" else ""),
          "hit5": report.at["hit"][5],
          "mrr": round(report.mrr, 3),
          "multilingual": round(found / len(MULTILINGUAL), 3),
          "per_question": "free"
          if "local" in name and policy == "off"
          else ("1 chat call if non-ASCII" if policy == "non_ascii" else "1 API request"),
        }
      )
  return [
    scorecard(
      rows,
      [
        Metric("hit5", "hit@5 (45 English)", "higher", "ratio"),
        Metric("mrr", "MRR", "higher", "ratio"),
        Metric("multilingual", "multilingual top-3 (10)", "higher", "ratio"),
        Metric("per_question", "cost per question", "none", "text"),
      ],
      title="Embedding model × query policy",
    ),
    callout(
      "bge-small with non-ASCII questions translated keeps the best free English quality and finds most "
      "multilingual questions, for a chat call only when the question isn't in English. mistral-embed is better "
      "on English, but every question becomes a request to the same provider (and rate limit) as the chat model, "
      "and the text leaves the machine. The multilingual local model gave up English quality. Production: "
      "bge-small, query_translation=non_ascii; mistral-embed is a setting (embedding_provider=mistral, a 1024-dim "
      "table) for deployments where its quality is worth it.",
      "decision",
      "Ship: local embeddings + query translation",
    ),
  ]
