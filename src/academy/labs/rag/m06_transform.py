"""RAG · Module 6 labs, part 2: reranking, query transformations, parent-document and multi-vector retrieval,
contextual compression, and the production retriever measured end to end."""

import os
import re
import time

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from academy.labs import cache
from academy.labs.rag import corpus
from academy.labs.rag.m06_retrieval import _rank, _store
from academy.labs.runtime import Input, Metric, callout, code, columns, example, lab, scorecard, table
from nimbus.config import get_settings
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import evaluate, is_relevant

DATA = get_settings().data_dir


def _has(phrase: str):
  return lambda text: phrase.casefold() in " ".join(text.split()).casefold()


# ── Lab 6: reranking ────────────────────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.

RERANK_QUESTIONS = {
  "When does the supplier contract end?": "31 December 2027",
  "Can I bring my family on health insurance?": "spouse",
  "What happens if beans score below 84 points?": "may be rejected",
  "How long can I stay on the café Wi-Fi?": "limited to 2 hours",
}


def bi_encoder_scores(embeddings, question, texts):
  import numpy as np

  q = np.array(embeddings.embed_query(question))  # the question alone
  d = np.array(embeddings.embed_documents(texts))  # each chunk alone, computed in advance
  return list(d @ q)  # fast, but neither side ever saw the other


def cross_encoder_scores(question, texts):
  from flashrank import RerankRequest

  from nimbus.rag.rerank import ranker

  s = get_settings()
  request = RerankRequest(query=question, passages=[{"id": i, "text": t} for i, t in enumerate(texts)])
  scored = ranker(s.reranker_model, s.model_cache_dir).rerank(request)  # reads question + chunk TOGETHER
  return {r["id"]: float(r["score"]) for r in scored}


def two_stage(store, question, candidates, keep):
  from nimbus.rag.rerank import rerank

  first = store.similarity_search(question, k=candidates)  # stage 1: fast and coarse, many candidates
  s = get_settings()
  second = rerank(question, first, s.reranker_model, s.model_cache_dir, keep)  # stage 2: careful, few kept
  return first, second


class Relevance(BaseModel):
  score: int = Field(description="0 = irrelevant … 10 = answers the question directly")


def llm_rerank(model, question, docs):
  scores = [
    cache.structured(
      model, Relevance, f"Question: {question}\n\nPassage:\n{d.page_content}\n\nHow relevant is the passage?"
    ).score
    for d in docs
  ]
  return [d for _, d in sorted(zip(scores, docs, strict=True), key=lambda p: -p[0])], scores  # a call per chunk


def cohere_rerank(question, docs, keep):
  if not os.getenv("COHERE_API_KEY"):
    return None  # a hosted reranker: needs COHERE_API_KEY in ai-course/.env (free trial key at dashboard.cohere.com)
  from langchain_cohere import CohereRerank

  return CohereRerank(model="rerank-v3.5", top_n=keep).compress_documents(docs, question)


@lab(
  "rag.reranking",
  "Reranking: a second, careful look at the top candidates",
  uses_model=True,
  calls="up to 10 calls to Mistral for LLM reranking (cached after the first run)",
  inputs=[
    Input("candidates", "Candidates fetched", "number", default=10, min=3, max=30),
    Input("keep", "Keep after reranking", "number", default=3, min=1, max=10),
  ],
)
def reranking(candidates: int, keep: int):
  """Embeddings fetch candidates; a cross-encoder (FlashRank, local) reorders them. Measured on all 45 questions."""
  store = _store()
  per_q = []
  for q, phrase in RERANK_QUESTIONS.items():
    first, second = two_stage(store, q, candidates, keep)
    per_q.append([q, _rank(first, _has(phrase)) or "✗", _rank([d for d, _ in second], _has(phrase)) or "✗"])
  rows = []
  s = get_settings()
  from nimbus.rag.rerank import rerank

  started = time.perf_counter()
  plain_hits, rr_hits = [], []
  for q in corpus.questions():
    first = store.similarity_search(q.question, k=candidates)
    plain_hits.append(_rank(first[:keep], lambda t, q=q: is_relevant(t, q)))
    rr_hits.append(
      _rank(
        [d for d, _ in rerank(q.question, first, s.reranker_model, s.model_cache_dir, keep)],
        lambda t, q=q: is_relevant(t, q),
      )
    )
  seconds = time.perf_counter() - started
  for label, hits in (("embeddings only", plain_hits), (f"+ FlashRank over {candidates}", rr_hits)):
    rows.append(
      {
        "variant": label,
        "hit_at_keep": round(sum(h is not None for h in hits) / len(hits), 3),
        "hit1": round(sum(h == 1 for h in hits) / len(hits), 3),
        "mrr": round(sum(1 / h for h in hits if h) / len(hits), 3),
      }
    )
  question = next(iter(RERANK_QUESTIONS))
  texts = [d.page_content for d in store.similarity_search(question, k=6)]
  bi = bi_encoder_scores(corpus.embeddings(), question, texts)
  cross = cross_encoder_scores(question, texts)
  model = get_chat_model()
  first, _ = two_stage(store, question, 10, keep)
  llm_order, llm_scores = llm_rerank(model, question, first)
  cohere = cohere_rerank(question, first, keep)
  return [
    scorecard(
      rows,
      [
        Metric("hit_at_keep", f"answer in the top {keep}", "higher", "ratio"),
        Metric("hit1", "answer first", "higher", "ratio"),
        Metric("mrr", "MRR", "higher", "ratio"),
      ],
      title=f"45 questions · rerank pass {seconds:.1f} s ({seconds / 45 * 1000:.0f} ms per question)",
    ),
    table(["Question", "embedding rank", "after FlashRank"], per_q, title="Where the answer chunk lands"),
    table(
      ["chunk", "bi-encoder (cosine)", "cross-encoder"],
      [[t[:70], round(b, 3), round(cross[i], 4)] for i, (t, b) in enumerate(zip(texts, bi, strict=True))],
      title=question,
    ),
    callout(
      "Embeddings compare a question and a chunk that were encoded separately, so many near-misses score alike. A "
      "cross-encoder reads them together and separates the right chunk sharply, at milliseconds per pair, so it "
      "only reorders a short candidate list. Fetch enough candidates for the answer to be among them, then keep "
      "few. Production reranks 10 hybrid candidates with FlashRank (local, no API key).",
      "decision",
      "Two-stage retrieval",
    ),
  ] + [
    example(
      "rag.l36.1",
      "Bi-encoders: fast but coarse",
      bi_encoder_scores,
      "The scores of near-misses bunch together.",
      [table(["chunk", "cosine"], [[t[:70], round(b, 3)] for t, b in zip(texts, bi, strict=True)])],
    ),
    example(
      "rag.l36.2",
      "Cross-encoders: question and chunk together",
      cross_encoder_scores,
      "",
      [
        table(
          ["chunk", "score"], [[texts[i][:70], round(sc, 4)] for i, sc in sorted(cross.items(), key=lambda p: -p[1])]
        )
      ],
    ),
    example(
      "rag.l36.3",
      "Two stages: many candidates, few kept",
      two_stage,
      "",
      [table(["Question", "stage 1", "stage 2"], per_q)],
    ),
    example(
      "rag.l36.4",
      "FlashRank: a local cross-encoder",
      cross_encoder_scores,
      f"Model `{s.reranker_model}`, ~22 MB, downloaded once; no API key.",
      [table(["", "hit@keep", "MRR"], [[r["variant"], r["hit_at_keep"], r["mrr"]] for r in rows])],
    ),
    example(
      "rag.l36.5",
      "Cohere Rerank (hosted)",
      cohere_rerank,
      "Runs only when COHERE_API_KEY is set; skipped here otherwise.",
      [
        table(
          ["result"],
          [["skipped: COHERE_API_KEY is not set" if cohere is None else str([d.page_content[:40] for d in cohere])]],
        )
      ],
    ),
    example(
      "lc.l47.3",
      "Re-ranking with an LLM",
      llm_rerank,
      "An LLM scores each candidate 0–10: flexible, but a call per chunk.",
      [
        table(
          ["LLM score", "chunk"],
          [[sc, d.page_content[:70]] for sc, d in sorted(zip(llm_scores, first, strict=True), key=lambda p: -p[0])],
        )
      ],
    ),
  ]


# ── Lab 7: query transformations ────────────────────────────────────────────────
VAGUE = "What perks come with the job?"


class Queries(BaseModel):
  queries: list[str]


def a_vague_question(store):
  hits = store.similarity_search(VAGUE, k=3)  # "perks" matches little: the benefits are spread across chunks
  return hits


def multi_query(model, store, question, n):
  rewrites = cache.structured(
    model,
    Queries,
    f"Write {n} short, different search queries to find this in Nimbus Coffee's internal documents (no "
    f"placeholders): {question}",
  ).queries
  merged = {d.page_content: d for q in [question, *rewrites] for d in store.similarity_search(q, k=3)}  # union
  return rewrites, list(merged.values())


def hyde(model, store, question):
  hypothetical = cache.text(model, f"Write a short passage from a coffee supply contract that answers: {question}")
  hits = store.similarity_search(hypothetical, k=10)  # an answer-shaped query looks like the documents
  return hypothetical, hits


def step_back(model, store, question):
  general = cache.text(
    model,
    "Rewrite as a short, general company-policy question without the specifics. Reply with the question "
    f"only.\n{question}",
  )
  hits = store.similarity_search(general, k=2)  # find the rule; the model applies it to the specifics later
  return general, hits


def decompose(model, store, question):
  parts = cache.structured(model, Queries, f"Split into independent sub-questions: {question}").queries
  found = {p: store.similarity_search(p, k=1)[0] for p in parts}  # one search per part
  return found


def built_in_multi_query(model, store):
  from langchain_classic.retrievers.multi_query import MultiQueryRetriever

  retriever = MultiQueryRetriever.from_llm(retriever=store.as_retriever(search_kwargs={"k": 3}), llm=model)
  return retriever.invoke(VAGUE)


@lab(
  "rag.query-transform",
  "Query transformations: improve the question before searching",
  uses_model=True,
  calls="~6 calls to Mistral (cached after the first run, except the built-in retriever)",
)
def query_transform():
  """Multi-query, HyDE, step-back and decomposition, each on the kind of question it helps."""
  store = _store()
  model = get_chat_model()
  plain = a_vague_question(store)
  rewrites, merged = multi_query(model, store, VAGUE, 3)
  benefit_words = ("discount", "learning budget", "insurance", "free drink", "parental", "leave")
  covered = lambda docs: sorted({w for w in benefit_words for d in docs if w in d.page_content.casefold()})  # noqa: E731
  hq = "What quality score must supplier beans reach?"
  hypothetical, hyde_hits = hyde(model, store, hq)
  plain_hq = store.similarity_search(hq, k=10)
  sb_q = "Can I expense a 7,000 rupee hotel night in Mumbai?"
  general, sb_hits = step_back(model, store, sb_q)
  dq = "What is the learning budget, and when are orders shipped after the roastery shutdown?"
  parts = decompose(model, store, dq)
  builtin = built_in_multi_query(model, store)
  rows = [
    {
      "variant": "vague question: plain search",
      "result": f"{len(covered(plain))} benefit topics: {', '.join(covered(plain))}",
    },
    {
      "variant": "vague question: multi-query",
      "result": f"{len(covered(merged))} benefit topics: {', '.join(covered(merged))}",
    },
    {
      "variant": "contract question: plain",
      "result": f"answer at rank {_rank(plain_hq, _has('84 points')) or '✗ (not in 10)'}",
    },
    {
      "variant": "contract question: HyDE",
      "result": f"answer at rank {_rank(hyde_hits, _has('84 points')) or '✗ (not in 10)'}",
    },
  ]
  return [
    scorecard(rows, [Metric("result", "result", "none", "text")], title="Each transformation on the question it's for"),
    table(
      ["Transformation", "Query used", "Top result"],
      [
        ["step-back", general, sb_hits[0].page_content[:90]],
        *[["decomposition", p, d.page_content[:90]] for p, d in parts.items()],
      ],
    ),
    callout(
      "Each transformation targets one failure: multi-query widens a vague question, HyDE searches with an "
      "answer-shaped text, step-back finds the general rule, decomposition searches each part separately. "
      + (
        "On this corpus neither multi-query nor HyDE changed the result (same topics, same rank): production's "
        "hybrid search and reranker already find these. "
        if rows[0]["result"].split(" ")[0] == rows[1]["result"].split(" ")[0] and rows[2]["result"] == rows[3]["result"]
        else "Here they changed what was found: compare the rows above. "
      )
      + "All of them cost model calls before every search, so production doesn't run them by default: add one "
      "where your evaluation shows that kind of question failing.",
      "decision",
      "Transform the question",
    ),
  ] + [
    example(
      "lc.l47.1",
      "The problem: a vague question retrieves weak chunks",
      a_vague_question,
      "",
      [table(["result"], [[d.page_content[:90]] for d in plain])],
    ),
    example(
      "lc.l47.2",
      "Multi-query: rewrite, search, merge",
      multi_query,
      "",
      [table(["rewrite"], [[r] for r in rewrites]), table(["benefit topics covered"], [[", ".join(covered(merged))]])],
    ),
    example(
      "rag.l37.1",
      "Multi-query for recall",
      multi_query,
      "",
      [table(["way", "topics"], [[r["variant"], r["result"]] for r in rows[:2]])],
    ),
    example(
      "rag.l37.2",
      "HyDE: search with a hypothetical answer",
      hyde,
      "The made-up facts don't matter; the contract-like wording does.",
      [
        code(hypothetical[:400], "text", "hypothetical answer"),
        table(["way", "rank"], [[r["variant"], r["result"]] for r in rows[2:]]),
      ],
    ),
    example(
      "rag.l37.3",
      "Step-back: the general rule",
      step_back,
      "",
      [table(["question", "general question", "found"], [[sb_q, general, sb_hits[0].page_content[:80]]])],
    ),
    example(
      "rag.l37.4",
      "Decomposition: one search per part",
      decompose,
      "",
      [table(["sub-question", "found"], [[p, d.page_content[:80]] for p, d in parts.items()])],
    ),
    example(
      "rag.l37.5",
      "The built-in MultiQueryRetriever",
      built_in_multi_query,
      "",
      [table(["result"], [[d.page_content[:90]] for d in builtin])],
    ),
  ]


# ── Lab 8: parent-document and multi-vector retrieval ───────────────────────────
def _parents() -> list[Document]:
  """Sections of the handbook and the supplier contract: the 'big' units an answer needs."""
  sections = [c for c in corpus.documents("v3-tesseract") if c.metadata["source"] == "supplier_contract.docx"]
  text = (DATA / "handbook.md").read_text()
  for part in text.split("\n## ")[1:]:
    sections.append(Document("## " + part.strip(), metadata={"source": "handbook.md"}))
  for i, s in enumerate(sections):
    s.metadata = {**s.metadata, "parent_id": f"parent-{i}"}
  return sections


def small_vs_big(parents, question):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  small = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=0).split_documents(parents)
  store = InMemoryVectorStore.from_documents(small, corpus.embeddings())
  child = store.similarity_search(question, k=1)[0]  # precise match, but too little to answer from
  return child


def parent_document(parents, question):
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  by_id = {p.metadata["parent_id"]: p for p in parents}  # the docstore: id → full section
  children = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=0).split_documents(parents)  # parent_id copied
  store = InMemoryVectorStore.from_documents(children, corpus.embeddings())
  child = store.similarity_search(question, k=1)[0]
  return child, by_id[child.metadata["parent_id"]]  # search small, return big


def summaries_index(model, parents, question):
  summaries = [
    cache.text(model, f"Summarise in one plain sentence (no quotes, no formatting):\n{p.page_content}") for p in parents
  ]
  store = InMemoryVectorStore.from_documents(
    [Document(s, metadata={"parent_id": p.metadata["parent_id"]}) for s, p in zip(summaries, parents, strict=True)],
    corpus.embeddings(),
  )
  hit = store.similarity_search(question, k=1)[0]
  return hit, next(p for p in parents if p.metadata["parent_id"] == hit.metadata["parent_id"])


def hypothetical_questions_index(model, parents, question):
  sets = [
    cache.structured(
      model, Queries, f"Write 3 questions a staff member might ask that this text answers:\n{p.page_content}"
    ).queries
    for p in parents
  ]
  docs = [
    Document(q, metadata={"parent_id": p.metadata["parent_id"]})
    for qs, p in zip(sets, parents, strict=True)
    for q in qs
  ]
  store = InMemoryVectorStore.from_documents(docs, corpus.embeddings())
  hit = store.similarity_search(question, k=1)[0]  # questions match questions better than passages do
  return hit, next(p for p in parents if p.metadata["parent_id"] == hit.metadata["parent_id"])


def built_in_parent_retriever(parents, question):
  from langchain_classic.retrievers import ParentDocumentRetriever
  from langchain_core.stores import InMemoryStore
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  retriever = ParentDocumentRetriever(
    vectorstore=InMemoryVectorStore(corpus.embeddings()),
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=0),
  )
  retriever.add_documents(parents)  # splits, indexes the children, stores the parents
  return retriever.invoke(question)


@lab(
  "rag.parent-docs",
  "Parent-document and multi-vector retrieval: search small, return big",
  uses_model=True,
  calls="2 calls per section the first time (summaries and questions), then cached",
)
def parent_docs():
  """Index small pieces, summaries or likely questions; return the whole section the answer needs."""
  model = get_chat_model()
  parents = _parents()
  question = "Do I need a doctor's note when I'm sick?"
  child = small_vs_big(parents, question)
  _, parent = parent_document(parents, question)
  s_hit, s_parent = summaries_index(model, parents, "rules for accepting presents from vendors")
  q_hit, q_parent = hypothetical_questions_index(model, parents, "What score do supplier beans need?")
  builtin = built_in_parent_retriever(parents, question)
  checks = []
  for q in corpus.questions():
    if q.source not in ("handbook.md", "supplier_contract.docx"):
      continue
    c, p = parent_document(parents, q.question)
    checks.append((is_relevant(c.page_content, q), is_relevant(p.page_content, q)))
  rows = [
    {
      "variant": "the matched small chunk (90 chars)",
      "contains_answer": round(sum(a for a, _ in checks) / len(checks), 3),
      "chars": len(child.page_content),
    },
    {
      "variant": "its parent section",
      "contains_answer": round(sum(b for _, b in checks) / len(checks), 3),
      "chars": len(parent.page_content),
    },
  ]
  return [
    scorecard(
      rows,
      [
        Metric(
          "contains_answer",
          "contains the whole answer",
          "higher",
          "ratio",
          f"{len(checks)} questions about these files",
        ),
        Metric("chars", "characters (example)", "none", "int"),
      ],
      title="Search small, return big",
    ),
    table(
      ["Index", "matched", "returned section starts"],
      [
        ["small chunks", child.page_content, parent.page_content[:80]],
        ["summaries", s_hit.page_content, s_parent.page_content[:80]],
        ["hypothetical questions", q_hit.page_content, q_parent.page_content[:80]],
      ],
    ),
    callout(
      "Small chunks match precisely but often hold half an answer; big chunks hold the answer but match vaguely. "
      "Index something small (chunks, a summary, or questions the section answers) and return the parent section. "
      "Summaries and hypothetical questions cost model calls per section at indexing time. Production's chunks are "
      "small (300) with headers; parent retrieval is the next lever if generation shows answers cut in half.",
      "decision",
      "Search small, return big",
    ),
  ] + [
    example(
      "rag.l38.1", "The chunk-size dilemma", small_vs_big, "", [table(["matched child"], [[child.page_content]])]
    ),
    example(
      "rag.l38.2",
      "Parent-document retrieval",
      parent_document,
      "",
      [code(parent.page_content, "markdown", "returned parent")],
    ),
    example(
      "rag.l38.3",
      "Multi-vector: index summaries",
      summaries_index,
      "",
      [table(["matched summary", "returned section"], [[s_hit.page_content, s_parent.page_content[:100]]])],
    ),
    example(
      "rag.l38.4",
      "Multi-vector: index hypothetical questions",
      hypothetical_questions_index,
      "",
      [table(["matched question", "returned section"], [[q_hit.page_content, q_parent.page_content[:100]]])],
    ),
    example(
      "rag.l38.5",
      "The built-in ParentDocumentRetriever",
      built_in_parent_retriever,
      "",
      [table(["returned"], [[d.page_content[:120]] for d in builtin[:2]])],
    ),
  ]


# ── Lab 9: contextual compression ───────────────────────────────────────────────
class Extract(BaseModel):
  sentences: list[str] = Field(
    description="Sentences copied word for word from the text that help answer the question; empty if none"
  )


def what_retrieval_returns(store, question):
  docs = store.similarity_search(question, k=3)
  return docs, sum(len(d.page_content) for d in docs)  # mostly sentences around the one that matters


def embedding_filter(embeddings, question, docs, threshold):
  import numpy as np

  q = np.array(embeddings.embed_query(question))
  kept = []
  for d in docs:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", d.page_content) if len(s) > 15]
    scores = np.array(embeddings.embed_documents(sentences)) @ q
    kept += [s for s, score in zip(sentences, scores, strict=True) if score > threshold]  # cheap: no model call
  return kept


def llm_extraction(model, question, docs):
  results = [cache.structured(model, Extract, f"Question: {question}\n\nText:\n{d.page_content}") for d in docs]
  extracted = [s for r in results for s in r.sentences]
  verbatim = all(any(s in d.page_content for d in docs) for s in extracted)  # an LLM may paraphrase: check
  return extracted, verbatim


def built_in_compression(store, embeddings, question, threshold):
  from langchain_classic.retrievers import ContextualCompressionRetriever
  from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

  retriever = ContextualCompressionRetriever(
    base_retriever=store.as_retriever(search_kwargs={"k": 3}),
    base_compressor=EmbeddingsFilter(embeddings=embeddings, similarity_threshold=threshold),  # drops whole chunks
  )
  return retriever.invoke(question)


@lab(
  "rag.compression",
  "Contextual compression: keep only what answers the question",
  uses_model=True,
  calls="3 calls to Mistral (cached after the first run)",
  inputs=[
    Input("question", "Question", "text", default="How long is parental leave?"),
    Input("threshold", "Similarity threshold", "number", default=0.7, min=0.4, max=0.9),
  ],
)
def compression(question: str, threshold: float):
  """Three retrieved chunks cut down to the sentences that matter, two ways, with the characters saved."""
  store = _store()
  model = get_chat_model()
  docs, total = what_retrieval_returns(store, question)
  kept = embedding_filter(corpus.embeddings(), question, docs, threshold)
  extracted, verbatim = llm_extraction(model, question, docs)
  builtin = built_in_compression(store, corpus.embeddings(), question, threshold)
  rows = [
    {"variant": "retrieved chunks", "chars": total, "saved": "—"},
    {
      "variant": f"embedding filter (> {threshold})",
      "chars": sum(map(len, kept)),
      "saved": f"{1 - sum(map(len, kept)) / total:.0%}",
    },
    {
      "variant": "LLM extraction",
      "chars": sum(map(len, extracted)),
      "saved": f"{1 - sum(map(len, extracted)) / total:.0%}",
    },
  ]
  return [
    scorecard(
      rows,
      [Metric("chars", "characters sent to the model", "lower", "int"), Metric("saved", "saved", "none", "text")],
      title=question,
    ),
    columns(
      [
        code("\n".join(kept) or "(nothing kept)", "text", "embedding filter"),
        code("\n".join(extracted) or "(nothing)", "text", f"LLM extraction (verbatim: {verbatim})"),
      ]
    ),
    callout(
      "Most of a retrieved chunk is context around the sentence that answers. An embedding filter keeps similar "
      "sentences for free but can drop the answer if the threshold is too high; LLM extraction is precise but costs "
      "a call per chunk and may paraphrase, so check its output against the source. With small chunks and a "
      "reranker, production sends little noise already; compression matters most with big chunks or parent sections.",
      "decision",
      "Compression",
    ),
  ] + [
    example(
      "rag.l39.1",
      "What retrieval returns",
      what_retrieval_returns,
      "",
      [table(["source", "chunk"], [[d.metadata["source"], d.page_content] for d in docs])],
    ),
    example(
      "rag.l39.2",
      "Embedding filter: keep similar sentences",
      embedding_filter,
      "",
      [table(["kept"], [[k] for k in kept])],
    ),
    example(
      "rag.l39.3",
      "LLM extraction: copy out what answers",
      llm_extraction,
      "",
      [table(["extracted", "verbatim?"], [[e, verbatim] for e in extracted])],
    ),
    example(
      "rag.l39.4",
      "Measure the savings",
      what_retrieval_returns,
      "",
      [table(["way", "characters", "saved"], [[r["variant"], r["chars"], r["saved"]] for r in rows])],
    ),
    example(
      "rag.l39.5",
      "The built-in ContextualCompressionRetriever",
      built_in_compression,
      "With EmbeddingsFilter it drops whole chunks below the threshold.",
      [table(["kept chunk"], [[d.page_content[:100]] for d in builtin])],
    ),
  ]


# ── Production path: retriever v3 ───────────────────────────────────────────────
CONFIGS = [  # (label, search_mode, rerank, candidates)
  ("v2 · dense", "dense", "off", 5),
  ("hybrid (Postgres full-text + vector, RRF)", "hybrid", "off", 20),
  ("dense + FlashRank over 10", "dense", "flashrank", 10),
  ("hybrid + FlashRank over 10", "hybrid", "flashrank", 10),
  ("hybrid + FlashRank over 40", "hybrid", "flashrank", 40),
]


def evaluate_retriever(k: int = 5) -> list[dict]:
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.rag.ingestion.store import VectorIndex
  from nimbus.rag.retriever import Retriever

  base = get_settings()
  embeddings = get_embeddings(base)
  with VectorIndex(base, embeddings) as index:
    texts, sources = index.texts(), index.sources()
  rows = []
  for label, mode, rr, candidates in CONFIGS:
    settings = base.model_copy(update={"search_mode": mode, "rerank": rr, "search_candidates": candidates})
    retriever = Retriever(settings, embeddings)
    report = evaluate(
      lambda q, n, r=retriever: [d for d, _ in r.search(q, n)[1]],
      corpus.questions(),
      texts,
      k=k,
      indexed_sources=sources,
    )
    rows.append(
      {
        "variant": label,
        "hit1": report.at["hit"][1],
        "hit3": report.at["hit"][3],
        "hit5": report.at["hit"][k],
        "mrr": round(report.mrr, 3),
        "ndcg": report.at["ndcg"][k],
        "p50_ms": round(report.latency_p50_ms),
      }
    )
  return rows


@lab(
  "rag.retrieval-eval",
  "Evaluate: the production retriever, five configurations",
  inputs=[Input("k", "Top k", "number", default=5, min=1, max=10)],
)
def retrieval_eval(k: int):
  """The real index in Postgres and the production Retriever; only search mode, reranking and candidates change."""
  started = time.perf_counter()
  rows = evaluate_retriever(k)
  seconds = time.perf_counter() - started
  best = max(rows, key=lambda r: (r["mrr"], -r["p50_ms"]))
  base = rows[0]
  return [
    scorecard(
      rows,
      [
        Metric("hit1", "hit@1", "higher", "ratio"),
        Metric("hit3", "hit@3", "higher", "ratio"),
        Metric("hit5", f"hit@{k}", "higher", "ratio"),
        Metric("mrr", "MRR", "higher", "ratio"),
        Metric("ndcg", f"nDCG@{k}", "higher", "ratio"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"45 questions against production's index · {seconds:.1f} s",
    ),
    callout(
      f"Best: {best['variant']}: hit@1 {base['hit1']:.2f} → {best['hit1']:.2f}, MRR {base['mrr']:.3f} → "
      f"{best['mrr']:.3f}, for {best['p50_ms'] - base['p50_ms']} ms more per question. Hybrid search found exact "
      "terms dense search missed; the reranker put the right chunk first far more often. More candidates for the "
      "reranker cost time without beating 10. Production ships search_mode=hybrid, rerank=flashrank, "
      "search_candidates=10, and filters every search to documents valid today.",
      "decision",
      "Ship retriever v3",
    ),
  ]
