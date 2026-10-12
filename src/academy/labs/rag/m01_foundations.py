"""RAG · Module 1 labs: why RAG exists, the two ways to search text, and how they compare on real questions."""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

from academy.labs.rag import corpus
from academy.labs.runtime import (
  Input,
  Metric,
  answer,
  bars,
  callout,
  columns,
  example,
  lab,
  md,
  results,
  scorecard,
  table,
)
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import evaluate, is_relevant, question_metrics, reciprocal_rank
from nimbus.rag.ingestion.loaders import FILE_TYPES

# Reference prices for the cost column: Claude Sonnet 5.5 list price (Oct 2026), per million tokens.
# The course runs on Mistral's free plan; this shows what the same traffic would cost on a paid model.
PRICE_IN, PRICE_OUT = 2.00, 10.00

WHY_RAG_QUESTIONS = {
  "How many days of paid annual leave does an employee get?": "24",
  "What was Nimbus's total revenue in 2025?": "208",
  "How many points do I need for a free bag?": "500",
}
ANSWER_RULES = "Answer in one or two sentences. If the information isn't available to you, say you don't know."


# ── Lab 1: why RAG ─────────────────────────────────────────────────────────────
@lab(
  "rag.why-rag",
  "Why RAG: the model doesn't know your data",
  uses_model=True,
  calls="3 calls to Mistral",
  inputs=[
    Input("question", "Question", "select", default=next(iter(WHY_RAG_QUESTIONS)), options=list(WHY_RAG_QUESTIONS)),
  ],
)
def why_rag(question: str):
  """Ask the same question three ways: no context, the whole corpus pasted in, and only the relevant chunks."""
  model = get_chat_model()
  everything = "\n\n---\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in corpus.documents())
  top = _semantic_rank(question, 3)
  relevant = "\n\n---\n\n".join(f"[{c.metadata['source']}]\n{c.page_content}" for c, _ in top)
  attempts = [
    ("No context (model's own knowledge)", [SystemMessage(ANSWER_RULES), HumanMessage(question)]),
    (
      "Stuff everything (all documents)",
      [SystemMessage(f"{ANSWER_RULES}\n\nDocuments:\n{everything}"), HumanMessage(question)],
    ),
    (
      "RAG (top 3 relevant chunks)",
      [SystemMessage(f"{ANSWER_RULES}\n\nDocuments:\n{relevant}"), HumanMessage(question)],
    ),
  ]
  expected = WHY_RAG_QUESTIONS[question]
  rows, answers = [], []
  for label, messages in attempts:
    started = time.perf_counter()
    reply = model.invoke(messages)
    ms = (time.perf_counter() - started) * 1000
    usage = reply.usage_metadata or {}
    tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    rows.append(
      {
        "variant": label,
        "correct": "✓" if expected in reply.text else "✗",
        "input_tokens": tokens_in,
        "latency_ms": round(ms),
        "cost_per_1k_questions": round((tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1000, 3),
      }
    )
    answers.append(answer(reply.text, title=label, meta={"input tokens": tokens_in, "output tokens": tokens_out}))
  return [
    scorecard(
      rows,
      [
        Metric("correct", "Correct?", "none", "text", f"Does the answer contain {expected}?"),
        Metric("input_tokens", "Input tokens", "lower", "int", "What every question pays for"),
        Metric("latency_ms", "Latency", "lower", "ms"),
        Metric(
          "cost_per_1k_questions",
          "$ per 1,000 questions",
          "lower",
          "usd",
          f"At ${PRICE_IN}/M input, ${PRICE_OUT}/M output tokens (Claude Sonnet 5.5 list price)",
        ),
      ],
      title="Three ways to answer",
    ),
    columns(answers, title="The answers"),
    callout(
      "Retrieval sends only what's relevant: the same answer as stuffing everything, for a fraction of the tokens. "
      "And stuffing stops working entirely once the corpus is bigger than the context window.",
      "decision",
      "Takeaway",
    ),
  ]


# ── Lab 2: keyword search ──────────────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def split_into_terms(text):
  import re

  stop_words = {"the", "a", "is", "do", "i", "how", "what", "who", "can"}  # (the lab's list is longer)
  words = re.findall(r"[a-z0-9@.\-]+[a-z0-9]|[a-z0-9]", text.lower())  # keeps it-help@nimbus.example as ONE term
  terms = [w for w in words if w not in stop_words]
  return terms


def score_by_overlap(query, chunks):
  from academy.labs.rag.corpus import tokenize

  terms = set(tokenize(query))
  scores = [len(terms & set(tokenize(chunk.page_content))) for chunk in chunks]  # words in common
  return scores


def rank_with_bm25(query, chunks):
  from rank_bm25 import BM25Okapi

  from academy.labs.rag.corpus import tokenize

  bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])  # word statistics of the whole corpus
  scores = bm25.get_scores(tokenize(query))  # rare words count more; repeats give diminishing returns
  top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:3]
  return scores, top


def search_an_exact_term(chunks):
  exact_scores, exact_top = rank_with_bm25("it-help@nimbus.example", chunks)  # an email, a code, an order id
  return exact_scores, exact_top


def search_a_paraphrase(chunks):
  # The handbook says "annual leave", never "holidays": no shared word, nothing for keyword search to find
  scores, top = rank_with_bm25("How many holidays do I get?", chunks)
  return scores, top


@lab(
  "rag.keyword-search",
  "Keyword search: word overlap and BM25",
  inputs=[
    Input(
      "query",
      "Search query",
      "text",
      default="Who approves remote work outside India?",
      help="Try exact terms (a name, an order id) and paraphrases. Evaluation questions show ✓/✗ relevance.",
    ),
  ],
)
def keyword_search(query: str):
  chunks = corpus.chunks()
  tokens = [corpus.tokenize(c.page_content) for c in chunks]
  terms = corpus.tokenize(query)
  overlap = [len(set(terms) & set(t)) for t in tokens]
  bm25 = BM25Okapi(tokens).get_scores(terms)
  question = corpus.question_for(query)

  def ranked(scores, label):
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:5]
    return results(
      [
        {
          "source": chunks[i].metadata["source"],
          "text": chunks[i].page_content,
          "score": round(float(scores[i]), 3),
          "matched": sorted(set(terms) & set(tokens[i])),
          "relevant": is_relevant(chunks[i].page_content, question) if question else None,
        }
        for i in order
        if scores[i] > 0
      ],
      title=label,
      score_label=label.split(" ")[0],
    )

  return [
    table(["Query terms (after removing stop words)"], [[", ".join(terms) or "(none)"]]),
    columns([ranked(overlap, "Overlap: words in common"), ranked(bm25, "BM25 score")], title="Top 5 chunks"),
    md(
      "**Tokenization decides what can match.** The text stores `it-help@nimbus.example` as one token, so the query "
      "*What is the IT help email address?* matches nothing useful: search for `it-help@nimbus.example` itself "
      "and it's found instantly. Production keyword search configures its analyzer (how text is split) per field.\n\n"
      "**BM25** improves on plain overlap in two ways: a word that's rare in the corpus (`it-help@nimbus.example`) "
      "counts far more than a common one (`nimbus`), and repeating a word in a long chunk gives diminishing returns."
    ),
  ] + _keyword_examples(query, chunks)


def _keyword_examples(query: str, chunks) -> list:
  """The "What it covers" tab: each learning point's code, run on this corpus."""

  def top_rows(scores, top):
    return [
      [
        rank,
        f"{chunks[i].metadata['source']} #{chunks[i].metadata['chunk_index']}",
        round(float(scores[i]), 2),
        chunks[i].page_content[:110].replace("\n", " ") + "…",
      ]
      for rank, i in enumerate(top, start=1)
    ]

  terms = split_into_terms(query)
  overlap = score_by_overlap(query, chunks)
  best_overlap = sorted(range(len(chunks)), key=lambda i: -overlap[i])[:3]
  bm25_scores, bm25_top = rank_with_bm25(query, chunks)
  exact_scores, exact_top = search_an_exact_term(chunks)
  para_scores, para_top = search_a_paraphrase(chunks)
  meaning = _semantic_rank("How many holidays do I get?", 1)[0][0]
  columns_ = ["Rank", "Chunk", "Score", "Text"]
  return [
    example(
      "rag.l02.1",
      "Split text into terms",
      split_into_terms,
      "Keyword search compares **terms**, so how you split text decides what can match: lower-case, keep emails "
      "and codes whole, drop stop words that appear everywhere.",
      [table(["Text", "Terms"], [[query, ", ".join(terms)]])],
    ),
    example(
      "rag.l02.1",
      "Score each chunk by words in common",
      score_by_overlap,
      "The simplest retrieval: count the query's terms in each chunk and rank by that count.",
      [table(columns_, top_rows(overlap, best_overlap), title=f"Top 3 for “{query}”")],
    ),
    example(
      "rag.l02.2",
      "BM25: the ranking search engines use",
      rank_with_bm25,
      "`BM25Okapi` learns each term's rarity from the corpus (IDF) and dampens repeats and long chunks. It's the "
      "default ranking in Elasticsearch, OpenSearch and Lucene.",
      [table(columns_, top_rows(bm25_scores, bm25_top), title=f"Top 3 for “{query}”")],
    ),
    example(
      "rag.l02.3",
      "Where it shines: exact terms",
      search_an_exact_term,
      "An email address, an order id or a product code is one rare term: BM25 puts the one chunk containing it "
      "first, with every other chunk at 0.",
      [table(columns_, top_rows(exact_scores, exact_top), title="Top 3 for “it-help@nimbus.example”")],
    ),
    example(
      "rag.l02.4",
      "Where it fails: same meaning, different words",
      search_a_paraphrase,
      "No chunk contains *holidays*, so BM25 ranks by the leftover words (*many*, *get*) and the leave policy "
      "(handbook.md) isn't first. "
      + (
        "Semantic search (next lab) ranks handbook.md first for the same question: it matches meaning, not words."
        if meaning.metadata["source"] == "handbook.md"
        else f"Semantic search (next lab) ranks **{meaning.metadata['source']}** first here: this small local "
        "embedding model misses it too. Neither method wins every query, which is why production systems combine "
        "them (hybrid search, Module 6) and measure on their own questions."
      ),
      [table(columns_, top_rows(para_scores, para_top), title="Top 3 for “How many holidays do I get?”")],
    ),
  ]


# ── Lab 3: semantic search ─────────────────────────────────────────────────────
@lab(
  "rag.semantic-search",
  "Semantic search: embeddings and similarity",
  inputs=[
    Input(
      "query",
      "Search query",
      "text",
      default="Can I work from home?",
      help="The handbook never says 'home' — it says 'remotely'.",
    ),
  ],
)
def semantic_search(query: str):
  chunks = corpus.chunks()
  vectors = corpus.chunk_vectors()
  q = np.array(corpus.embeddings().embed_query(query))
  cosine = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
  dot = vectors @ q
  euclid = np.linalg.norm(vectors - q, axis=1)
  question = corpus.question_for(query)
  order = np.argsort(-cosine)[:5]
  hits = [
    {
      "source": chunks[i].metadata["source"],
      "text": chunks[i].page_content,
      "score": round(float(cosine[i]), 3),
      "relevant": is_relevant(chunks[i].page_content, question) if question else None,
    }
    for i in order
  ]
  same = list(np.argsort(-cosine)[:5]) == list(np.argsort(-dot)[:5]) == list(np.argsort(euclid)[:5])
  return [
    results(hits, title="Top 5 by cosine similarity", score_label="cosine"),
    table(
      ["Chunk", "cosine (↑)", "dot product (↑)", "euclidean distance (↓)"],
      [
        [
          f"{chunks[i].metadata['source']} #{chunks[i].metadata['chunk_index']}",
          round(float(cosine[i]), 4),
          round(float(dot[i]), 4),
          round(float(euclid[i]), 4),
        ]
        for i in order
      ],
      title="Three similarity measures",
    ),
    callout(
      f"Vector length: {np.linalg.norm(q):.3f}. This model outputs **normalized** vectors (length 1), so cosine, "
      f"dot product and euclidean distance rank chunks in {'exactly the same' if same else 'a different'} order. "
      "With unnormalized vectors they can disagree: use the measure the model was trained for.",
      "info",
    ),
  ]


def _semantic_rank(query: str, k: int, size: int = 500, overlap: int = 80):
  chunks = corpus.chunks(size, overlap)
  vectors = corpus.chunk_vectors(size, overlap)
  q = np.array(corpus.embeddings().embed_query(query))
  scores = vectors @ q
  return [(chunks[i], float(scores[i])) for i in np.argsort(-scores)[:k]]


# ── Lab 4: retrieval metrics ───────────────────────────────────────────────────
def _loadable_questions() -> list[str]:
  """Questions whose source is a format v1 loads (decided by extension, so importing this module stays instant)."""
  return [q.question for q in corpus.questions() if Path(q.source).suffix.lower() in FILE_TYPES]


@lab(
  "rag.retrieval-metrics",
  "Retrieval metrics: hit, precision, recall, nDCG, MAP and MRR",
  inputs=[
    Input(
      "question", "Question", "select", default="Who approves remote work outside India?", options=_loadable_questions()
    ),
    Input("k", "Top k", "number", default=5, min=1, max=10),
  ],
)
def retrieval_metrics(question: str, k: int):
  """One question's ranking with every metric worked out by hand, then the k trade-off over all 45 questions."""
  texts = [c.page_content for c in corpus.chunks()]
  q = next(x for x in corpus.questions() if x.question == question)
  report = evaluate(_semantic_retriever, [q], texts, k=k)
  r = report.results[0]
  ranked = _semantic_rank(question, k)
  total, rel = r.total_relevant, r.relevant
  at = [i for i, x in enumerate(rel, start=1) if x]
  found = len(at)
  dcg = " + ".join(f"1/log2({i + 1})" for i in at) or "0"
  ideal = " + ".join(f"1/log2({i + 1})" for i in range(1, min(total, k) + 1)) or "0"
  precisions = " + ".join(f"{n}/{i}" for n, i in enumerate(at, start=1)) or "0"
  m = question_metrics(r, k)
  rows = [
    ["hit@k", "1 if any of the top k is relevant", f"relevant at ranks {at or 'none'}", m["hit"]],
    ["precision@k", "relevant in top k ÷ k", f"{found} ÷ {k}", m["precision"]],
    ["recall@k", "relevant in top k ÷ relevant in the whole index", f"{found} ÷ {total}", m["recall"]],
    ["nDCG@k", "DCG ÷ ideal DCG (all relevant first)", f"({dcg}) ÷ ({ideal})", m["ndcg"]],
    [
      "AP@k (→ MAP)",
      "precision at each relevant rank, summed ÷ min(relevant, k)",
      f"({precisions}) ÷ {min(total, k)}",
      m["map"],
    ],
    ["F1@k", "2·P·R ÷ (P + R)", f"P={m['precision']:.2f}, R={m['recall']:.2f}", m["f1"]],
    [
      "RR (→ MRR)",
      "1 ÷ rank of the first relevant",
      f"1 ÷ {at[0]}" if at else "nothing relevant: 0",
      reciprocal_rank(rel),
    ],
  ]
  views = [
    results(
      [
        {"source": c.metadata["source"], "text": c.page_content, "score": round(sim, 3), "relevant": is_rel}
        for (c, sim), is_rel in zip(ranked, rel, strict=True)
      ],
      title=f"Semantic search, top {k}: ✓ = contains the evidence ({', '.join(q.evidence)})",
      score_label="cosine",
    ),
    callout(
      (f"**{total}** chunks in the index contain" if total != 1 else "**1** chunk in the index contains")
      + " the whole answer. "
      + (
        "That number is the denominator of recall: return all of them and recall is 1."
        if total
        else "None: the answer is split across chunks or lives in a file v1 can't load, so every metric is 0."
      ),
      "info",
      "Relevant chunks in the whole index",
    ),
    table(
      ["Metric", "Formula", "This question", "Value"],
      [[a, b, c, f"{v:.3f}"] for a, b, c, v in rows],
      title=f"Every metric for this question at k = {k}",
    ),
  ]
  return views + _k_tradeoff(tuple(texts))


def _semantic_retriever(query: str, n: int):
  return [c for c, _ in _semantic_rank(query, n)]


@lru_cache
def _k_tradeoff(texts: tuple[str, ...]) -> list:
  """Mean metrics over all 45 questions at several k (search once at k = 10, then cut)."""
  cutoffs = (1, 2, 3, 5, 8, 10)
  report = evaluate(_semantic_retriever, corpus.questions(), texts, k=max(cutoffs))
  rows = []
  for c in cutoffs:
    scores = [question_metrics(r, c) for r in report.results]
    mean = {m: round(sum(s[m] for s in scores) / len(scores), 3) for m in ("hit", "precision", "recall", "ndcg", "f1")}
    tokens = sum(sum(t["chars"] for t in r.top[:c]) for r in report.results) / len(report.results) / 4
    rows.append({"variant": f"k = {c}", **mean, "tokens": round(tokens)})
  at = {row["variant"]: row for row in rows}
  k3, k5, k10 = at["k = 3"], at["k = 5"], at["k = 10"]
  best_f1 = max(rows, key=lambda row: row["f1"])
  answerable = [r for r in report.results if r.total_relevant]
  per_question = sum(r.total_relevant for r in answerable) / len(answerable) if answerable else 0
  return [
    scorecard(
      rows,
      [
        Metric("hit", "hit@k", "higher", "ratio", "Share of questions with at least one relevant chunk in the top k"),
        Metric("precision", "precision@k", "higher", "ratio", "Share of the k chunks that are relevant"),
        Metric("recall", "recall@k", "higher", "ratio", "Share of all relevant chunks that were returned"),
        Metric("ndcg", "nDCG@k", "higher", "ratio", "Ranking quality, 1 = perfect order"),
        Metric("f1", "F1@k", "higher", "ratio", "Harmonic mean of precision and recall"),
        Metric("tokens", "≈ context tokens", "lower", "int", "Average size of the retrieved context (chars ÷ 4)"),
      ],
      title="The k trade-off: all 45 questions, semantic search",
    ),
    callout(
      f"Hit rate and recall rise with k while precision falls, and every extra chunk costs tokens. From k = 3 to 5, "
      f"hit rate gains {k5['hit'] - k3['hit']:+.2f} for ≈{k5['tokens'] - k3['tokens']} more tokens per question; from "
      f"5 to 10 it gains {k10['hit'] - k5['hit']:+.2f}, while precision falls from {k5['precision']:.2f} to "
      f"{k10['precision']:.2f} and the context grows from ≈{k5['tokens']} to ≈{k10['tokens']} tokens. F1 peaks at "
      f"**{best_f1['variant']}** because an answerable question has only {per_question:.1f} relevant chunks on "
      "average: extra slots can only add noise. Pick k from this curve and your token budget, then let a reranker "
      "(Module 6) raise precision at that k.",
      "decision",
      "Choosing k",
    ),
  ]


# ── Lab 5: keyword vs semantic on the whole evaluation set ─────────────────────
@lab(
  "rag.keyword-vs-semantic",
  "Compare: keyword vs semantic search on 45 real questions",
  inputs=[
    Input("k", "Results per question (k)", "number", default=5, min=1, max=10),
  ],
)
def keyword_vs_semantic(k: int):
  chunks = corpus.chunks()
  texts = [c.page_content for c in chunks]
  bm25 = BM25Okapi([corpus.tokenize(c.page_content) for c in chunks])

  def keyword(query: str, n: int):
    scores = bm25.get_scores(corpus.tokenize(query))
    return [chunks[i] for i in np.argsort(-scores)[:n] if scores[i] > 0]

  indexed = corpus.indexed_sources()
  reports = {
    "Keyword (BM25)": evaluate(keyword, corpus.questions(), texts, k=k, indexed_sources=indexed),
    "Semantic (embeddings)": evaluate(_semantic_retriever, corpus.questions(), texts, k=k, indexed_sources=indexed),
  }
  rows = [
    {
      "variant": name,
      "hit@1": r.at["hit"][1],
      f"hit@{k}": r.at["hit"][k],
      f"precision@{k}": r.at["precision"][k],
      f"recall@{k}": r.at["recall"][k],
      f"ndcg@{k}": r.at["ndcg"][k],
      "mrr": r.mrr,
      "latency_p50_ms": round(r.latency_p50_ms, 1),
      "latency_p95_ms": round(r.latency_p95_ms, 1),
    }
    for name, r in reports.items()
  ]
  kinds = sorted(next(iter(reports.values())).by_kind)
  reachable = [r for r in reports["Keyword (BM25)"].results if r.reachable]
  sem = {r.question.id: r for r in reports["Semantic (embeddings)"].results}
  only_kw = [r for r in reachable if r.rank and not sem[r.question.id].rank]
  only_sem = [r for r in reachable if not r.rank and sem[r.question.id].rank]
  coverage = reports["Keyword (BM25)"].coverage
  metrics = [Metric("hit@1", "hit@1", "higher", "ratio", "Share of questions answered by the first chunk")]
  if k != 1:
    metrics.append(Metric(f"hit@{k}", f"hit@{k}", "higher", "ratio", "Share with a relevant chunk in the top k"))
  metrics += [
    Metric(f"precision@{k}", f"precision@{k}", "higher", "ratio", "Share of the k chunks that are relevant"),
    Metric(f"recall@{k}", f"recall@{k}", "higher", "ratio", "Share of all relevant chunks that were returned"),
    Metric(f"ndcg@{k}", f"nDCG@{k}", "higher", "ratio", "Ranking quality, 1 = perfect order"),
    Metric("mrr", "MRR", "higher", "ratio", "Mean of 1/rank of the first relevant chunk"),
    Metric("latency_p50_ms", "p50", "lower", "ms"),
    Metric("latency_p95_ms", "p95", "lower", "ms"),
  ]
  return [
    scorecard(rows, metrics, title=f"45 questions, top {k}"),
    bars(
      [{"label": kind, "values": {name: r.by_kind[kind]["hit"] for name, r in reports.items()}} for kind in kinds],
      title=f"hit@{k} by question type",
      max_value=1,
    ),
    table(
      ["Question", "type"],
      [[r.question.question, r.question.kind] for r in only_kw],
      title=f"Found by keyword only ({len(only_kw)})",
    ),
    table(
      ["Question", "type"],
      [[r.question.question, r.question.kind] for r in only_sem],
      title=f"Found by semantic only ({len(only_sem)})",
    ),
    callout(
      f"Only {coverage:.0%} of the questions are answerable at all: the rest live in files v1 can't load "
      "(Word, Excel, slides, email, scans, images). Hit rate can't beat coverage. Module 2 fixes that.",
      "warning",
      "Coverage caps everything",
    ),
    callout(
      _verdict(
        rows, k, len(only_kw), len(only_sem), max(r.by_kind.get("table", {}).get("hit", 0) for r in reports.values())
      ),
      "decision",
      "What the numbers say",
    ),
  ]


def _verdict(rows: list[dict], k: int, only_kw: int, only_sem: int, best_table_hit: float) -> str:
  """The conclusion, computed from this run's numbers (so it's never out of date)."""
  kw, sem = rows
  best = kw if kw[f"hit@{k}"] >= sem[f"hit@{k}"] else sem
  other = sem if best is kw else kw
  return (
    f"On this corpus **{best['variant']}** finds more answers: hit@{k} {best[f'hit@{k}']:.2f} vs "
    f"{other[f'hit@{k}']:.2f}, MRR {best['mrr']:.2f} vs {other['mrr']:.2f}, precision@{k} "
    f"{best[f'precision@{k}']:.2f} vs {other[f'precision@{k}']:.2f}. Yet each finds questions the other "
    f"misses ({only_kw} keyword-only, {only_sem} semantic-only)"
    + (f", and both struggle with tables (best hit rate {best_table_hit:.2f})" if best_table_hit < 0.5 else "")
    + ". Don't assume embeddings "
    "beat keywords: measure on your own questions. Production systems run both and fuse the rankings "
    "(hybrid search, Module 6)."
  )
