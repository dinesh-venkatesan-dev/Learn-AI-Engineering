"""RAG · Module 1 labs: why RAG exists, the two ways to search text, and how they compare on real questions."""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
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
  radar,
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
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.
def ask_without_context(model, question):
  from langchain_core.messages import HumanMessage, SystemMessage

  reply = model.invoke([SystemMessage(ANSWER_RULES), HumanMessage(question)])  # only what it learned in training
  return reply


def stuff_everything(model, question):
  from langchain_core.messages import HumanMessage, SystemMessage

  from academy.labs.rag.corpus import documents

  everything = "\n\n---\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in documents())  # all of it
  reply = model.invoke([SystemMessage(f"{ANSWER_RULES}\n\nDocuments:\n{everything}"), HumanMessage(question)])
  return reply


def cost_of(reply):
  usage = reply.usage_metadata  # the tokens the provider billed for this call
  dollars = (usage["input_tokens"] * PRICE_IN + usage["output_tokens"] * PRICE_OUT) / 1_000_000
  return dollars


def retrieve_then_answer(model, question):
  from langchain_core.messages import HumanMessage, SystemMessage

  from academy.labs.rag.corpus import chunk_vectors, chunks

  scores, top = rank_by_meaning(question, chunks(), chunk_vectors())  # semantic search (lab 3), top 3
  relevant = "\n\n---\n\n".join(f"[{chunks()[i].metadata['source']}]\n{chunks()[i].page_content}" for i in top)
  reply = model.invoke([SystemMessage(f"{ANSWER_RULES}\n\nDocuments:\n{relevant}"), HumanMessage(question)])
  return reply


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
  attempts = [
    ("No context (model's own knowledge)", ask_without_context),
    ("Stuff everything (all documents)", stuff_everything),
    ("RAG (top 3 relevant chunks)", retrieve_then_answer),
  ]
  expected = WHY_RAG_QUESTIONS[question]
  rows, answers, replies = [], [], []
  for label, ask in attempts:
    started = time.perf_counter()
    reply = ask(model, question)
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
    replies.append(reply)
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
  ] + _why_rag_examples(question, rows, replies)


def _why_rag_examples(question: str, rows: list[dict], replies: list) -> list:
  """The "What it covers" tab: the three calls the lab just made, one learning point each."""
  bare, stuffed, rag = rows
  ratio = stuffed["input_tokens"] / rag["input_tokens"] if rag["input_tokens"] else 0
  per_day = [
    [r["variant"], r["input_tokens"], f"${cost_of(reply) * 1000:.2f}"] for r, reply in zip(rows, replies, strict=True)
  ]

  def said(row, reply) -> list:
    return [table(["Answer", "Correct?", "Input tokens"], [[reply.text, row["correct"], row["input_tokens"]]])]

  return [
    example(
      "rag.l01.1",
      "Ask the model on its own",
      ask_without_context,
      "Nimbus's handbook, prices and figures were never in the model's training data. "
      + (
        "It said it doesn't know, which is the honest outcome; a weaker prompt often gets a confident guess."
        if bare["correct"] == "✗"
        else "It got this one right anyway: a lucky or generic guess, which you can't rely on for private data."
      ),
      said(bare, replies[0]),
    ),
    example(
      "rag.l01.2",
      "Context stuffing: paste every document into the prompt",
      stuff_everything,
      "Give the model the documents and it can answer: "
      + ("it did." if stuffed["correct"] == "✓" else "here it still missed, lost in everything else.")
      + f" The price: {stuffed['input_tokens']:,} input tokens for one short question.",
      said(stuffed, replies[1]),
    ),
    example(
      "rag.l01.3",
      "Why stuffing doesn't scale",
      cost_of,
      f"Every question pays for every token in the corpus: stuffing sent {ratio:.0f}× the tokens RAG did. The cost "
      "grows with the corpus and with traffic, and once the corpus outgrows the context window it can't be sent at "
      "all. Long contexts also make answers less reliable; this lab measures the cost side.",
      [table(["Variant", "Input tokens", "$ per 1,000 questions"], per_day)],
    ),
    example(
      "rag.l01.4",
      "RAG: retrieve the relevant chunks, then answer",
      retrieve_then_answer,
      f"Search first, send only the top 3 chunks: {rag['input_tokens']:,} input tokens instead of "
      f"{stuffed['input_tokens']:,}"
      + (
        ", and the same correct answer."
        if rag["correct"] == stuffed["correct"] == "✓"
        else f". Correct: {rag['correct']}."
      ),
      said(rag, replies[2]),
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
    *_meaning_map(chunks, vectors, q, cosine),
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
  ] + _semantic_examples(query, chunks, vectors)


def embed_text(text):
  from academy.labs.rag.corpus import embeddings

  vector = embeddings().embed_query(text)  # BAAI/bge-small-en-v1.5, running locally: no API key, no cost
  return vector


def three_similarities(a, b):
  import numpy as np

  a, b = np.array(a), np.array(b)
  cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))  # the angle between them: 1 = same direction
  dot = a @ b  # cosine × both lengths: the same as cosine when vectors have length 1
  euclidean = np.linalg.norm(a - b)  # straight-line distance: LOWER means closer
  return cosine, dot, euclidean


def rank_by_meaning(query, chunks, vectors):
  import numpy as np

  from academy.labs.rag.corpus import embeddings

  q = np.array(embeddings().embed_query(query))  # the chunks were embedded once, at indexing time
  scores = vectors @ q  # one similarity per chunk, all at once
  top = np.argsort(-scores)[:3]
  return scores, top


def keyword_vs_meaning(query, chunks, vectors):
  bm25_scores, bm25_top = rank_with_bm25(query, chunks)  # from the keyword lab
  semantic_scores, semantic_top = rank_by_meaning(query, chunks, vectors)
  return chunks[bm25_top[0]], chunks[semantic_top[0]]


SIMILARITY_PAIRS = [
  "Employees may work remotely for up to two days a week.",  # same meaning, different words
  "The office kitchen is cleaned every Friday.",  # same topic (the workplace), different meaning
  "Our cold brew is steeped for eighteen hours.",  # unrelated
]


def _semantic_examples(query: str, chunks, vectors) -> list:
  """The "What it covers" tab: each learning point's code, run on this corpus."""
  question = corpus.question_for(query)

  def label(chunk) -> str:
    return f"{chunk.metadata['source']} #{chunk.metadata['chunk_index']}"

  def mark(chunk, q) -> str:
    return "" if q is None else (" ✓" if is_relevant(chunk.page_content, q) else " ✗")

  vector = embed_text(query)
  pairs = [(text, *three_similarities(vector, embed_text(text))) for text in SIMILARITY_PAIRS]
  scores, top = rank_by_meaning(query, chunks, vectors)
  # One question each where only one method's first result holds the answer, picked from the evaluation set
  firsts = {q.id: keyword_vs_meaning(q.question, chunks, vectors) for q in corpus.questions()}
  found = {
    q.id: (is_relevant(firsts[q.id][0].page_content, q), is_relevant(firsts[q.id][1].page_content, q))
    for q in corpus.questions()
  }
  cases = [
    ("only semantic", next((q for q in corpus.questions() if q.kind == "paraphrase" and found[q.id] == (0, 1)), None)),
    (
      "only keyword",
      next((q for q in corpus.questions() if q.kind in ("number", "exact") and found[q.id] == (1, 0)), None),
    ),
    ("only keyword", next((q for q in corpus.questions() if q.kind == "paraphrase" and found[q.id] == (1, 0)), None)),
  ]
  duel = [
    [
      who,
      q.kind,
      q.question,
      label(firsts[q.id][0]) + mark(firsts[q.id][0], q),
      label(firsts[q.id][1]) + mark(firsts[q.id][1], q),
    ]
    for who, q in cases
    if q is not None
  ]
  split = sum(1 for a, b in found.values() if a != b)
  return [
    example(
      "rag.l03.1",
      "Turn text into a vector",
      embed_text,
      "An embedding model maps text to a fixed-length list of numbers. Texts with similar meaning land close "
      "together, whatever words they use. The numbers themselves mean nothing on their own: only distances do.",
      [
        table(
          ["Text", "Dimensions", "First 6 numbers", "Length"],
          [[query, len(vector), ", ".join(f"{x:+.3f}" for x in vector[:6]) + " …", f"{np.linalg.norm(vector):.3f}"]],
        )
      ],
    ),
    example(
      "rag.l03.2",
      "Cosine, dot product, euclidean distance",
      three_similarities,
      "Three ways to say how close two vectors are. Cosine and dot product: higher = closer. Euclidean distance: "
      "lower = closer. This model's vectors have length 1, so all three agree on the order.",
      [
        table(
          ["Compared with your query", "cosine (↑)", "dot (↑)", "euclidean (↓)"],
          [[text, round(float(c), 3), round(float(d), 3), round(float(e), 3)] for text, c, d, e in pairs],
          title=f"“{query}” vs three sentences",
        )
      ],
    ),
    example(
      "rag.l03.3",
      "Semantic search: rank chunks by similarity",
      rank_by_meaning,
      "Embed the question, score it against every chunk's vector, keep the highest. Chunk vectors are computed "
      "once when the documents are indexed; each search only embeds the question.",
      [
        table(
          ["Rank", "Chunk", "Similarity", "Text"],
          [
            [
              rank,
              label(chunks[i]) + mark(chunks[i], question),
              round(float(scores[i]), 3),
              chunks[i].page_content[:110].replace("\n", " ") + "…",
            ]
            for rank, i in enumerate(top, start=1)
          ],
          title=f"Top 3 for “{query}”",
        )
      ],
    ),
    example(
      "rag.l03.4",
      "The trade-off: meaning vs exact terms",
      keyword_vs_meaning,
      f"On {split} of the 45 evaluation questions, only one method's first result holds the answer (✓). Three of "
      "them, picked to show the pattern: embeddings find the paraphrase whose words don't match, keywords find "
      "the exact number or term the embedding blurs. The third row is the exception: a paraphrase only keyword "
      "search gets right. Neither wins every time; the last lab of this module measures both on all 45.",
      [table(["Found by", "Type", "Question", "Keyword (BM25) top 1", "Semantic top 1"], duel)],
    ),
  ]


def _meaning_map(chunks, vectors, q, cosine, k: int = 5) -> list:
  """The query at the centre and every chunk at its TRUE distance (1 - cosine) from it, so the k innermost dots are
  exactly the top k. Only the direction comes from a 2-D projection (PCA), which keeps similar chunks together."""
  centred = vectors - vectors.mean(axis=0)
  _, strengths, axes = np.linalg.svd(centred, full_matrices=False)
  flat = centred @ axes[:2].T
  angle = np.arctan2(flat[:, 1], flat[:, 0])
  distance = 1 - cosine
  rank = {int(i): r for r, i in enumerate(np.argsort(-cosine)[:k], start=1)}
  points = [
    {
      "x": round(float(distance[i] * np.cos(angle[i])), 4),
      "y": round(float(distance[i] * np.sin(angle[i])), 4),
      "label": f"{c.metadata['source']} #{c.metadata['chunk_index']} · similarity {cosine[i]:.3f}",
      "text": c.page_content,
      "rank": rank.get(i),
    }
    for i, c in enumerate(chunks)
  ]
  low, high = float(cosine.min()), float(cosine.max())
  rings = [round(1 - s / 10, 2) for s in range(int(np.ceil(low * 10)), int(np.floor(high * 10)) + 1)]
  kept = float((strengths[:2] ** 2).sum() / (strengths**2).sum())
  return [
    radar(points, rings, title=f"Map of meaning: {len(chunks)} chunks around your query"),
    callout(
      f"Distance from the centre is exact: 1 − cosine similarity, so the {k} innermost dots are exactly the top {k} "
      f"above. Similarities run from {low:.2f} to {high:.2f}: even an unrelated chunk is never at 0, so what matters "
      "is the ranking, not the number. The direction is only approximate: squeezing 384 dimensions into a flat "
      f"picture keeps {kept:.0%} of the variation between chunks, so two dots side by side may still be about "
      "different things.",
      "info",
      "Reading the map",
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


# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.
def hit_at_k(relevant, k):
  hit = 1.0 if any(relevant[:k]) else 0.0  # relevant[i]: does the chunk at rank i+1 contain the answer?
  return hit


def mean_reciprocal_rank(results):
  firsts = [next((i for i, rel in enumerate(r.relevant, start=1) if rel), None) for r in results]
  mrr = sum(1 / rank for rank in firsts if rank) / len(firsts)  # a question with no relevant chunk adds 0
  return mrr


def precision_at_k(relevant, k):
  precision = sum(relevant[:k]) / k  # divide by k: an irrelevant or empty slot still costs tokens
  return precision


def recall_at_k(relevant, total_relevant, k):
  recall = sum(relevant[:k]) / total_relevant if total_relevant else 0.0  # out of ALL relevant chunks
  return recall


def count_relevant_chunks(question, texts):
  from nimbus.rag.evaluation import normalize

  evidence = [normalize(e) for e in question.evidence]  # the facts the answer needs, labelled once per question
  total = sum(1 for text in texts if all(e in normalize(text) for e in evidence))  # chunks holding ALL of them
  return total


def answerability(questions, texts):
  answerable = [q for q in questions if count_relevant_chunks(q, texts) > 0]
  share = len(answerable) / len(questions)  # the ceiling for every retriever on this index
  return share, answerable


def ndcg_at_k(relevant, total_relevant, k):
  import math

  # each relevant chunk scores 1/log2(rank + 1): rank 1 → 1, rank 2 → 0.63, rank 3 → 0.5
  dcg = sum(1 / math.log2(rank + 1) for rank, rel in enumerate(relevant[:k], start=1) if rel)
  ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))  # all relevant first
  ndcg = dcg / ideal if ideal else 0.0
  return ndcg


def average_precision_at_k(relevant, total_relevant, k):
  found, total = 0, 0.0
  for rank, rel in enumerate(relevant[:k], start=1):
    if rel:
      found += 1
      total += found / rank  # precision at this relevant chunk
  ap = total / min(total_relevant, k) if total_relevant else 0.0  # MAP = the mean of this over all questions
  return ap


def f1_at_k(precision, recall):
  f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0  # low if either is low
  return f1


def metrics_at_cutoffs(results, cutoffs):
  from nimbus.rag.evaluation import question_metrics

  curve = {}
  for k in cutoffs:  # search once at the largest k, then cut the same ranking shorter
    scores = [question_metrics(r, k) for r in results]
    curve[k] = {m: sum(s[m] for s in scores) / len(scores) for m in ("hit", "precision", "recall")}
  return curve


def report_slices(report):
  cutoffs = {c: report.at["hit"][c] for c in (1, 3, 5)}  # several cutoffs, not just one
  by_type = {kind: s["hit"] for kind, s in report.by_kind.items()}  # WHERE it fails, not just how often
  by_source = {source: s["hit"] for source, s in report.by_source.items()}
  return cutoffs, by_type, by_source


METRIC_FOR = {
  "Loading, chunking, first-stage retrieval": ["hit", "recall"],  # is the answer in there at all?
  "Reranking and choosing k": ["precision", "ndcg", "mrr"],  # is it near the top, with little noise?
  "Loaders and OCR": ["coverage", "answerable"],  # could any chunk answer it?
}


def metric_for(decision, report):
  values = {"mrr": report.mrr, "coverage": report.coverage, "answerable": report.answerable}
  chosen = {m: values.get(m, report.at.get(m, {}).get(5)) for m in METRIC_FOR[decision]}
  return chosen


@lru_cache
def _all_questions(texts: tuple[str, ...], k: int):
  """Semantic search on all 45 questions at this k (cached: every metric example reads from it)."""
  return evaluate(_semantic_retriever, corpus.questions(), texts, k=k, indexed_sources=corpus.indexed_sources())


def _metric_examples(q, k: int, r, texts: tuple[str, ...]) -> list:
  """The "What it covers" tab for the metrics lab: this question by hand, then all 45."""
  rel, total = r.relevant, r.total_relevant
  marks = " ".join("✓" if x else "✗" for x in rel)
  report = _all_questions(texts, max(k, 5))
  firsts = report.results[:6]
  share, answerable = answerability(corpus.questions(), texts)
  missing = [x for x in corpus.questions() if x not in answerable]
  indexed = corpus.indexed_sources()
  not_loaded = sum(1 for x in missing if x.source not in indexed)
  totals = [count_relevant_chunks(x, texts) for x in corpus.questions()]
  spread = {
    label: sum(1 for n in totals if test(n))
    for label, test in (
      ("0", lambda n: n == 0),
      ("1", lambda n: n == 1),
      ("2", lambda n: n == 2),
      ("3 or more", lambda n: n >= 3),
    )
  }
  p, rc = precision_at_k(rel, k), recall_at_k(rel, total, k)
  curve = metrics_at_cutoffs(_all_questions(texts, 10).results, (1, 3, 5, 10))
  cutoffs, by_type, by_source = report_slices(report)
  this = [["Ranks 1…" + str(k), marks]]
  return [
    example(
      "new.l02.1",
      "hit@k: is the answer in the top k at all?",
      hit_at_k,
      "The most basic question: could the model answer from what was retrieved? 1 or 0 per question; the mean over "
      "questions is the hit rate.",
      [table(["This question", "Relevant?"], this), table(["k", "hit@k"], [[c, hit_at_k(rel, c)] for c in (1, 3, k)])],
    ),
    example(
      "new.l02.2",
      "MRR: how high is the first relevant chunk?",
      mean_reciprocal_rank,
      "1/rank of the first relevant chunk: 1 at rank 1, 0.5 at rank 2, 0.33 at rank 3, 0 if none. MRR is its mean "
      "over questions: it rewards putting the answer FIRST.",
      [
        table(
          ["Question", "First relevant at", "1/rank"],
          [[x.question.question, x.rank or "—", round(1 / x.rank, 2) if x.rank else 0] for x in firsts],
          title="The first 6 of 45 questions",
        ),
        table(["MRR over all 45 questions"], [[round(mean_reciprocal_rank(report.results), 3)]]),
      ],
    ),
    example(
      "new.l02.3",
      "precision@k: how much of what we send is useful?",
      precision_at_k,
      "Of the k chunks sent to the model, the share that is relevant. The rest is noise: it costs tokens and can "
      "mislead the model.",
      [table(["This question", "Relevant?", f"precision@{k}"], [[this[0][0], marks, round(p, 3)]])],
    ),
    example(
      "new.l02.4",
      "recall@k: how much of all the evidence did we get?",
      recall_at_k,
      "Of ALL the chunks in the index that hold the answer, the share that was returned. It matters when an answer "
      "is spread over several chunks.",
      [table(["Relevant in top k", "Relevant in the index", f"recall@{k}"], [[sum(rel[:k]), total, round(rc, 3)]])],
    ),
    example(
      "new.l02.5",
      "Count every relevant chunk in the index",
      count_relevant_chunks,
      "Recall and nDCG divide by the number of relevant chunks in the WHOLE index. Labelling each question once with "
      "its evidence (the facts the answer needs) lets code count them for any chunking, without labelling chunks "
      "by hand.",
      [
        table(["Question", "Evidence", "Relevant chunks"], [[q.question, ", ".join(q.evidence), total]]),
        table(
          ["Relevant chunks per question", "Questions"], [[a, b] for a, b in spread.items()], title="All 45 questions"
        ),
      ],
    ),
    example(
      "new.l02.6",
      "Answerability: can ANY chunk answer it?",
      answerability,
      f"{share:.0%} of the 45 questions have at least one chunk holding the whole answer: no retriever can do "
      f"better on this index. Of the {len(missing)} others, {not_loaded} are in files Module 1 can't load yet "
      f"(Module 2 adds them) and {len(missing) - not_loaded} have their answer split across chunks.",
      [
        table(
          ["Unanswerable question", "Why"],
          [
            [x.question, "file not loaded yet" if x.source not in indexed else "answer split across chunks"]
            for x in missing[:8]
          ],
          title=f"{len(missing)} unanswerable (first 8)",
        )
      ],
    ),
    example(
      "new.l02.7",
      "nDCG@k: relevant chunks count more the higher they rank",
      ndcg_at_k,
      "Each relevant chunk scores 1/log2(rank+1): 1 at rank 1, 0.63 at rank 2, 0.5 at rank 3. Divided by the best "
      "possible score, 1 means a perfect order. The standard metric for comparing rankers and rerankers.",
      [table(["This question", "Relevant?", f"nDCG@{k}"], [[this[0][0], marks, round(ndcg_at_k(rel, total, k), 3)]])],
    ),
    example(
      "new.l02.8",
      "MAP@k: precision at every relevant rank",
      average_precision_at_k,
      "Average precision takes precision at each rank where a relevant chunk appears and averages them; MAP is the "
      "mean over questions.",
      [
        table(
          ["This question: AP@" + str(k), "MAP@5 over 45 questions"],
          [[round(average_precision_at_k(rel, total, k), 3), round(report.at["map"][5], 3)]],
        )
      ],
    ),
    example(
      "new.l02.9",
      "F1@k: precision and recall in one number",
      f1_at_k,
      "The harmonic mean is low when either side is low, so a retriever can't score well by returning everything "
      "(high recall, low precision) or one safe chunk (the reverse).",
      [table(["precision", "recall", "F1"], [[round(p, 3), round(rc, 3), round(f1_at_k(p, rc), 3)]])],
    ),
    example(
      "new.l02.10",
      "The k trade-off",
      metrics_at_cutoffs,
      "As k grows, hit rate and recall rise while precision falls, and every extra chunk costs context tokens. "
      "Choose k from this curve and your token budget (the scorecard on the Run tab adds tokens and F1).",
      [
        table(
          ["k", "hit", "precision", "recall"],
          [[c, round(v["hit"], 3), round(v["precision"], 3), round(v["recall"], 3)] for c, v in curve.items()],
          title="All 45 questions",
        )
      ],
    ),
    example(
      "new.l02.11",
      "Report several cutoffs and slices",
      report_slices,
      "One number hides where retrieval fails. Several cutoffs show how far down the answer sits; slices by question "
      "type and source show what to fix. Every number is a mean over questions, so each question counts equally.",
      [
        table(["hit@1", "hit@3", "hit@5"], [[round(cutoffs[c], 3) for c in (1, 3, 5)]]),
        table(["Question type", "hit@5"], [[a, round(b, 3)] for a, b in sorted(by_type.items())]),
        table(["Source file", "hit@5"], [[a, round(b, 3)] for a, b in sorted(by_source.items(), key=lambda x: x[1])]),
      ],
    ),
    example(
      "new.l02.12",
      "Which metric for which decision",
      metric_for,
      "Each decision has its own question, so its own metric. These are this run's values (semantic search, "
      "all 45 questions, @5).",
      [
        table(
          ["Decision", "Watch", "Now"],
          [
            [d, ", ".join(ms), ", ".join(f"{m} {v:.2f}" for m, v in metric_for(d, report).items())]
            for d, ms in METRIC_FOR.items()
          ],
        )
      ],
    ),
  ]


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
  return views + _k_tradeoff(tuple(texts)) + _metric_examples(q, k, r, tuple(texts))


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
def make_keyword_retriever(chunks):
  import numpy as np
  from rank_bm25 import BM25Okapi

  from academy.labs.rag.corpus import tokenize

  bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])  # built once, like an index

  def keyword(query, k):
    scores = bm25.get_scores(tokenize(query))
    return [chunks[i] for i in np.argsort(-scores)[:k] if scores[i] > 0]  # no shared word, no result

  return keyword


def shared_terms(question, answer_chunk):
  from academy.labs.rag.corpus import tokenize

  shared = sorted(set(tokenize(question)) & set(tokenize(answer_chunk.page_content)))  # all BM25 can match on
  return shared


def hit_rate_by_type(reports):
  rates = {}
  for name, report in reports.items():  # one evaluation report per retriever, same 45 questions
    for kind, scores in report.by_kind.items():
      rates.setdefault(kind, {})[name] = scores["hit"]
  return rates


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
  keyword = make_keyword_retriever(chunks)
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
  ] + _comparison_examples(reports, chunks, k)


def _comparison_examples(reports: dict, chunks, k: int) -> list:
  """The "What it covers" tab: each learning point's code, run on all 45 questions."""
  kw_name, sem_name = reports
  by_type = hit_rate_by_type(reports)
  sem_rank = {r.question.id: r.rank for r in reports[sem_name].results}
  exact_rows = [
    [kind, by_type[kind][kw_name], by_type[kind][sem_name]] for kind in ("exact", "number") if kind in by_type
  ]
  kw_ahead = sum(1 for _, kw, sem in exact_rows if kw > sem)
  level = sum(1 for _, kw, sem in exact_rows if kw == sem)
  misses = []
  for r in reports[kw_name].results:
    q = r.question
    answer_chunk = next((c for c in chunks if is_relevant(c.page_content, q)), None)
    if q.kind == "paraphrase" and r.rank is None and answer_chunk is not None:
      shared = shared_terms(q.question, answer_chunk)
      rank = sem_rank[q.id]
      misses.append([q.question, ", ".join(shared) or "(none)", f"#{rank}" if rank else f"not in top {k}"])
  para = by_type.get("paraphrase", {})
  return [
    example(
      "rag.l02.3",
      "Where keyword search shines: exact terms and numbers",
      make_keyword_retriever,
      f"hit@{k} on the questions that ask for an exact term or a number. Keyword search is ahead on {kw_ahead}, "
      f"level on {level} and behind on {len(exact_rows) - kw_ahead - level} of these {len(exact_rows)} types. A rare "
      "term such as a code, a name or a figure scores highly in BM25, while an embedding blurs it into the topic.",
      [table(["Question type", kw_name, sem_name], [[a, f"{b:.2f}", f"{c:.2f}"] for a, b, c in exact_rows])],
    ),
    example(
      "rag.l02.4",
      "Where it fails: same meaning, different words",
      shared_terms,
      f"Paraphrased questions keyword search missed in its top {k}, with the only words they share with the chunk "
      f"that holds the answer. On this corpus that's the exception, not the rule: keyword search found "
      f"{para.get(kw_name, 0):.0%} of the paraphrases (semantic {para.get(sem_name, 0):.0%}), because most still "
      "share a distinctive word with the answer. It fails when they share none.",
      [table(["Paraphrased question", "Words shared with the answer", "Semantic search found it at"], misses)],
    ),
    example(
      "rag.l03.4",
      "The trade-off, measured by question type",
      hit_rate_by_type,
      f"hit@{k} per question type for both retrievers. Which one wins depends on the type and on the corpus, so "
      "measure on your own questions before choosing. Module 6 runs both and fuses their rankings (hybrid search).",
      [
        table(
          ["Question type", kw_name, sem_name],
          [[kind, f"{v[kw_name]:.2f}", f"{v[sem_name]:.2f}"] for kind, v in sorted(by_type.items())],
        )
      ],
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
