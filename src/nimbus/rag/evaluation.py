"""Retrieval evaluation: does search put the chunks that answer each question in the top k, and how high?

Every retrieval decision in Nimbus (chunk size, splitter, embedding model, hybrid search, reranker) is made with these
numbers, not by eye. Per question, then averaged over the evaluation set, at cutoffs k = 1, 3, 5:

  hit@k        did at least one relevant chunk reach the top k? (can the model answer at all?)
  precision@k  share of the k returned chunks that are relevant (the rest is noise the model must read)
  recall@k     share of ALL relevant chunks in the index that were returned (matters when an answer spans chunks)
  nDCG@k       ranking quality: relevant chunks count more the higher they rank, scaled so a perfect ranking is 1
  MAP@k        mean average precision: precision measured at each relevant chunk, averaged
  F1@k         harmonic mean of precision@k and recall@k
  MRR          mean of 1/rank of the first relevant chunk (how high did we rank it?)
  coverage     share of questions whose source document is in the index at all (a file we can't load is invisible)
  answerable   share of questions where some indexed chunk contains the whole answer (chunking can split it apart)
  latency      p50 / p95 per search (what users wait)

Relevance is judged by *evidence*: each question lists the text that answers it, and a chunk is relevant when it
contains all of it. This stays fair when chunking changes, unlike labels tied to chunk ids, and it lets us count the
relevant chunks in the whole index, which recall and nDCG need.
"""

import json
import math
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

Retriever = Callable[[str, int], list[Document]]


@dataclass(frozen=True)
class EvalQuestion:
  id: str
  question: str
  evidence: tuple[str, ...]  # all of these must appear in a chunk for it to count as relevant
  source: str  # the file that answers it
  kind: str  # exact | number | paraphrase | table | general


def load_questions(path: Path) -> list[EvalQuestion]:
  return [
    EvalQuestion(q["id"], q["question"], tuple(q["evidence"]), q["source"], q["kind"])
    for q in json.loads(path.read_text())
  ]


def normalize(text: str) -> str:
  """Compare text the way a reader would: case, markdown bold and spacing don't matter."""
  return re.sub(r"\s+", " ", text.replace("**", "").replace(" ", " ")).casefold().strip()


def is_relevant(chunk_text: str, question: EvalQuestion) -> bool:
  text = normalize(chunk_text)
  return all(normalize(e) in text for e in question.evidence)


@dataclass
class QuestionResult:
  question: EvalQuestion
  relevant: list[bool]  # for each returned chunk, in rank order: does it contain the answer?
  total_relevant: int  # relevant chunks in the whole index (0 when no chunk holds the whole answer)
  latency_ms: float
  reachable: bool  # is the source file in the index at all?
  top: list[dict] = field(default_factory=list)

  @property
  def rank(self) -> int | None:
    """1-based rank of the first relevant chunk, None if none was returned."""
    return next((i for i, rel in enumerate(self.relevant, start=1) if rel), None)


# ── The metrics for one question at cutoff k (binary relevance) ──────────────────
def hit_at(relevant: list[bool], k: int) -> float:
  return 1.0 if any(relevant[:k]) else 0.0


def precision_at(relevant: list[bool], k: int) -> float:
  return sum(relevant[:k]) / k  # divide by k, not by what came back: an empty slot is a wasted slot


def recall_at(relevant: list[bool], total: int, k: int) -> float:
  return sum(relevant[:k]) / total if total else 0.0


def ndcg_at(relevant: list[bool], total: int, k: int) -> float:
  dcg = sum(1 / math.log2(i + 1) for i, rel in enumerate(relevant[:k], start=1) if rel)
  ideal = sum(1 / math.log2(i + 1) for i in range(1, min(total, k) + 1))  # all relevant chunks first
  return dcg / ideal if ideal else 0.0


def average_precision_at(relevant: list[bool], total: int, k: int) -> float:
  found, score = 0, 0.0
  for i, rel in enumerate(relevant[:k], start=1):
    if rel:
      found += 1
      score += found / i  # precision at this relevant chunk
  return score / min(total, k) if total else 0.0


def f1_at(relevant: list[bool], total: int, k: int) -> float:
  p, r = precision_at(relevant, k), recall_at(relevant, total, k)
  return 2 * p * r / (p + r) if p + r else 0.0


def reciprocal_rank(relevant: list[bool]) -> float:
  return next((1 / i for i, rel in enumerate(relevant, start=1) if rel), 0.0)


METRICS = ("hit", "precision", "recall", "ndcg", "map", "f1")


def question_metrics(r: QuestionResult, k: int) -> dict[str, float]:
  return {
    "hit": hit_at(r.relevant, k),
    "precision": precision_at(r.relevant, k),
    "recall": recall_at(r.relevant, r.total_relevant, k),
    "ndcg": ndcg_at(r.relevant, r.total_relevant, k),
    "map": average_precision_at(r.relevant, r.total_relevant, k),
    "f1": f1_at(r.relevant, r.total_relevant, k),
  }


@dataclass
class RetrievalReport:
  k: int
  questions: int
  at: dict[str, dict[int, float]]  # metric → cutoff → mean over all questions, e.g. at["hit"][5]
  hit_reachable: float  # hit@k over questions whose file is indexed: the retriever's own quality
  mrr: float
  coverage: float
  answerable: float
  latency_p50_ms: float
  latency_p95_ms: float
  by_kind: dict[str, dict]
  by_source: dict[str, dict]
  results: list[QuestionResult]

  def summary(self) -> dict:
    return {
      "k": self.k,
      "questions": self.questions,
      **{f"{metric}@{c}": v for metric, cuts in self.at.items() for c, v in cuts.items()},
      "hit_reachable": round(self.hit_reachable, 3),
      "mrr": round(self.mrr, 3),
      "coverage": round(self.coverage, 3),
      "answerable": round(self.answerable, 3),
      "latency_p50_ms": round(self.latency_p50_ms, 1),
      "latency_p95_ms": round(self.latency_p95_ms, 1),
    }


def evaluate(
  retrieve: Retriever,
  questions: Iterable[EvalQuestion],
  corpus: Iterable[str],
  k: int = 5,
  indexed_sources: set[str] | None = None,
) -> RetrievalReport:
  """Run every question through `retrieve(query, k)` and score the results.

  `corpus` is the text of every indexed chunk: it tells us how many relevant chunks exist for each question.
  """
  texts = [normalize(t) for t in corpus]
  results = []
  for q in questions:
    started = time.perf_counter()
    docs = retrieve(q.question, k)
    latency = (time.perf_counter() - started) * 1000
    evidence = [normalize(e) for e in q.evidence]
    results.append(
      QuestionResult(
        q,
        [is_relevant(d.page_content, q) for d in docs],
        sum(1 for t in texts if all(e in t for e in evidence)),
        latency,
        indexed_sources is None or q.source in indexed_sources,
        [
          {
            "source": d.metadata.get("source"),
            "text": d.page_content[:160],
            "chars": len(d.page_content),
            "relevant": is_relevant(d.page_content, q),
          }
          for d in docs
        ],
      )
    )
  return _report(results, k)


def _mean(values: list[float]) -> float:
  return round(sum(values) / len(values), 3) if values else 0.0


def _report(results: list[QuestionResult], k: int) -> RetrievalReport:
  def rates(rs: list[QuestionResult]) -> dict:
    scores = [question_metrics(r, k) for r in rs]
    return {
      "questions": len(rs),
      "found": sum(1 for r in rs if r.rank),
      **{m: _mean([s[m] for s in scores]) for m in ("hit", "precision", "recall", "ndcg")},
      "mrr": _mean([reciprocal_rank(r.relevant) for r in rs]),
    }

  by_kind, by_source = defaultdict(list), defaultdict(list)
  for r in results:
    by_kind[r.question.kind].append(r)
    by_source[r.question.source].append(r)
  reachable = [r for r in results if r.reachable]
  latencies = sorted(r.latency_ms for r in results) or [0.0]
  cutoffs = sorted({c for c in (1, 3, 5, k) if c <= k})
  at = {m: {c: _mean([question_metrics(r, c)[m] for r in results]) for c in cutoffs} for m in METRICS}
  return RetrievalReport(
    k=k,
    questions=len(results),
    at=at,
    hit_reachable=rates(reachable)["hit"],
    mrr=_mean([reciprocal_rank(r.relevant) for r in results]),
    coverage=len(reachable) / len(results) if results else 0.0,
    answerable=sum(1 for r in results if r.total_relevant) / len(results) if results else 0.0,
    latency_p50_ms=statistics.median(latencies),
    latency_p95_ms=latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
    by_kind={kind: rates(rs) for kind, rs in sorted(by_kind.items())},
    by_source={source: {**rates(rs), "reachable": rs[0].reachable} for source, rs in sorted(by_source.items())},
    results=results,
  )
