"""The retrieval metrics, checked against values worked out by hand."""

import math

import pytest
from langchain_core.documents import Document

from nimbus.rag.evaluation import (
  EvalQuestion,
  average_precision_at,
  evaluate,
  f1_at,
  hit_at,
  ndcg_at,
  precision_at,
  recall_at,
  reciprocal_rank,
)

# ranks 1..5: relevant at 2 and 4; the index holds 3 relevant chunks in total
RANKED = [False, True, False, True, False]


def test_hit_and_reciprocal_rank():
  assert hit_at(RANKED, 1) == 0.0
  assert hit_at(RANKED, 2) == 1.0
  assert reciprocal_rank(RANKED) == 0.5
  assert reciprocal_rank([False, False]) == 0.0


def test_precision_divides_by_k_even_when_fewer_came_back():
  assert precision_at(RANKED, 5) == pytest.approx(2 / 5)
  assert precision_at([True], 5) == pytest.approx(1 / 5)


def test_recall_uses_every_relevant_chunk_in_the_index():
  assert recall_at(RANKED, 3, 5) == pytest.approx(2 / 3)
  assert recall_at(RANKED, 0, 5) == 0.0  # nothing in the index holds the answer


def test_ndcg_against_the_ideal_ranking():
  dcg = 1 / math.log2(3) + 1 / math.log2(5)
  ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
  assert ndcg_at(RANKED, 3, 5) == pytest.approx(dcg / ideal)
  assert ndcg_at([True, True, True], 3, 3) == pytest.approx(1.0)


def test_average_precision_and_f1():
  assert average_precision_at(RANKED, 3, 5) == pytest.approx((1 / 2 + 2 / 4) / 3)
  p, r = 2 / 5, 2 / 3
  assert f1_at(RANKED, 3, 5) == pytest.approx(2 * p * r / (p + r))


def test_evaluate_counts_relevant_chunks_in_the_corpus():
  q = EvalQuestion("q1", "leave?", ("24 days",), "handbook.md", "number")
  corpus = ["You get 24 days of leave.", "Unrelated.", "Again: 24 DAYS a year.", "No."]
  docs = [Document(t, metadata={"source": "handbook.md"}) for t in ("Unrelated.", "You get 24 days of leave.")]
  report = evaluate(lambda _q, k: docs[:k], [q], corpus, k=3)
  result = report.results[0]
  assert result.total_relevant == 2
  assert result.rank == 2
  assert report.at["hit"][1] == 0.0 and report.at["hit"][3] == 1.0
  assert report.at["recall"][3] == 0.5
  assert report.answerable == 1.0
