"""Reranking: a second, careful look at the candidates.

Embeddings (bi-encoders) embed the question and each chunk separately, so search is fast but coarse. A
cross-encoder reads the question and one chunk together and scores how well that chunk answers it: slower, so it
only scores the top candidates. FlashRank runs a small cross-encoder locally (~30 MB, no API key).
"""

from functools import lru_cache
from pathlib import Path

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document


@lru_cache
def ranker(model: str, cache_dir: Path) -> Ranker:
  return Ranker(model_name=model, cache_dir=str(cache_dir))  # downloaded once, loaded once per process


def rerank(
  question: str, documents: list[Document], model: str, cache_dir: Path, top_n: int
) -> list[tuple[Document, float]]:
  """The documents reordered by the cross-encoder's score (higher = better), best `top_n` kept."""
  if not documents:
    return []
  request = RerankRequest(query=question, passages=[{"id": i, "text": d.page_content} for i, d in enumerate(documents)])
  return [(documents[r["id"]], float(r["score"])) for r in ranker(model, cache_dir).rerank(request)[:top_n]]
