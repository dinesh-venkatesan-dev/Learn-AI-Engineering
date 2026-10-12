"""Step 4 — Embed: turn text into vectors (lists of numbers) that capture its meaning.

Texts with similar meaning get vectors that point in similar directions; search compares directions. The model runs
locally with FastEmbed (ONNX on the CPU): no API key, no per-token cost, nothing leaves the machine.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
  """LangChain's Embeddings interface, two methods: embed_documents (many texts) and embed_query (one question)."""

  def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 64):
    self.model_name, self.cache_dir, self.batch_size = model_name, cache_dir, batch_size
    self._model: TextEmbedding | None = None
    self._lock = threading.Lock()

  @property
  def model(self) -> TextEmbedding:
    # Loaded on first use (downloaded once, ~70 MB) and shared by every thread afterwards.
    with self._lock:
      if self._model is None:
        self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
      return self._model

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in self.model.embed(texts, batch_size=self.batch_size)]

  def embed_query(self, text: str) -> list[float]:
    return next(iter(self.model.query_embed(text))).tolist()


@dataclass
class EmbeddingMetrics:
  texts: int = 0
  seconds: float = 0.0
  sample: list[float] = field(default_factory=list)  # the first vector produced, for inspection
  sample_text: str = ""


class MeteredEmbeddings(Embeddings):
  """Wraps any Embeddings to count texts and time spent: in production these feed cost and latency dashboards."""

  def __init__(self, inner: Embeddings):
    self.inner, self.metrics = inner, EmbeddingMetrics()

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    started = time.perf_counter()
    vectors = self.inner.embed_documents(texts)
    self.metrics.texts += len(texts)
    self.metrics.seconds += time.perf_counter() - started
    if vectors and not self.metrics.sample:
      self.metrics.sample, self.metrics.sample_text = vectors[0], texts[0]
    return vectors

  def embed_query(self, text: str) -> list[float]:
    return self.inner.embed_query(text)


def get_embeddings(settings) -> Embeddings:
  """The configured embedding model. local: FastEmbed on this machine (free, private, fast). mistral: Mistral's
  embedding API (better on this corpus and multilingual, but every question becomes an API request, shares the
  provider's rate limit, and sends text to the provider). Its vectors have 1024 dimensions: a separate table."""
  if settings.embedding_provider == "mistral":
    from langchain_mistralai import MistralAIEmbeddings

    import nimbus.llm.models  # noqa: F401  (loads ai-course/.env: MISTRAL_API_KEY)

    return MistralAIEmbeddings(model=settings.embedding_model)
  return LocalEmbeddings(settings.embedding_model, settings.model_cache_dir, settings.embedding_batch_size)
