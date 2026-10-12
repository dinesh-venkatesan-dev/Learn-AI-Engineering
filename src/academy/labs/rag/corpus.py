"""The sample corpus as the labs see it: loaded, cleaned and chunked by the same nimbus code as production.

Cached per chunking setting, so labs respond instantly after the first call.
"""

import re
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

from nimbus.config import get_settings
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import EvalQuestion, load_questions
from nimbus.rag.ingestion.chunking import make_chunks
from nimbus.rag.ingestion.cleaning import clean_file
from nimbus.rag.ingestion.embeddings import LocalEmbeddings
from nimbus.rag.ingestion.loaders import discover, loader_for

STOPWORDS = {
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "can",
  "do",
  "does",
  "for",
  "from",
  "how",
  "i",
  "in",
  "is",
  "it",
  "its",
  "me",
  "my",
  "of",
  "on",
  "or",
  "our",
  "the",
  "to",
  "was",
  "we",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "will",
  "with",
  "you",
  "your",
}


@lru_cache
def documents(version: str = "v1") -> tuple[Document, ...]:
  """Every loadable file, cleaned. v1: the Module 1 loaders (text, HTML, PDF); v2: every format but images;
  v3-tesseract / v3-vision: + OCR of images and scanned pages with that engine."""
  root = get_settings().data_dir
  types, _, engine = version.partition("-")
  files, _ = discover(root, types)
  return tuple(d for path in files for d in clean_file(loader_for(path, root, engine or None).load()))


@lru_cache
def chunks(
  size: int = 500, overlap: int = 80, version: str = "v1", strategy: str = "recursive", context: str = "none"
) -> tuple[Document, ...]:
  by_file: dict[str, list[Document]] = {}
  for d in documents(version):
    by_file.setdefault(d.metadata["source"], []).append(d)
  model = get_chat_model() if context == "llm" else None
  made = make_chunks(by_file, size, overlap, get_settings().min_chunk_chars, strategy, context, model)
  return tuple(made)


@lru_cache
def embeddings() -> LocalEmbeddings:
  s = get_settings()
  return LocalEmbeddings(s.embedding_model, s.model_cache_dir, s.embedding_batch_size)


@lru_cache
def chunk_vectors(
  size: int = 500, overlap: int = 80, version: str = "v1", strategy: str = "recursive", context: str = "none"
) -> np.ndarray:
  texts = [c.page_content for c in chunks(size, overlap, version, strategy, context)]
  return np.array(embeddings().embed_documents(texts))


@lru_cache
def questions() -> tuple[EvalQuestion, ...]:
  return tuple(load_questions(get_settings().eval_questions))


def indexed_sources(size: int = 500, overlap: int = 80, version: str = "v1") -> set[str]:
  return {c.metadata["source"] for c in chunks(size, overlap, version)}


def question_for(text: str) -> EvalQuestion | None:
  """The evaluation question with this exact text, if any (so labs can mark results as relevant)."""
  return next((q for q in questions() if q.question.strip().lower() == text.strip().lower()), None)


def tokenize(text: str) -> list[str]:
  return [w for w in re.findall(r"[a-z0-9@.\-]+[a-z0-9]|[a-z0-9]", text.lower()) if w not in STOPWORDS]
