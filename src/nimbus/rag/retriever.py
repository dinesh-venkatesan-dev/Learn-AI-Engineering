"""The retriever: a customer's question in, the most relevant chunks out.

    retriever = Retriever(settings, embeddings, model)
    hits = retriever.search("मुझे साल में कितनी छुट्टी मिलती है?", k=5)   # translated, then searched

The index (built by ingestion) holds English documents embedded with an English model. Questions arrive in
Hindi, Tamil, Hinglish and more, so the retriever first prepares the query: by policy, a non-English question is
translated to English by the chat model (one call), which measured 8/8 on the multilingual set against 2/8 without.
Module 6 adds hybrid search (Postgres full-text + vectors, fused by rank) and reranking (a local cross-encoder);
both are settings, chosen by the evaluation.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from nimbus.config import PROJECT_DIR, Settings
from nimbus.rag.ingestion.registry import valid_on
from nimbus.rag.ingestion.store import VectorIndex
from nimbus.rag.rerank import rerank

TRANSLATE = (
  "Translate this search question into English. If it is already English, return it unchanged. "
  "Reply with the question only.\n\n{question}"
)
CACHE = PROJECT_DIR / ".cache" / "translations"


def needs_translation(question: str, policy: str) -> bool:
  if policy == "always":
    return True
  if policy == "non_ascii":
    return any(not c.isascii() and c.isalpha() for c in question)  # Devanagari, Tamil, é, ß… (not Hinglish)
  return False


def translate(question: str, model: BaseChatModel) -> str:
  """The question in English, cached on disk (the same question is asked again and again)."""
  path = CACHE / (hashlib.sha256(question.encode()).hexdigest()[:24] + ".json")
  if path.exists():
    return json.loads(path.read_text())["english"]
  english = model.invoke(TRANSLATE.format(question=question)).text.strip()
  CACHE.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({"question": question, "english": english}, ensure_ascii=False))
  return english


@dataclass
class PreparedQuery:
  original: str
  query: str  # what is embedded and searched
  translated: bool
  steps: list[str] = field(default_factory=list)


def prepare(question: str, policy: str, model: BaseChatModel | None) -> PreparedQuery:
  if model is not None and needs_translation(question, policy):
    english = translate(question, model)
    return PreparedQuery(question, english, True, [f"translated: {english}"])
  return PreparedQuery(question, question, False)


class Retriever:
  def __init__(self, settings: Settings, embeddings: Embeddings, model: BaseChatModel | None = None):
    self.settings, self.embeddings, self.model = settings, embeddings, model

  def search(
    self, question: str, k: int = 5, as_of: date | None = None, audience: str = "staff"
  ) -> tuple[PreparedQuery, list[tuple[Document, float]]]:
    """Prepare the query, fetch candidates (dense or hybrid) among documents valid on `as_of` (today by default)
    that `audience` may read (public | staff), optionally rerank them, keep the best k."""
    s = self.settings
    prepared = prepare(question, s.query_translation, self.model)
    day = (as_of or date.today()).isoformat()
    prepared.steps.append(f"valid on {day}")
    fetch = max(k, s.search_candidates) if s.rerank != "off" or s.search_mode == "hybrid" else k
    with VectorIndex(s, self.embeddings) as index:
      candidates = index.search(prepared.query, k=fetch, filter=valid_on(day, audience))  # access control HERE
    if s.rerank == "flashrank":
      prepared.steps.append(f"reranked {len(candidates)} candidates")
      return prepared, rerank(prepared.query, [d for d, _ in candidates], s.reranker_model, s.model_cache_dir, k)
    return prepared, candidates[:k]
