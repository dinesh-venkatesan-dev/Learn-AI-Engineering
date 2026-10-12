"""The ingestion pipeline: discover → load → clean → chunk → embed + store.

    pipeline = IngestionPipeline(settings)
    report = pipeline.run()            # safe to run again: unchanged chunks are skipped

Failures are per file: one unreadable file is logged and reported, and the rest still get indexed.
An optional observer receives each step's result as it finishes (the CLI prints them, the web UI streams them).
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from nimbus.config import Settings
from nimbus.llm import get_chat_model
from nimbus.rag.ingestion.chunking import make_chunks
from nimbus.rag.ingestion.cleaning import clean_file
from nimbus.rag.ingestion.embeddings import LocalEmbeddings, MeteredEmbeddings, get_embeddings
from nimbus.rag.ingestion.loaders import discover, loader_for
from nimbus.rag.ingestion.registry import apply_registry, load_registry
from nimbus.rag.ingestion.security import secure
from nimbus.rag.ingestion.store import VectorIndex

log = logging.getLogger(__name__)


@dataclass
class StepResult:
  step: str  # load | clean | chunk | embed | store
  ms: int
  stats: dict[str, Any]
  data: dict[str, Any] = field(default_factory=dict)  # the step's output, for inspection


Observer = Callable[[StepResult], None]


@dataclass
class IngestionReport:
  files: int
  unsupported: list[str]  # files with no loader yet: not searchable
  quarantined: list[dict]  # chunks kept out of the index for review (instructions aimed at the model)
  failed: list[dict[str, str]]
  documents: int
  chunks: int
  added: int
  updated: int
  skipped: int
  deleted: int
  seconds: float


class IngestionPipeline:
  def __init__(self, settings: Settings, embeddings: LocalEmbeddings | None = None):
    self.settings = settings
    self.embeddings = embeddings or get_embeddings(settings)

  def run(self, observer: Observer | None = None, *, force: bool = False) -> IngestionReport:
    started = time.perf_counter()
    notify = observer or (lambda result: None)
    root = self.settings.data_dir

    # 1. Load: every supported file; a failing file is recorded, not fatal
    t = time.perf_counter()
    files, unsupported = discover(root, self.settings.loader_version)
    skipped = [path.relative_to(root).as_posix() for path in unsupported]
    if skipped:
      log.warning("no loader for %d file(s): %s", len(skipped), ", ".join(skipped))
    loaded: dict[str, list[Document]] = {}
    failed: list[dict[str, str]] = []
    for path in files:
      source = path.relative_to(root).as_posix()
      try:
        ocr = self.settings.ocr_engine if self.settings.loader_version == "v3" else None
        loaded[source] = loader_for(path, root, ocr).load()
      except Exception as error:  # noqa: BLE001 — any parser error: report the file and carry on
        log.warning("could not load %s: %s", source, error)
        failed.append({"source": source, "error": f"{type(error).__name__}: {error}"})
    registry = load_registry(root)
    for docs in loaded.values():
      apply_registry(docs, registry)  # validity dates (and any other registry facts) into every Document
    documents = [d for docs in loaded.values() for d in docs]
    notify(
      StepResult(
        "load",
        _ms(t),
        {"files": len(files), "documents": len(documents), "failed": len(failed), "unsupported": len(skipped)},
        {"documents": documents, "failed": failed, "unsupported": skipped},
      )
    )

    # 2. Clean: per file, so repeated page headers can be detected across its pages
    t = time.perf_counter()
    cleaned_by_file = {source: clean_file(docs) for source, docs in loaded.items()}
    cleaned = [d for docs in cleaned_by_file.values() for d in docs]
    chars_before = sum(len(d.page_content) for d in documents)
    chars_after = sum(len(d.page_content) for d in cleaned)
    notify(
      StepResult(
        "clean",
        _ms(t),
        {"chars_before": chars_before, "chars_after": chars_after},
        {"before": documents, "after": cleaned},
      )
    )

    # 3. Chunk
    t = time.perf_counter()
    s = self.settings
    model = get_chat_model(s) if s.chunk_context == "llm" else None  # only contextual retrieval needs a model
    chunks = make_chunks(
      cleaned_by_file, s.chunk_size, s.chunk_overlap, s.min_chunk_chars, s.chunk_strategy, s.chunk_context, model
    )
    chunks, quarantined, redactions = secure(chunks)  # injection quarantined, personal data redacted (Module 9)
    for q in quarantined:
      log.warning("quarantined a chunk of %s: %s", q["source"], q["reason"])
    notify(
      StepResult(
        "chunk",
        _ms(t),
        {
          "chunks": len(chunks),
          "chunk_size": s.chunk_size,
          "chunk_overlap": s.chunk_overlap,
          "strategy": s.chunk_strategy,
          "context": s.chunk_context,
          "quarantined": len(quarantined),
          "redacted": redactions,
        },
        {"documents": cleaned, "chunks": chunks, "quarantined": quarantined},
      )
    )

    # 4 + 5. Embed and store: index() embeds only the chunks it hasn't seen before, then writes them
    t = time.perf_counter()
    metered = MeteredEmbeddings(self.embeddings)
    with VectorIndex(self.settings, metered) as vector_index:
      result = vector_index.index(chunks, full_sync=True, force=force)
      stats = vector_index.stats()
    store_ms = _ms(t)
    notify(
      StepResult(
        "embed",
        round(metered.metrics.seconds * 1000),
        {
          "embedded": metered.metrics.texts,
          "dims": self.settings.embedding_dims,
          "model": self.settings.embedding_model,
        },
        {"sample": metered.metrics.sample, "sample_text": metered.metrics.sample_text},
      )
    )
    notify(
      StepResult(
        "store",
        store_ms - round(metered.metrics.seconds * 1000),
        {
          "added": result["num_added"],
          "updated": result["num_updated"],
          "skipped": result["num_skipped"],
          "deleted": result["num_deleted"],
          "rows": stats["rows"],
        },
        {"table": stats},
      )
    )

    report = IngestionReport(
      files=len(files),
      unsupported=skipped,
      quarantined=quarantined,
      failed=failed,
      documents=len(documents),
      chunks=len(chunks),
      added=result["num_added"],
      updated=result["num_updated"],
      skipped=result["num_skipped"],
      deleted=result["num_deleted"],
      seconds=round(time.perf_counter() - started, 2),
    )
    log.info(
      "ingested %d files → %d chunks: %d added, %d skipped, %d deleted in %.1fs",
      report.files,
      report.chunks,
      report.added,
      report.skipped,
      report.deleted,
      report.seconds,
    )
    return report


def _ms(since: float) -> int:
  return round((time.perf_counter() - since) * 1000)
