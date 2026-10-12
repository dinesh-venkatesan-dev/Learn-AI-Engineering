"""Step 5 — Store: write chunks and vectors to Postgres, incrementally.

Two LangChain pieces work together:
  - PGVectorStore keeps one row per chunk in its own table, with real columns for the metadata we filter on.
  - The record manager remembers a hash of every chunk it has written. LangChain's `index()` uses it to skip chunks
    that haven't changed, and to delete chunks that no longer exist. Re-running ingestion is cheap and never
    creates duplicates.
"""

import asyncio
import logging
from types import TracebackType

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import IndexingResult, index
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
from langchain_postgres.v2.indexes import HNSWIndex
from sqlalchemy import create_engine, text

from nimbus.config import Settings

log = logging.getLogger(__name__)

METADATA_COLUMNS = [
  Column("source", "TEXT", nullable=False),
  Column("page", "INTEGER"),
  Column("file_type", "TEXT"),
  Column("valid_from", "TEXT"),  # from the document registry (Module 6): filters need real columns
  Column("valid_to", "TEXT"),
  Column("access", "TEXT"),  # from the registry (Module 9): public | staff, filtered at retrieval
]


def hybrid_config(settings: Settings) -> HybridSearchConfig:
  """Hybrid search inside Postgres: full-text search (ts_rank over to_tsvector(content)) and vector search, each
  returning `search_candidates`, merged by Reciprocal Rank Fusion (positions only, so their scores never mix)."""
  return HybridSearchConfig(
    tsv_column="",  # computed from the content column; a GIN index on the expression keeps it fast
    tsv_lang="pg_catalog.english",
    fusion_function=reciprocal_rank_fusion,
    fusion_function_parameters={"rrf_k": 60, "fetch_top_k": settings.search_candidates},
    primary_top_k=settings.search_candidates,
    secondary_top_k=settings.search_candidates,
  )


class VectorIndex:
  """The chunks table and its record manager. Use it as a context manager so connections are always closed."""

  def __init__(self, settings: Settings, embeddings: Embeddings):
    self.settings = settings
    self._sql = create_engine(settings.database_url)  # plain SQL: table checks and inspection queries
    self.engine = PGEngine.from_connection_string(settings.database_url)
    self._ensure_table()
    self.store = PGVectorStore.create_sync(
      self.engine,
      embeddings,
      settings.table_name,
      metadata_columns=[c.name for c in METADATA_COLUMNS],
      hybrid_search_config=hybrid_config(settings) if settings.search_mode == "hybrid" else None,
    )
    self.records = SQLRecordManager(namespace=f"pgvector/{settings.table_name}", db_url=settings.database_url)
    self.records.create_schema()  # creates the `upsertion_record` table once
    self.ensure_vector_index()
    if settings.search_mode == "hybrid":
      self.ensure_text_index()

  def ensure_text_index(self) -> None:
    """A GIN index on to_tsvector(content): the full-text half of hybrid search, created once."""
    name = f"{self.settings.table_name}_tsv"
    with self._sql.connect() as db:
      exists = db.execute(text("select to_regclass(:i)"), {"i": name}).scalar()
    if not exists:
      log.info("creating full-text index %s", name)
      with self._sql.begin() as db:
        db.execute(
          text(f"create index {name} on {self.settings.table_name} using gin (to_tsvector('english', content))")
        )

  def ensure_vector_index(self) -> None:
    """An HNSW index on the embedding column, created once. Without it every search compares the question with
    every row (fine for 100 chunks, slow for a million); with it, search walks a graph to the nearest ones."""
    name = f"{self.settings.table_name}_hnsw"  # one per table: index names are unique in a schema
    if not self.store.is_valid_index(name):
      log.info("creating HNSW index %s", name)
      self.store.apply_vector_index(HNSWIndex(name=name, m=16, ef_construction=64))

  def _ensure_table(self) -> None:
    with self._sql.connect() as db:
      exists = db.execute(text("select to_regclass(:t)"), {"t": self.settings.table_name}).scalar()
    if exists:  # a table from an earlier module: add the newer metadata columns (a small migration)
      with self._sql.begin() as db:
        for column in METADATA_COLUMNS[3:]:
          db.execute(
            text(f"alter table {self.settings.table_name} add column if not exists {column.name} {column.data_type}")
          )
    if not exists:
      log.info("creating table %s (vector(%d))", self.settings.table_name, self.settings.embedding_dims)
      self.engine.init_vectorstore_table(
        self.settings.table_name,
        self.settings.embedding_dims,
        metadata_columns=METADATA_COLUMNS,
        id_column=Column("langchain_id", "TEXT", nullable=False),
      )  # sha256 keys

  def index(self, chunks: list[Document], *, full_sync: bool = True, force: bool = False) -> IndexingResult:
    """Write chunks; return how many were added, updated, skipped (unchanged) and deleted.

    full_sync=True means `chunks` is the WHOLE corpus: anything indexed before but missing now (a deleted file,
    a removed section) is deleted. Use full_sync=False to ingest just some files without touching the others.
    """
    return index(
      chunks,
      self.records,
      self.store,
      cleanup="full" if full_sync else "incremental",
      source_id_key="source",  # all chunks of one file share a source id
      batch_size=self.settings.index_batch_size,
      key_encoder="sha256",  # chunk identity = hash of its content + metadata
      force_update=force,
    )

  def search(self, query: str, k: int = 3, filter: dict | None = None) -> list[tuple[Document, float]]:  # noqa: A002
    """Dense mode: the chunks closest in meaning, with their cosine distance (0 = identical direction).
    Hybrid mode: the fused ranking of full-text and vector search, with its RRF score (higher = better)."""
    return self.store.similarity_search_with_score(query, k=k, filter=filter)

  def sources(self) -> set[str]:
    """Every file that has at least one chunk in the index."""
    with self._sql.connect() as db:
      return {row[0] for row in db.execute(text(f"select distinct source from {self.settings.table_name}"))}

  def texts(self) -> list[str]:
    """The text of every chunk in the index (evaluation counts the relevant ones for recall and nDCG)."""
    with self._sql.connect() as db:
      return [row[0] for row in db.execute(text(f"select content from {self.settings.table_name}"))]

  def stats(self, sample: int = 8) -> dict:
    """What's in the table: row count, sizes, and a few rows exactly as stored."""
    table = self.settings.table_name
    with self._sql.connect() as db:
      count = db.execute(text(f"select count(*) from {table}")).scalar()
      size = db.execute(text("select pg_size_pretty(pg_total_relation_size(:t))"), {"t": table}).scalar()
      rows = db.execute(
        text(
          f"select langchain_id, left(content, 90) as content, source, page, file_type, langchain_metadata, "
          f"vector_dims(embedding) as dims, (embedding::real[])[1:4] as first_numbers "
          f"from {table} order by source, page, (langchain_metadata->>'chunk_index')::int limit :n"
        ),
        {"n": sample},
      )
      records = db.execute(
        text("select count(*) from upsertion_record where namespace = :ns"), {"ns": self.records.namespace}
      ).scalar()
    return {"table": table, "rows": count, "size": size, "records": records, "sample": [dict(r._mapping) for r in rows]}

  def close(self) -> None:
    self._sql.dispose()
    self.records.engine.dispose()
    asyncio.run(self.engine.close())  # PGEngine's pool is async; close() disposes it

  def __enter__(self) -> "VectorIndex":
    return self

  def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, tb: TracebackType | None) -> None:
    self.close()
