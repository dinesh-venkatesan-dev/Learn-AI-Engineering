"""End to end against the real database: incremental indexing adds, skips, updates and deletes correctly."""

import uuid

import pytest
from sqlalchemy import create_engine, text

from nimbus.config import Settings
from nimbus.rag.ingestion.pipeline import IngestionPipeline

pytestmark = pytest.mark.db


@pytest.fixture
def settings(tmp_path):
  table = f"test_chunks_{uuid.uuid4().hex[:8]}"
  s = Settings(data_dir=tmp_path, table_name=table, chunk_size=200, chunk_overlap=20, min_chunk_chars=10)
  yield s
  engine = create_engine(s.database_url)
  with engine.begin() as db:
    db.execute(text(f"drop table if exists {table}"))
    db.execute(text("delete from upsertion_record where namespace = :ns"), {"ns": f"pgvector/{table}"})
  engine.dispose()


def test_rerun_skips_unchanged_and_tracks_edits_and_deletes(settings):
  (settings.data_dir / "a.md").write_text("# A\n\nAlpha facts about the first topic.")
  (settings.data_dir / "b.md").write_text("# B\n\nBeta facts about the second topic.")
  pipeline = IngestionPipeline(settings)

  first = pipeline.run()
  assert (first.added, first.skipped, first.deleted) == (2, 0, 0)

  again = pipeline.run()
  assert (again.added, again.skipped, again.deleted) == (0, 2, 0)  # nothing re-embedded

  (settings.data_dir / "a.md").write_text("# A\n\nAlpha facts, now revised.")
  edited = pipeline.run()
  assert (edited.added, edited.skipped, edited.deleted) == (1, 1, 1)  # new version in, old version out

  (settings.data_dir / "b.md").unlink()
  removed = pipeline.run()
  assert (removed.added, removed.skipped, removed.deleted) == (0, 1, 1)  # deleted file → its chunks deleted
