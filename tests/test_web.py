"""The API the course UI uses: lesson content resolves to real source code, and ingestion streams its steps."""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from academy.web.app import create_app
from nimbus.config import Settings


def test_catalog_reports_coverage():
  with TestClient(create_app()) as client:
    data = client.get("/api/catalog").json()
  assert data["items"] > 1000
  rag = next(t for t in data["tracks"] if t["id"] == "rag")
  assert rag["modules"][0]["covered"] == rag["modules"][0]["items"] > 0


def test_module_page_resolves_labs_and_real_code():
  with TestClient(create_app()) as client:
    page = client.get("/api/modules/rag/01").json()
  assert [lab["id"] for lab in page["labs"]][:1] == ["rag.why-rag"]
  code = [c for step in page["path"]["steps"] for c in step["code"]] + [c for lab in page["labs"] for c in lab["code"]]
  for c in code:
    assert c["source"].lstrip().startswith(("def ", "class ", "@")), c["symbol"]


def test_unbuilt_module_is_404():
  with TestClient(create_app()) as client:
    assert client.get("/api/modules/production/07").status_code == 404  # still planned: no page yet


def test_lab_runs_with_coerced_inputs(tmp_path):
  with TestClient(create_app(lab_results=tmp_path)) as client:
    result = client.post("/api/labs/rag.semantic-search/run", json={"query": "Can I work from home?"}).json()
    assert result["views"][0]["type"] == "results"
    assert client.get("/api/labs/rag.semantic-search/last").json()["inputs"] == {"query": "Can I work from home?"}


def test_last_lab_result_survives_a_restart(tmp_path):
  with TestClient(create_app(lab_results=tmp_path)) as client:
    ran = client.post("/api/labs/rag.semantic-search/run", json={"query": "Can I work from home?"}).json()
  with TestClient(create_app(lab_results=tmp_path)) as client:  # a fresh app: nothing in memory
    last = client.get("/api/labs/rag.semantic-search/last").json()
    assert last == ran  # views and the "What it covers" examples, exactly as they were
    assert client.get("/api/labs/rag.keyword-search/last").json() is None  # never run: the page offers ▶ Run
    assert client.get("/api/labs/no-such-lab/last").status_code == 404


@pytest.mark.db
def test_ingest_streams_every_step_then_done(tmp_path):
  (tmp_path / "a.md").write_text("# A\n\nAlpha facts about the first topic, long enough to keep.")
  table = f"test_web_{uuid.uuid4().hex[:8]}"
  settings = Settings(data_dir=tmp_path, table_name=table)
  try:
    with TestClient(create_app(settings)) as client:
      body = client.get("/api/rag/ingest").text
      last = client.get("/api/rag/ingest/last").json()
    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event: ")]
    assert events == ["step"] * 5 + ["done"]
    assert last["report"]["added"] == 1
    data = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert [d.get("step") for d in data[:5]] == ["load", "clean", "chunk", "embed", "store"]
  finally:
    engine = create_engine(settings.database_url)
    with engine.begin() as db:
      db.execute(text(f"drop table if exists {table}"))
      db.execute(text("delete from upsertion_record where namespace = :ns"), {"ns": f"pgvector/{table}"})
    engine.dispose()


def test_bad_settings_are_rejected():
  with TestClient(create_app()) as client:
    response = client.get("/api/rag/ingest", params={"chunk_size": 100, "chunk_overlap": 500})
  assert response.status_code == 422
  assert response.json()["detail"] == "chunk_overlap must be smaller than chunk_size"
  assert "nimbus:nimbus@" not in response.text  # never echo the database URL (it holds the password)
