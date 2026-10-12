"""The course app's HTTP API: the catalogue, module pages, labs, the Nimbus ingestion run and its scorecards.

Run:  uv run academy          → http://localhost:8300
"""

import json
import logging
import queue
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from pydantic import ValidationError

import academy.labs.agents.m01_graphs  # noqa: F401
import academy.labs.agents.m02_control  # noqa: F401
import academy.labs.agents.m03_workflows  # noqa: F401
import academy.labs.agents.m04_agents  # noqa: F401
import academy.labs.agents.m05_memory  # noqa: F401
import academy.labs.agents.m06_hitl  # noqa: F401
import academy.labs.agents.m07_multi  # noqa: F401
import academy.labs.agents.m08_production  # noqa: F401
import academy.labs.agents.m09_deep  # noqa: F401
import academy.labs.agents.m10_deep_control  # noqa: F401
import academy.labs.data.m01_text_to_sql  # noqa: F401
import academy.labs.data.m02_sql_agents  # noqa: F401
import academy.labs.data.m03_analytics  # noqa: F401
import academy.labs.fastapi.m01_apis  # noqa: F401
import academy.labs.fastapi.m02_data_streaming  # noqa: F401
import academy.labs.fastapi.m03_security_testing  # noqa: F401
import academy.labs.fastapi.m04_agent_service  # noqa: F401
import academy.labs.langchain.m01_models  # noqa: F401  (registers the labs)
import academy.labs.langchain.m02_lcel  # noqa: F401
import academy.labs.langchain.m03_tools  # noqa: F401
import academy.labs.langchain.m04_agents  # noqa: F401
import academy.labs.langchain.m05_middleware  # noqa: F401
import academy.labs.mcp.m01_protocol  # noqa: F401
import academy.labs.mcp.m02_servers  # noqa: F401
import academy.labs.mcp.m03_clients  # noqa: F401
import academy.labs.mcp.m04_production  # noqa: F401
import academy.labs.production.m01_reliability  # noqa: F401
import academy.labs.production.m02_caching  # noqa: F401
import academy.labs.production.m03_tracing  # noqa: F401
import academy.labs.production.m04_evaluation  # noqa: F401
import academy.labs.production.m05_shipping  # noqa: F401
import academy.labs.rag.m01_foundations  # noqa: F401  (registers the labs)
import academy.labs.rag.m02_loading  # noqa: F401
import academy.labs.rag.m03_ocr  # noqa: F401
import academy.labs.rag.m04_chunking  # noqa: F401
import academy.labs.rag.m05_storage  # noqa: F401
import academy.labs.rag.m06_retrieval  # noqa: F401
import academy.labs.rag.m06_transform  # noqa: F401
import academy.labs.rag.m07_generation  # noqa: F401
import academy.labs.rag.m08_advanced  # noqa: F401
import academy.labs.rag.m09_production  # noqa: F401
from academy import catalog
from academy.labs.runtime import REGISTRY
from nimbus.config import PROJECT_DIR, Settings, get_settings
from nimbus.llm.models import aclose_clients
from nimbus.logging_setup import setup_logging
from nimbus.rag.evaluation import evaluate, load_questions
from nimbus.rag.ingestion.embeddings import LocalEmbeddings
from nimbus.rag.ingestion.pipeline import IngestionPipeline, StepResult
from nimbus.rag.ingestion.store import VectorIndex

log = logging.getLogger(__name__)
FRONTEND_DIR = PROJECT_DIR / "frontend" / "dist"
STATE_DIR = PROJECT_DIR / ".state"  # scorecard history (git-ignored)
HISTORY = STATE_DIR / "rag_scorecards.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
  settings: Settings = app.state.settings
  setup_logging(settings.log_level)
  # One embedding model for the whole process: loading it takes ~1 s and ~100 MB of memory.
  app.state.embeddings = LocalEmbeddings(
    settings.embedding_model, settings.model_cache_dir, settings.embedding_batch_size
  )
  app.state.ingest_lock = threading.Lock()
  app.state.last_run = None  # the latest ingestion's step results, so a page refresh can show them again
  app.state.last_labs = {}  # lab id → latest result
  yield
  await aclose_clients()  # the cached model clients' HTTP connections


def create_app(settings: Settings | None = None) -> FastAPI:
  app = FastAPI(title="Nimbus academy", lifespan=lifespan)
  app.state.settings = settings or get_settings()

  @app.get("/api/health")
  def health() -> dict:
    return {"ok": True}

  # ── Catalogue ────────────────────────────────────────────────────────────────
  @app.get("/api/catalog")
  def catalog_view() -> dict:
    return catalog.coverage()

  @app.get("/api/syllabus/{track}")
  def syllabus(track: str) -> list[dict]:
    """Every learning point of a track, with its home module and whether it's covered yet."""
    out = []
    for t in catalog.coverage()["tracks"]:
      if t["id"] != track:
        continue
      for module in t["modules"]:
        page = catalog.load_module(track, module["id"]) if module["status"] != "planned" else None
        patterns = catalog.covers_of(page) if page else []
        for item in catalog.home_items(module):
          out.append(
            {
              **item,
              "module": module["id"],
              "module_title": module["title"],
              "covered": any(catalog.matches(item, p) for p in patterns),
              "stars": catalog.lesson_stars(item["lesson"]),
            }
          )
    if not out:
      raise HTTPException(status.HTTP_404_NOT_FOUND, f"No track {track}")
    return out

  @app.get("/api/modules/{track}/{module_id}")
  def module_page(track: str, module_id: str) -> dict:
    page = catalog.load_module(track, module_id)
    if page is None:
      raise HTTPException(status.HTTP_404_NOT_FOUND, f"Module {track}/{module_id} isn't built yet")
    page.update(catalog.module_importance(track, module_id))
    items = {s["id"]: {**s, "stars": catalog.lesson_stars(s["lesson"])} for s in catalog.syllabus()}
    for entry in page["labs"]:
      lab = REGISTRY[entry["lab"]]
      entry.update(lab.describe())
      entry["code"] = [catalog.source_of(ref) for ref in entry.get("code", [])]
      entry["covered_items"] = [s for s in items.values() if any(catalog.matches(s, p) for p in entry["covers"])]
    for step in page["path"]["steps"]:
      step["code"] = [catalog.source_of(ref) for ref in step["code"]]
      if step.get("lab"):  # a step can embed a lab that runs the production code
        step["lab_entry"] = {
          **REGISTRY[step["lab"]].describe(),
          "lab": step["lab"],
          "icon": step["icon"],
          "explain": "",
          "covers": [],
          "code": [],
          "covered_items": [],
        }
    page["path"]["covered_items"] = [
      s for s in items.values() if any(catalog.matches(s, p) for p in page["path"].get("covers", []))
    ]
    return page

  # ── Labs ─────────────────────────────────────────────────────────────────────
  @app.post("/api/labs/{lab_id}/run")
  def run_lab(request: Request, lab_id: str, inputs: Annotated[dict[str, Any], Body()] = None) -> dict:
    lab = REGISTRY.get(lab_id)
    if lab is None:
      raise HTTPException(status.HTTP_404_NOT_FOUND, f"No lab {lab_id}")
    try:
      result = lab.run(inputs or {})
    except Exception as error:  # noqa: BLE001 — show any lab failure on the page
      log.exception("lab %s failed", lab_id)
      raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"{type(error).__name__}: {error}") from None
    request.app.state.last_labs[lab_id] = result
    return result

  @app.get("/api/labs/{lab_id}/last")
  def last_lab(request: Request, lab_id: str) -> dict | None:
    return request.app.state.last_labs.get(lab_id)

  # ── Production path: ingestion ───────────────────────────────────────────────
  def run_settings(request: Request, chunk_size: int | None = None, chunk_overlap: int | None = None) -> Settings:
    """Validate the requested settings BEFORE streaming starts, so a bad value is a proper 422 response."""
    overrides = {k: v for k, v in {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}.items() if v is not None}
    try:
      return Settings(**{**request.app.state.settings.model_dump(), **overrides})
    except ValidationError as error:
      # only the messages: the full error echoes the input, which includes the database URL and its password
      messages = [e["msg"].removeprefix("Value error, ") for e in error.errors()]
      raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "; ".join(messages)) from None

  @app.get("/api/rag/ingest", response_class=EventSourceResponse)
  def ingest(
    request: Request, settings: Annotated[Settings, Depends(run_settings)], force: bool = False
  ) -> Iterator[ServerSentEvent]:
    """Run the ingestion pipeline and stream each step's result as it finishes."""
    lock: threading.Lock = request.app.state.ingest_lock
    if not lock.acquire(blocking=False):
      yield ServerSentEvent(event="error", data={"message": "An ingestion is already running. Try again in a moment."})
      return
    events: queue.Queue[ServerSentEvent | None] = queue.Queue()

    def work() -> None:
      steps: list[dict] = []

      def on_step(step: StepResult) -> None:
        steps.append(_step_json(step))
        events.put(ServerSentEvent(event="step", data=steps[-1]))

      try:
        report = IngestionPipeline(settings, request.app.state.embeddings).run(on_step, force=force)
        used = {"chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}
        request.app.state.last_run = {"steps": steps, "report": report.__dict__, "settings": used}
        request.app.state.settings = settings  # later evaluations describe the index as it is now
        events.put(ServerSentEvent(event="done", data=report.__dict__))
      except Exception as error:  # noqa: BLE001 — report any failure to the client
        log.exception("ingestion failed")
        events.put(ServerSentEvent(event="error", data={"message": f"{type(error).__name__}: {error}"}))
      finally:
        events.put(None)
        lock.release()

    threading.Thread(target=work, daemon=True).start()
    while (event := events.get()) is not None:
      yield event

  @app.get("/api/rag/ingest/last")
  def last_run(request: Request) -> dict | None:
    return request.app.state.last_run

  # ── Production path: the scorecard ───────────────────────────────────────────
  @app.post("/api/rag/evaluate")
  def evaluate_index(request: Request, k: int = 5) -> dict:
    """Score the current index on the evaluation set and add the result to the history."""
    settings: Settings = request.app.state.settings
    with VectorIndex(settings, request.app.state.embeddings) as index:
      report = evaluate(
        lambda q, n: [d for d, _ in index.search(q, n)],
        load_questions(settings.eval_questions),
        index.texts(),
        k=k,
        indexed_sources=index.sources(),
      )
    record = {
      "at": datetime.now(UTC).isoformat(timespec="seconds"),
      "config": {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "retrieval": "semantic (pgvector cosine)",
      },
      **report.summary(),
      "by_kind": report.by_kind,
      "by_source": report.by_source,
      "results": [
        {
          "id": r.question.id,
          "question": r.question.question,
          "kind": r.question.kind,
          "source": r.question.source,
          "rank": r.rank,
          "relevant_in_index": r.total_relevant,
          "precision": round(sum(r.relevant[: report.k]) / report.k, 3),
          "recall": round(sum(r.relevant[: report.k]) / r.total_relevant, 3) if r.total_relevant else 0.0,
          "reachable": r.reachable,
          "top": r.top,
        }
        for r in report.results
      ],
    }
    STATE_DIR.mkdir(exist_ok=True)
    with HISTORY.open("a") as f:
      f.write(json.dumps({k: v for k, v in record.items() if k != "results"}) + "\n")
    return record

  @app.get("/api/rag/scorecards")
  def scorecards() -> list[dict]:
    if not HISTORY.exists():
      return []
    return [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]

  if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
  return app


def _doc(d: Document, text: bool = True) -> dict[str, Any]:
  out = {"source": d.metadata["source"], "page": d.metadata.get("page"), "chars": len(d.page_content)}
  if text:
    out["text"] = d.page_content
  return out


def _step_json(step: StepResult) -> dict:
  """A step's result as JSON for the page. The corpus is small; a real one would be paged."""
  data: dict[str, Any] = {}
  if step.step == "load":
    data = {
      "documents": [_doc(d) for d in step.data["documents"]],
      "failed": step.data["failed"],
      "unsupported": step.data["unsupported"],
    }
  elif step.step == "clean":
    after = {(d.metadata["source"], d.metadata.get("page")): d for d in step.data["after"]}
    data = {
      "documents": [
        {
          **_doc(b, text=False),
          "before": b.page_content,
          "after": after.get((b.metadata["source"], b.metadata.get("page")), Document("")).page_content,
        }
        for b in step.data["before"]
      ]
    }
  elif step.step == "chunk":
    data = {
      "documents": [_doc(d) for d in step.data["documents"]],
      "chunks": [
        {**_doc(c), "start": c.metadata["start_index"], "index": c.metadata["chunk_index"]} for c in step.data["chunks"]
      ],
    }
  elif step.step == "embed":
    data = {"vector": [round(x, 5) for x in step.data["sample"]], "text": step.data["sample_text"]}
  elif step.step == "store":
    data = step.data["table"]
  return {"step": step.step, "ms": step.ms, "stats": step.stats, "data": data}


app = create_app()


def main() -> None:
  import uvicorn

  uvicorn.run("academy.web.app:app", host="127.0.0.1", port=8300, log_level="warning")
