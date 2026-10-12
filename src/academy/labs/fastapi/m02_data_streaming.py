"""FastAPI · Module 2 labs: data, streaming and background work in the Nimbus API. Notes in a database with
SQLModel, uploads and downloads, answers streamed as Server-Sent Events, a WebSocket chat, and summaries written
after the response. (Module 3 explains the sign-in these labs use.)"""

import json
import tempfile
import threading
import time
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table

PASSWORD = "lab-password-123"


def lab_app(chat_model=None):
  """A Nimbus API with its own temporary database (and the real model unless one is given)."""
  from nimbus.api.app import create_app
  from nimbus.config import Settings

  folder = tempfile.mkdtemp(prefix="nimbus-api-lab-")
  return create_app(settings=Settings(api_database_url=f"sqlite:///{folder}/notes.db"), chat_model=chat_model)


def fake_model(answer_text: str = "A one-line summary of the note."):
  from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
  from langchain_core.messages import AIMessage

  return GenericFakeChatModel(messages=iter([AIMessage(answer_text)] * 200))


def sign_in(client, username: str = "priya") -> str:
  client.post("/auth/register", json={"username": username, "password": PASSWORD})
  token = client.post("/auth/token", data={"username": username, "password": PASSWORD}).json()["access_token"]
  client.headers["Authorization"] = f"Bearer {token}"
  return token


def wait_for_summaries(client, timeout_s: float = 60) -> None:
  """Background summaries share the model's rate limit: let them finish before timing anything else."""
  started = time.perf_counter()
  while time.perf_counter() - started < timeout_s:
    if all(n["summary"] for n in client.get("/notes").json()):
      return
    time.sleep(0.3)


@contextmanager
def served(app):
  """The app on a real uvicorn server (a thread with its own event loop): TestClient collects whole responses and
  waits for background tasks, so streaming and timing need a real server. Yields the base URL."""
  import uvicorn
  from fastmcp.utilities.tests import find_available_port

  port = find_available_port()
  server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
  thread = threading.Thread(target=server.run, daemon=True)
  thread.start()
  while not server.started:
    time.sleep(0.02)
  try:
    yield f"http://127.0.0.1:{port}"
  finally:
    server.should_exit = True
    thread.join(timeout=15)


# ── Lab 1: a database with SQLModel ────────────────────────────────────────────
def note_models():
  from sqlmodel import Field, SQLModel

  class NoteBase(SQLModel):  # the shared fields
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=50_000)

  class Note(NoteBase, table=True):  # the table: ids and owners only the server sets
    __tablename__ = "lab_note"
    id: int | None = Field(default=None, primary_key=True)
    owner_id: int

  class NoteCreate(NoteBase):  # what clients send
    pass

  class NotePublic(NoteBase):  # what clients see: no owner_id
    id: int

  return Note, NoteCreate, NotePublic


def crud(client) -> list[list[str]]:
  rows = []
  for label, method, path, body in (
    ("create", "POST", "/notes", {"title": "Refunds", "content": "Refunds within 14 days of delivery."}),
    ("create another", "POST", "/notes", {"title": "Oat milk", "content": "Oat milk costs 40 rupees extra."}),
    ("the same title again", "POST", "/notes", {"title": "Refunds", "content": "A duplicate."}),
    ("read", "GET", "/notes/1", None),
    ("list, filtered", "GET", "/notes?q=Oat&limit=5", None),
    ("PATCH one field", "PATCH", "/notes/1", {"content": "Refunds within 14 days."}),
    ("delete", "DELETE", "/notes/2", None),
    ("read the deleted one", "GET", "/notes/2", None),
  ):
    r = client.request(method, path, json=body)
    rows.append([label, f"{method} {path}", str(r.status_code), r.text[:180] or "(no body)"])
  return rows


def session_per_request():
  from nimbus.api.db import get_session

  return get_session  # `with Session(engine) as session: yield session` — closed after every request


@lab("api.database", "A database with SQLModel: tables that are also API models", calls="No model calls")
def database():
  """Notes in SQLite through the Nimbus API: create, read, list with a filter, PATCH, delete, and a duplicate title
  that becomes 409."""
  Note, NoteCreate, NotePublic = note_models()
  with TestClient(lab_app(fake_model()), raise_server_exceptions=False) as client:
    sign_in(client)
    rows = crud(client)
  view = table(["step", "request", "status", "response"], rows)
  fields = table(
    ["model", "fields"],
    [[m.__name__, ", ".join(m.model_fields)] for m in (Note, NoteCreate, NotePublic)],
  )
  return [
    view,
    fields,
    callout(
      f"The duplicate title answered {rows[2][2]} with the error shape, not a 500: the unique constraint is caught "
      "and translated. PATCH changed only the field the client sent, and a deleted note reads like any missing one.",
      "decision",
      "Notes in SQLModel",
    ),
    example("api.l10.1", "Base, table, Create, Public", note_models, "", [fields]),
    example("api.l10.2", "A session per request", session_per_request, "nimbus/api/db.py", [view]),
    example("api.l10.3", "CRUD, pagination, PATCH", crud, "nimbus/api/routers/notes.py", [view]),
    example(
      "api.l10.4",
      "Database errors → HTTP errors",
      crud,
      "save() catches IntegrityError → 409.",
      [table(["step", "status", "response"], [[rows[2][0], rows[2][2], rows[2][3]]])],
    ),
  ]


# ── Lab 2: files and forms ─────────────────────────────────────────────────────
def a_pdf(text: str) -> bytes:
  import pymupdf

  with pymupdf.open() as document:  # a real PDF, made in memory
    document.new_page().insert_text((72, 72), text)
    return document.tobytes()


def uploads(client) -> list[list[str]]:
  rows = []
  for label, files, data in (
    (
      "a PDF and a text file, with a form field",
      [
        ("files", ("rota.pdf", a_pdf("Holiday rota: closed on Diwali."), "application/pdf")),
        ("files", ("prices.txt", b"Oat milk costs 40 rupees.", "text/plain")),
      ],
      {"title_prefix": "HR "},
    ),
    ("an image", [("files", ("photo.png", b"\x89PNG....", "image/png"))], {}),
    ("3 MB of text", [("files", ("huge.txt", b"x" * 3_000_000, "text/plain"))], {}),
  ):
    r = client.post("/notes/upload", files=files, data=data)
    rows.append([label, str(r.status_code), r.text[:200]])
  return rows


def download(client, note_id: int):
  return client.get(f"/notes/{note_id}/download")


@lab("api.files", "Files and forms: upload documents, download results", calls="No model calls")
def files():
  """Upload a PDF and a text file in one request with a form field, an unsupported image (415), a file over the
  limit (413), and download a note as Markdown."""
  from nimbus.api.notes import extract_text

  with TestClient(lab_app(fake_model()), raise_server_exceptions=False) as client:
    sign_in(client)
    rows = uploads(client)
    notes = client.get("/notes").json()
    pdf_note = next(n for n in notes if n["title"].endswith(".pdf"))
    got = download(client, pdf_note["id"])
  view = table(["upload", "status", "response"], rows)
  return [
    view,
    table(["note from the PDF", "content"], [[pdf_note["title"], pdf_note["content"]]]),
    table(["download", "value"], [["Content-Disposition", got.headers.get("content-disposition")], ["body", got.text]]),
    callout(
      f"Two files became two notes in one request ({rows[0][1]}); the image was refused with {rows[1][1]} and the "
      f"big file with {rows[2][1]}, after reading at most limit + 1 bytes.",
      "info",
      "Validate before you process",
    ),
    example("api.l11.1", "UploadFile and Form", uploads, "nimbus/api/routers/notes.py upload()", [view]),
    example(
      "api.l11.2", "Types and size limits", uploads, "", [table(["upload", "status"], [[r[0], r[1]] for r in rows[1:]])]
    ),
    example("api.l11.3", "Text from PDFs", extract_text, "", [code(pdf_note["content"], "text")]),
    example(
      "api.l11.4",
      "Several files in one request",
      uploads,
      "files: list[UploadFile]",
      [table(["upload", "status", "response"], rows[:1])],
    ),
    example(
      "api.l11.5",
      "Downloads with Content-Disposition",
      download,
      "",
      [code(got.headers.get("content-disposition", ""), "text")],
    ),
  ]


# ── Lab 3: streaming ───────────────────────────────────────────────────────────
QUESTION = "When can a customer get a refund, and how much does oat milk cost?"
NOTES = [
  {"title": "Refunds", "content": "Customers can get a refund within 14 days of delivery if the bag is unopened."},
  {"title": "Oat milk", "content": "Oat milk costs 40 rupees extra on any drink."},
]


def read_sse(client: httpx.Client, question: str) -> dict:
  started, first, first_token, events = time.perf_counter(), None, None, []
  with client.stream("POST", "/chat/stream", json={"question": question}) as response:
    event = None
    for line in response.iter_lines():  # lines arrive as the server sends them
      if line.startswith("event:"):
        event = line.split(":", 1)[1].strip()
      elif line.startswith("data:"):
        first = first or time.perf_counter() - started
        if event == "token":
          first_token = first_token or time.perf_counter() - started
        events.append((event, json.loads(line.split(":", 1)[1]), round(time.perf_counter() - started, 2)))
    content_type = response.headers["content-type"]
  total = time.perf_counter() - started
  return {
    "events": events,
    "first_s": first,
    "token_s": first_token or total,
    "total_s": total,
    "content_type": content_type,
  }


def json_lines_app():
  import asyncio
  from collections.abc import AsyncIterable

  from fastapi import FastAPI
  from pydantic import BaseModel

  class Indexed(BaseModel):
    doc: str
    status: str

  app = FastAPI()

  @app.get("/documents/process")
  async def process() -> AsyncIterable[Indexed]:  # one JSON object per line, as each is ready
    for doc in ("menu.md", "refund-policy.md", "handbook.pdf"):
      await asyncio.sleep(0.3)  # parsing, chunking, embedding...
      yield Indexed(doc=doc, status="indexed")

  return app


def read_json_lines(base: str) -> list[list[str]]:
  rows, started = [], time.perf_counter()
  with httpx.Client(base_url=base) as client, client.stream("GET", "/documents/process") as response:
    for line in response.iter_lines():
      if line:
        rows.append([f"{time.perf_counter() - started:.2f} s", line])
    rows.append(["content-type", response.headers["content-type"]])
  return rows


@lab("api.streaming", "Streaming: tokens as they're generated", uses_model=True, calls="2 calls to Mistral")
def streaming():
  """One question about two notes, answered as JSON and as Server-Sent Events from a real uvicorn server: time to
  the first event against the whole answer; then a JSON Lines stream of documents as each is processed."""
  app = lab_app()
  with served(app) as base, httpx.Client(base_url=base, timeout=60) as client:
    sign_in(client)
    for note in NOTES:
      client.post("/notes", json=note)
    wait_for_summaries(client)
    started = time.perf_counter()
    whole = client.post("/chat", json={"question": QUESTION}).json()
    json_s = time.perf_counter() - started
    sse = read_sse(client, QUESTION)
  with served(json_lines_app()) as base:
    lines = read_json_lines(base)
  events = sse["events"]
  tokens = [e for e in events if e[0] == "token"]
  shown = [[e[0], json.dumps(e[1], ensure_ascii=False), f"{e[2]} s"] for e in events[:4]] + [
    ["…", f"{len(tokens)} token events", ""],
    [events[-1][0], json.dumps(events[-1][1]), f"{events[-1][2]} s"],
  ]
  timing = table(
    ["", "first event", "first words of the answer", "whole answer"],
    [
      ["JSON (POST /chat)", f"{json_s:.2f} s", f"{json_s:.2f} s", f"{json_s:.2f} s"],
      ["SSE (POST /chat/stream)", f"{sse['first_s']:.2f} s", f"{sse['token_s']:.2f} s", f"{sse['total_s']:.2f} s"],
    ],
  )
  return [
    timing,
    table(["event", "data", "at"], shown, title=sse["content_type"]),
    answer(whole["answer"], "The JSON answer", {"sources": ", ".join(whole["sources"])}),
    table(["arrived", "line"], lines, title="JSON Lines"),
    callout(
      f"The stream's first event (the sources) arrived after {sse['first_s']:.2f} s and the first words after "
      f"{sse['token_s']:.2f} s; the JSON answer showed nothing until {json_s:.2f} s. The whole answer took "
      f"{sse['total_s']:.2f} s streamed: streaming changes when the user starts reading, not how long the model "
      "takes.",
      "decision",
      "Stream chat answers",
    ),
    example(
      "api.l12.1",
      "Server-Sent Events",
      read_sse,
      "nimbus/api/routers/chat.py chat_stream()",
      [table(["event", "data", "at"], shown)],
    ),
    example(
      "api.l12.2",
      "Token by token",
      read_sse,
      "async for chunk in model.astream(...): yield ServerSentEvent(event='token', ...)",
      [answer("".join(e[1] for e in tokens))],
    ),
    example("api.l12.3", "JSON Lines", json_lines_app, "", [table(["arrived", "line"], lines)]),
    example("api.l12.4", "Reading streams on the client", read_sse, "", [table(["event", "data", "at"], shown)]),
    example("api.l12.5", "First event vs total", read_sse, "", [timing]),
  ]


# ── Lab 4: WebSockets ──────────────────────────────────────────────────────────
def ask_over_socket(websocket, question: str) -> dict:
  websocket.send_text(question)
  parts = []
  while True:
    message = websocket.receive_json()
    if message["type"] == "token":
      parts.append(message["text"])
    else:
      return {"answer": "".join(parts), "token_messages": len(parts), **message}


@lab("api.websockets", "WebSockets: a two-way chat connection", uses_model=True, calls="2 calls to Mistral")
def websockets():
  """A bad token refused before the connection is accepted; then a conversation about the notes over one socket,
  where the follow-up only makes sense with the first turn."""
  from starlette.websockets import WebSocketDisconnect

  with TestClient(lab_app()) as client:
    token = sign_in(client)
    for note in NOTES:
      client.post("/notes", json=note)
    try:
      with client.websocket_connect("/chat/ws?token=not-a-token") as websocket:
        websocket.receive_text()
      refused = "accepted?"
    except WebSocketDisconnect as error:
      refused = f"closed with code {error.code}: {error.reason}"
    with client.websocket_connect(f"/chat/ws?token={token}") as websocket:
      first = ask_over_socket(websocket, "How much does oat milk cost?")
      follow = ask_over_socket(websocket, "And is it the same price on every drink?")
  turns = table(
    ["question", "answer", "token messages", "turns", "sources"],
    [
      [
        "How much does oat milk cost?",
        first["answer"],
        str(first["token_messages"]),
        str(first["turns"]),
        ", ".join(first["sources"]),
      ],
      [
        "And is it the same price on every drink?",
        follow["answer"],
        str(follow["token_messages"]),
        str(follow["turns"]),
        ", ".join(follow["sources"]),
      ],
    ],
  )
  return [
    table(["bad token", "result"], [["?token=not-a-token", refused]]),
    turns,
    callout(
      "The token was checked before accept(): a bad one never gets a connection. The follow-up has no subject of its "
      "own; it was answered from the history this socket kept, and the history ended with the connection.",
      "info",
      "One socket, one conversation",
    ),
    example(
      "api.l13.1", "accept, then receive and send", ask_over_socket, "nimbus/api/routers/chat.py chat_socket()", [turns]
    ),
    example(
      "api.l13.2",
      "Authenticate before accepting",
      ask_over_socket,
      "username_from(token) before websocket.accept()",
      [code(refused, "text")],
    ),
    example("api.l13.3", "Tokens as JSON messages", ask_over_socket, "", [turns]),
    example("api.l13.4", "State per connection", ask_over_socket, "", [turns]),
    example(
      "api.l13.5",
      "WebSocketDisconnect",
      ask_over_socket,
      "except WebSocketDisconnect: the history goes with the connection",
      [turns],
    ),
  ]


# ── Lab 5: background tasks ────────────────────────────────────────────────────
class FailingModel:
  async def ainvoke(self, *args, **kwargs):
    raise ConnectionError("the summary model is down")


def create_and_watch(base: str, note: dict) -> dict:
  with httpx.Client(base_url=base, timeout=60) as client:
    sign_in(client, "watcher")
    started = time.perf_counter()
    created = client.post("/notes", json=note)
    responded = time.perf_counter() - started
    note_id, summary = created.json()["id"], None
    while summary is None and time.perf_counter() - started < 30:  # the summary arrives after the response
      time.sleep(0.2)
      summary = client.get(f"/notes/{note_id}").json()["summary"]
  return {
    "status": created.status_code,
    "response_s": responded,
    "summary_s": time.perf_counter() - started,
    "summary": summary,
  }


def audit_dependency_app(log: list):
  from fastapi import BackgroundTasks, Depends, FastAPI, Request

  def audit(request: Request, tasks: BackgroundTasks) -> None:  # a dependency can add tasks too
    tasks.add_task(log.append, f"audit: {request.method} {request.url.path}")

  app = FastAPI(dependencies=[Depends(audit)])

  @app.get("/ping")
  def ping() -> dict:
    log.append("endpoint returned")
    return {"ok": True}

  return app


@lab("api.background", "Background tasks: work after the response", uses_model=True, calls="1 call to Mistral")
def background():
  """A note saved on a real server: the response comes back at once and its summary is written afterwards; a
  summary model that's down changes nothing for the client; an audit dependency adding its own task."""
  import logging

  with served(lab_app()) as base:
    ok = create_and_watch(base, NOTES[0])
  records = []
  handler = logging.Handler()
  handler.emit = records.append
  logging.getLogger("nimbus.api.notes").addHandler(handler)
  try:
    with served(lab_app(FailingModel())) as base, httpx.Client(base_url=base, timeout=30) as client:
      sign_in(client)
      failed = client.post("/notes", json=NOTES[1])
      time.sleep(1)
  finally:
    logging.getLogger("nimbus.api.notes").removeHandler(handler)
  log = []
  with TestClient(audit_dependency_app(log)) as client:
    client.get("/ping")
  timing = table(
    ["", "value"],
    [
      ["POST /notes answered", f"{ok['status']} after {ok['response_s']:.2f} s"],
      ["summary saved", f"after {ok['summary_s']:.2f} s"],
      ["summary", ok["summary"] or "(none)"],
    ],
  )
  failure = table(
    ["summary model down", "value"],
    [["the client got", f"{failed.status_code}"], ["the log got", records[0].getMessage() if records else "(nothing)"]],
  )
  return [
    timing,
    failure,
    table(["order of events"], [[e] for e in log]),
    callout(
      f"The client had its 201 after {ok['response_s']:.2f} s; the summary took until {ok['summary_s']:.2f} s and "
      "nobody waited for it. When the summary model failed, the client still got its 201 and the failure went to the "
      "log. Tasks live inside the web process: a restart loses them and nothing retries them, so must-not-be-lost "
      "work goes to a queue (Production track).",
      "decision",
      "Summaries in the background",
    ),
    example(
      "api.l14.1", "BackgroundTasks.add_task", create_and_watch, "nimbus/api/routers/notes.py create_note()", [timing]
    ),
    example("api.l14.2", "The client doesn't wait", create_and_watch, "", [timing]),
    example(
      "api.l14.3",
      "Failures are logged, not returned",
      create_and_watch,
      "summarize_in_background logs exceptions",
      [failure],
    ),
    example("api.l14.4", "Dependencies add tasks", audit_dependency_app, "", [table(["order"], [[e] for e in log])]),
    example("api.l14.5", "The limits", create_and_watch, "In-process, lost on restart, no retries.", [timing]),
  ]


# ── Production path: what the user waits for ───────────────────────────────────
CHAT_QUESTIONS = [
  "How much does oat milk cost?",
  "When can a customer get a refund?",
  "Can I get a refund on an opened bag?",
]
NEW_NOTES = [
  {"title": "Delivery", "content": "Deliveries in Mumbai take two days; elsewhere three to five days."},
  {"title": "Loyalty", "content": "Customers earn one Nimbus Bean point for every ten rupees spent."},
  {"title": "Allergens", "content": "The brownie contains nuts; the banana bread contains gluten and eggs."},
]


def inline_summary_app():
  """The alternative: the endpoint writes the summary before answering."""
  from fastapi import Request

  from nimbus.api.db import SessionDep
  from nimbus.api.deps import ChatModelDep
  from nimbus.api.models import Note, NoteCreate, NotePublic
  from nimbus.api.notes import summarize
  from nimbus.api.routers.notes import save
  from nimbus.api.security import CurrentUser

  app = lab_app()

  @app.post("/notes-inline", status_code=201)
  async def create_inline(
    body: NoteCreate, user: CurrentUser, session: SessionDep, model: ChatModelDep, request: Request
  ) -> NotePublic:
    note = Note.model_validate(body, update={"owner_id": user.id})
    note.summary = await summarize(model, note.content)  # the client waits for this
    return save(session, note)

  return app


def evaluate_waits() -> list[dict]:
  app = inline_summary_app()
  inline, streamed = "JSON answer, summary inline", "SSE answer, summary in the background"
  results = {name: {"first": [], "total": [], "save": []} for name in (inline, streamed)}
  with served(app) as base, httpx.Client(base_url=base, timeout=60) as client:
    sign_in(client)
    for note in NOTES:
      client.post("/notes", json=note)
    wait_for_summaries(client)  # nothing queued in the rate limiter while we time
    for question in CHAT_QUESTIONS:
      started = time.perf_counter()
      client.post("/chat", json={"question": question})
      seconds = time.perf_counter() - started
      results[inline]["first"].append(seconds)
      results[inline]["total"].append(seconds)
      sse = read_sse(client, question)
      results[streamed]["first"].append(sse["token_s"])  # the first words of the answer
      results[streamed]["total"].append(sse["total_s"])
    for note in NEW_NOTES:  # all inline saves first, so no background summary is queued ahead of them
      started = time.perf_counter()
      client.post("/notes-inline", json={**note, "title": note["title"] + " (inline)"})
      results[inline]["save"].append(time.perf_counter() - started)
    for note in NEW_NOTES:
      started = time.perf_counter()
      client.post("/notes", json=note)
      results[streamed]["save"].append(time.perf_counter() - started)
    wait_for_summaries(client)  # let the background summaries finish before the server stops

  def p50(values):
    return round(sorted(values)[len(values) // 2] * 1000)

  return [
    {"variant": name, "first_ms": p50(r["first"]), "total_ms": p50(r["total"]), "save_ms": p50(r["save"])}
    for name, r in results.items()
  ]


@lab("api.waits-eval", "Evaluate: what the user waits for", uses_model=True, calls="~12 calls to Mistral")
def waits_eval():
  """Three chat questions answered as JSON and as SSE, and three notes saved with the summary written inline and in
  the background, on a real uvicorn server: time to the first thing on screen, the whole answer, and saving."""
  rows = evaluate_waits()
  before, after = rows
  return [
    scorecard(
      rows,
      [
        Metric("first_ms", "first words on screen (p50)", "lower", "ms"),
        Metric("total_ms", "whole answer (p50)", "lower", "ms"),
        Metric("save_ms", "saving a note (p50)", "lower", "ms"),
      ],
      title="3 questions, 3 notes, a real server",
    ),
    callout(
      f"Streaming put the first words on screen after {after['first_ms']} ms instead of {before['first_ms']} ms, "
      f"with the whole answer about as long ({after['total_ms']} vs {before['total_ms']} ms). Writing the summary in "
      f"the background made saving a note {before['save_ms'] / max(after['save_ms'], 1):.0f}× faster "
      f"({after['save_ms']} vs {before['save_ms']} ms).",
      "decision",
      "Stream answers, summarise in the background",
    ),
  ]
