"""Chat about your notes: one JSON answer, a Server-Sent Events stream (sources, tokens, done), or a WebSocket that
keeps the conversation for as long as it's open. Links are removed from every answer: notes are untrusted text."""

from collections.abc import AsyncIterable

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from nimbus.api.db import SessionDep
from nimbus.api.deps import ChatModelDep
from nimbus.api.errors import Unauthorized, ask_model
from nimbus.api.models import User
from nimbus.api.notes import LinkFilter, notes_prompt, relevant_notes, strip_links
from nimbus.api.security import CurrentUser, username_from
from nimbus.api.streaming import in_own_task
from nimbus.llm.errors import classify
from nimbus.observability.tracing import run_config

router = APIRouter(prefix="/chat", tags=["chat"])


class Question(BaseModel):
  question: str = Field(min_length=3, max_length=1000)


class ChatAnswer(BaseModel):
  answer: str
  sources: list[str]
  run_id: str  # POST /feedback with it to rate this answer


@router.post("")
async def chat(body: Question, user: CurrentUser, session: SessionDep, model: ChatModelDep) -> ChatAnswer:
  notes = relevant_notes(session, user.id, body.question)
  run_id, config = run_config("notes-chat", user=user.username, sources=len(notes))  # the trace's id, chosen here
  reply = await ask_model(model, notes_prompt(body.question, notes), config=config)  # busy provider → 503
  return ChatAnswer(answer=strip_links(reply.text), sources=[n.title for n in notes], run_id=run_id)


@router.post("/stream", response_class=EventSourceResponse)
async def chat_stream(
  body: Question, user: CurrentUser, session: SessionDep, model: ChatModelDep
) -> AsyncIterable[ServerSentEvent]:
  notes = relevant_notes(session, user.id, body.question)
  yield ServerSentEvent(event="sources", data=[n.title for n in notes])  # at once: the reader sees what's used
  links = LinkFilter()  # URLs are removed even when they arrive split across tokens
  try:
    async for chunk in in_own_task(model.astream(notes_prompt(body.question, notes))):  # stops if the client leaves
      if chunk.text and (text := links.feed(chunk.text)):
        yield ServerSentEvent(event="token", data=text)
  except Exception as error:  # noqa: BLE001 - the status line is already sent: report the failure as an event
    retry = classify(error).action == "retry"
    yield ServerSentEvent(event="error", data={"code": "model_unavailable" if retry else "model_error"})
    return
  if rest := links.flush():
    yield ServerSentEvent(event="token", data=rest)
  yield ServerSentEvent(event="done", data="")


@router.websocket("/ws")
async def chat_socket(websocket: WebSocket, token: str = Query()):
  """Authenticate with ?token=... (browsers can't set headers on WebSockets), then ask; answers stream back."""
  app = websocket.app
  try:
    username = username_from(token, app.state.settings.jwt_secret.get_secret_value())
  except Unauthorized:
    await websocket.close(code=1008, reason="invalid token")  # refused before accepting
    return
  await websocket.accept()
  history: list[tuple[str, str]] = []  # this conversation, for as long as the socket is open
  try:
    while True:
      question = await websocket.receive_text()
      with Session(app.state.engine) as session:
        user = session.exec(select(User).where(User.username == username)).one()
        notes = relevant_notes(session, user.id, question + " " + " ".join(q for role, q in history if role == "human"))
      answer, links = "", LinkFilter()
      async for chunk in app.state.chat_model.astream(notes_prompt(question, notes, history)):
        if chunk.text and (text := links.feed(chunk.text)):
          answer += text
          await websocket.send_json({"type": "token", "text": text})
      if rest := links.flush():
        answer += rest
        await websocket.send_json({"type": "token", "text": rest})
      history += [("human", question), ("ai", answer)]
      await websocket.send_json({"type": "done", "sources": [n.title for n in notes], "turns": len(history) // 2})
  except WebSocketDisconnect:
    pass  # the user closed the tab: the history goes with the connection
