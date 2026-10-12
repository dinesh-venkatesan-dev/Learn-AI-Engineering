"""Notes: CRUD with pagination, uploads (text and PDF) that become notes, downloads, and summaries written in the
background after the response is sent. Every note belongs to the signed-in user; someone else's looks missing."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Form, Query, Request, Response, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from nimbus.api.db import SessionDep
from nimbus.api.deps import ChatModelDep, SettingsDep
from nimbus.api.errors import ApiError, NotFound
from nimbus.api.models import Note, NoteCreate, NotePublic, NoteUpdate
from nimbus.api.notes import UPLOAD_TYPES, extract_text, summarize_in_background
from nimbus.api.security import CurrentUser

router = APIRouter(prefix="/notes", tags=["notes"])


class DuplicateTitle(ApiError):
  status, code = 409, "duplicate_title"


class UnsupportedType(ApiError):
  status, code = 415, "unsupported_media_type"


class TooLarge(ApiError):
  status, code = 413, "too_large"


def owned(session, user, note_id: int) -> Note:
  note = session.get(Note, note_id)
  if note is None or note.owner_id != user.id:  # someone else's note looks exactly like a missing one
    raise NotFound(f"No note {note_id}")
  return note


def save(session, note: Note) -> Note:
  session.add(note)
  try:
    session.commit()
  except IntegrityError as error:  # the unique constraint: a database error becomes a 409, not a 500
    session.rollback()
    raise DuplicateTitle(f"You already have a note called {note.title!r}.") from error
  session.refresh(note)
  return note


@router.post("", status_code=201)
def create_note(
  body: NoteCreate,
  user: CurrentUser,
  session: SessionDep,
  request: Request,
  model: ChatModelDep,
  tasks: BackgroundTasks,
) -> NotePublic:
  note = save(session, Note.model_validate(body, update={"owner_id": user.id}))
  tasks.add_task(summarize_in_background, request.app.state.engine, model, note.id)  # after the response
  return note


@router.get("")
def list_notes(
  user: CurrentUser,
  session: SessionDep,
  offset: Annotated[int, Query(ge=0)] = 0,
  limit: Annotated[int, Query(ge=1, le=50)] = 20,
  q: Annotated[str | None, Query(max_length=50, description="Only notes whose title contains this")] = None,
) -> list[NotePublic]:
  query = select(Note).where(Note.owner_id == user.id)
  if q:
    query = query.where(Note.title.contains(q))
  return session.exec(query.order_by(Note.id.desc()).offset(offset).limit(limit)).all()


@router.get("/{note_id}")
def read_note(note_id: int, user: CurrentUser, session: SessionDep) -> NotePublic:
  return owned(session, user, note_id)


@router.patch("/{note_id}")
def update_note(note_id: int, body: NoteUpdate, user: CurrentUser, session: SessionDep) -> NotePublic:
  note = owned(session, user, note_id)
  note.sqlmodel_update(body.model_dump(exclude_unset=True))  # only the fields the client sent
  return save(session, note)


@router.delete("/{note_id}", status_code=204)
def delete_note(note_id: int, user: CurrentUser, session: SessionDep) -> None:
  session.delete(owned(session, user, note_id))
  session.commit()


@router.post("/upload", status_code=201)
async def upload(
  files: list[UploadFile],
  user: CurrentUser,
  session: SessionDep,
  settings: SettingsDep,
  request: Request,
  model: ChatModelDep,
  tasks: BackgroundTasks,
  title_prefix: Annotated[str, Form(max_length=40)] = "",
) -> list[NotePublic]:
  """Upload text, Markdown or PDF files; each becomes a note."""
  notes = []
  for file in files:
    if file.content_type not in UPLOAD_TYPES:
      raise UnsupportedType(f"{file.filename}: {file.content_type} isn't supported (text, Markdown or PDF).")
    data = await file.read(settings.upload_max_bytes + 1)  # never more than the limit + 1 byte
    if len(data) > settings.upload_max_bytes:
      raise TooLarge(f"{file.filename} is larger than {settings.upload_max_bytes:,} bytes.")
    text = extract_text(file.content_type, data)
    note = Note(title=f"{title_prefix}{file.filename}"[:120], content=text or "(empty)", owner_id=user.id)
    note.source = f"upload:{file.filename}"
    notes.append(save(session, note))
    tasks.add_task(summarize_in_background, request.app.state.engine, model, note.id)
  return notes


@router.get("/{note_id}/download")
def download(note_id: int, user: CurrentUser, session: SessionDep) -> Response:
  note = owned(session, user, note_id)
  filename = "".join(c if c.isalnum() or c in "-_." else "-" for c in note.title)[:60] or "note"
  return Response(
    f"# {note.title}\n\n{note.content}\n",
    media_type="text/markdown",
    headers={"Content-Disposition": f'attachment; filename="{filename}.md"'},
  )
