"""Tables and API shapes with SQLModel: one class can be a table AND a Pydantic model.

User / UserPublic                         the password hash never leaves
Note (table) / NoteCreate / NoteUpdate / NotePublic   clients can't set ids, owners or summaries
"""

from datetime import UTC, datetime

from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def now() -> datetime:
  return datetime.now(UTC)


class User(SQLModel, table=True):
  id: int | None = Field(default=None, primary_key=True)
  username: str = Field(index=True, unique=True, min_length=3, max_length=40)
  password_hash: str  # never returned by the API


class UserPublic(SQLModel):
  id: int
  username: str


class NoteBase(SQLModel):
  title: str = Field(min_length=1, max_length=120)
  content: str = Field(min_length=1, max_length=50_000)


class Note(NoteBase, table=True):
  __table_args__ = (UniqueConstraint("owner_id", "title"),)  # one title per user: a duplicate is a 409

  id: int | None = Field(default=None, primary_key=True)
  owner_id: int = Field(foreign_key="user.id", index=True)
  source: str = "typed"  # typed / upload:<filename>
  summary: str | None = None  # written in the background after the note is saved
  created_at: datetime = Field(default_factory=now)


class NoteCreate(NoteBase):
  pass


class NoteUpdate(SQLModel):
  title: str | None = Field(default=None, min_length=1, max_length=120)
  content: str | None = Field(default=None, min_length=1, max_length=50_000)


class NotePublic(NoteBase):
  id: int
  source: str
  summary: str | None
  created_at: datetime


class Feedback(SQLModel, table=True):
  """A user's rating of one answer, kept in our own database (and sent to LangSmith when tracing is on)."""

  id: int | None = Field(default=None, primary_key=True)
  run_id: str = Field(index=True)
  user_id: int = Field(foreign_key="user.id", index=True)
  key: str = Field(max_length=40)  # "thumbs", "helpful", "wrong_order", ...
  score: float | None = None
  value: str | None = Field(default=None, max_length=40)
  comment: str | None = Field(default=None, max_length=1000)
  correction: str | None = Field(default=None, max_length=2000)
  created_at: datetime = Field(default_factory=now)


class FeedbackCreate(BaseModel):
  run_id: str = PydanticField(min_length=8, max_length=64)
  key: str = PydanticField(default="thumbs", pattern=r"^[a-z_]{2,40}$")
  score: float | None = PydanticField(default=None, ge=0, le=1)
  value: str | None = PydanticField(default=None, max_length=40)
  comment: str | None = PydanticField(default=None, max_length=1000)
  correction: str | None = PydanticField(default=None, max_length=2000)
