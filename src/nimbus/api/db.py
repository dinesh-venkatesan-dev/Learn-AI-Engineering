"""The API's database: an engine created once (in the lifespan) and one session per request."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request
from sqlmodel import Session, SQLModel, create_engine


def make_engine(url: str):
  if url.startswith("sqlite:///"):
    Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
  return create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})


def create_tables(engine) -> None:
  import nimbus.api.models  # noqa: F401 - registers the tables on SQLModel.metadata

  SQLModel.metadata.create_all(engine)


def get_session(request: Request) -> Iterator[Session]:
  """One session per request, closed afterwards even if the endpoint raised."""
  with Session(request.app.state.engine) as session:
    yield session


SessionDep = Annotated[Session, Depends(get_session)]
