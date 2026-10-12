"""Ratings and corrections from users, about one answer each (its run_id). Stored in our database, always: the data
is ours. When tracing to LangSmith is on, each rating is also attached to its trace, after the response."""

from fastapi import APIRouter, BackgroundTasks
from sqlmodel import select

from nimbus.api.db import SessionDep
from nimbus.api.deps import SettingsDep
from nimbus.api.models import Feedback, FeedbackCreate
from nimbus.api.security import CurrentUser
from nimbus.observability.tracing import langsmith_ready, tracing_client

router = APIRouter(prefix="/feedback", tags=["feedback"])


def forward(settings, item: Feedback) -> None:
  client = tracing_client(settings)
  client.create_feedback(
    item.run_id,
    item.key,
    score=item.score,
    value=item.value,
    comment=item.comment,
    correction={"answer": item.correction} if item.correction else None,
  )


@router.post("", status_code=201)
def rate(body: FeedbackCreate, user: CurrentUser, session: SessionDep, settings: SettingsDep, tasks: BackgroundTasks):
  item = Feedback(**body.model_dump(), user_id=user.id)
  session.add(item)
  session.commit()
  session.refresh(item)
  if langsmith_ready(settings):
    tasks.add_task(forward, settings, item)
  return {"id": item.id, "run_id": item.run_id}


@router.get("")
def mine(user: CurrentUser, session: SessionDep, run_id: str | None = None) -> list[Feedback]:
  query = select(Feedback).where(Feedback.user_id == user.id)
  if run_id:
    query = query.where(Feedback.run_id == run_id)
  return session.exec(query.order_by(Feedback.id.desc()).limit(50)).all()
