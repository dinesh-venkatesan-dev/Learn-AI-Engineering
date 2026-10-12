"""Accept now, work later: POST /reports enqueues a job and answers 202 with its id; GET /reports/{id} says how it's
going. The work runs in RQ worker processes, scaled separately from the API."""

import uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from nimbus.api.deps import audit
from nimbus.api.errors import ApiError, NotFound

router = APIRouter(prefix="/reports", tags=["reports"], dependencies=[Depends(audit)])


class QueueDown(ApiError):
  status, code = 503, "queue_unavailable"


class Accepted(BaseModel):
  job_id: str
  status_url: str


class JobStatus(BaseModel):
  job_id: str
  status: str
  result: str | None = None
  retries_left: int | None = None


def queue(request: Request):
  if getattr(request.app.state, "queue", None) is None:
    raise QueueDown("Background jobs are unavailable (no Redis).")
  return request.app.state.queue


@router.post("", status_code=202)
def create_report(request: Request) -> Accepted:
  job = queue(request).enqueue("nimbus.api.jobs.weekly_digest", job_timeout=180, result_ttl=3600)
  return Accepted(job_id=job.id, status_url=f"/reports/{job.id}")


@router.post("/sync", status_code=202)
def start_sync(request: Request) -> Accepted:
  from rq import Retry

  job = queue(request).enqueue(
    "nimbus.api.jobs.flaky_sync", f"attempts:{uuid.uuid4()}", retry=Retry(max=3, interval=[1, 2])
  )
  return Accepted(job_id=job.id, status_url=f"/reports/{job.id}")


@router.get("/{job_id}")
def report_status(job_id: str, request: Request) -> JobStatus:
  from rq.exceptions import NoSuchJobError
  from rq.job import Job

  try:
    job = Job.fetch(job_id, connection=queue(request).connection)
  except NoSuchJobError as error:
    raise NotFound(f"No job {job_id}") from error
  result = job.return_value()
  return JobStatus(
    job_id=job.id,
    status=job.get_status().value,
    result=str(result) if result is not None else None,
    retries_left=job.retries_left,
  )
