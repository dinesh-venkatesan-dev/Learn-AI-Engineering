"""Background jobs run by RQ workers. Workers import jobs by dotted path, so they live in a module, and each job
builds what it needs (a worker is a separate process: nothing from the web app is shared).

  uv run rq worker --url redis://127.0.0.1:6390/0 --worker-class rq.worker.SimpleWorker --with-scheduler reports
"""

import os


def weekly_digest() -> str:
  """A slow job: this week's numbers for every café, then one paragraph from the model."""
  from nimbus.llm import get_chat_model
  from nimbus.ops.data import cafe_names, cafe_week

  rows = [cafe_week(cafe) for cafe in cafe_names()]
  lines = "\n".join(
    f"{r['cafe']}: ₹{r['revenue_rupees']:,} ({r['change_percent']:+.0f}% vs last week), rating {r['average_rating']}"
    for r in rows
  )
  summary = get_chat_model().invoke(
    f"Write a three-sentence weekly digest for Nimbus Coffee's managers. Quote numbers exactly.\n\n{lines}"
  )
  return f"{lines}\n\n{summary.text}"


def flaky_sync(attempts_key: str) -> str:
  """Fails on its first two attempts (like a supplier API that times out), then succeeds."""
  import redis

  with redis.Redis.from_url(os.environ.get("NIMBUS_REDIS_URL", "redis://127.0.0.1:6390/0")) as client:
    attempts = client.incr(attempts_key)
  if attempts < 3:
    raise ConnectionError(f"supplier API timed out (attempt {attempts})")
  return f"synced on attempt {attempts}"
