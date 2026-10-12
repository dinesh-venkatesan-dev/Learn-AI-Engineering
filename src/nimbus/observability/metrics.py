"""What the service is doing, as numbers and as structured log lines.

RequestMetrics   pure ASGI middleware: requests by path and status, a latency histogram, and the request id bound
                 into every log line written while the request runs
metrics_app()    the /metrics endpoint Prometheus scrapes
TOKENS           a counter of model tokens by model, incremented by the code that calls models
setup_json_logs  structlog: one JSON object per line, with the request id, so logs can be filtered and aggregated
"""

import logging
import time

import structlog
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from starlette.responses import Response

REGISTRY = CollectorRegistry()
REQUESTS = Counter("nimbus_requests_total", "HTTP requests", ["path", "status"], registry=REGISTRY)
LATENCY = Histogram(
  "nimbus_request_seconds",
  "Request latency",
  ["path"],
  registry=REGISTRY,
  buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
TOKENS = Counter("nimbus_model_tokens_total", "Model tokens", ["model", "kind"], registry=REGISTRY)
ALERTS = {  # what to alert on, as PromQL
  "error rate > 2%": (
    'sum(rate(nimbus_requests_total{status=~"5.."}[5m])) / sum(rate(nimbus_requests_total[5m])) > 0.02'
  ),
  "p95 latency > 10 s": "histogram_quantile(0.95, sum by (le) (rate(nimbus_request_seconds_bucket[5m]))) > 10",
  "429s > 5%": 'sum(rate(nimbus_requests_total{status="429"}[5m])) / sum(rate(nimbus_requests_total[5m])) > 0.05',
  "tokens per hour": "sum(increase(nimbus_model_tokens_total[1h])) > 2e6",
}


def route_of(scope) -> str:
  """The route template (/notes/{note_id}), not the raw path: one time series per endpoint, not per note."""
  route = scope.get("route")
  return getattr(route, "path", None) or scope.get("path", "")


class RequestMetrics:
  def __init__(self, app):
    self.app = app

  async def __call__(self, scope, receive, send):
    if scope["type"] != "http" or scope["path"] == "/metrics":
      await self.app(scope, receive, send)
      return
    started, status = time.perf_counter(), 500
    request_id = (scope.get("state") or {}).get("request_id", "-")
    structlog.contextvars.bind_contextvars(request_id=request_id, path=scope["path"])

    async def capture(message):
      nonlocal status
      if message["type"] == "http.response.start":
        status = message["status"]
      await send(message)

    try:
      await self.app(scope, receive, capture)
    finally:
      path = route_of(scope)
      REQUESTS.labels(path, str(status)).inc()
      LATENCY.labels(path).observe(time.perf_counter() - started)
      structlog.contextvars.clear_contextvars()


async def metrics_endpoint(request) -> Response:
  return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def setup_json_logs(level: str = "INFO", stream=None) -> None:
  """Every log line, structlog's and the standard library's (uvicorn, the access log, the routers), as ONE JSON
  object: level, time, logger, message, the fields passed in `extra`, and the request id bound for the request."""
  shared = [
    structlog.contextvars.merge_contextvars,  # the request id and path bound by RequestMetrics
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
  ]
  structlog.configure(
    processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    logger_factory=structlog.stdlib.LoggerFactory(),
  )
  handler = logging.StreamHandler(stream)
  handler.setFormatter(
    structlog.stdlib.ProcessorFormatter(
      foreign_pre_chain=[*shared, structlog.stdlib.add_logger_name, structlog.stdlib.ExtraAdder()],
      processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, structlog.processors.JSONRenderer()],
    )
  )
  root = logging.getLogger()
  root.handlers = [handler]
  root.setLevel(level)
