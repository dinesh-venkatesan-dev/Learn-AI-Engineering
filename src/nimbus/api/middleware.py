"""Code that wraps every request: a request id (from the caller, or new), the time to the response's first byte,
and one access-log line.

Pure ASGI on purpose: @app.middleware("http") (BaseHTTPMiddleware) hides client disconnects from streaming endpoints,
so a stream would keep spending tokens after the user left.
"""

import logging
import time
import uuid

from fastapi import FastAPI

log = logging.getLogger("nimbus.api.access")


class RequestContext:
  def __init__(self, app):
    self.app = app

  async def __call__(self, scope, receive, send):
    if scope["type"] != "http":
      await self.app(scope, receive, send)
      return
    headers = dict(scope["headers"])
    request_id = headers.get(b"x-request-id", b"").decode()[:64] or uuid.uuid4().hex[:12]
    scope.setdefault("state", {})["request_id"] = request_id  # request.state.request_id in handlers
    started, status = time.perf_counter(), 500

    async def send_with_headers(message):
      nonlocal status
      if message["type"] == "http.response.start":
        status = message["status"]
        ms = (time.perf_counter() - started) * 1000
        message["headers"] = [
          *message.get("headers", []),
          (b"x-request-id", request_id.encode()),
          (b"server-timing", f"app;dur={ms:.1f}".encode()),
        ]
      await send(message)

    try:
      await self.app(scope, receive, send_with_headers)  # the rest of the app: routing, dependencies, the endpoint
    finally:
      ms = (time.perf_counter() - started) * 1000
      fields = {
        "method": scope["method"],
        "path": scope["path"],
        "status": status,
        "ms": round(ms, 1),
        "request_id": request_id,
      }
      log.info("%s %s %d %.1fms id=%s", scope["method"], scope["path"], status, ms, request_id, extra=fields)


def install_request_context(app: FastAPI) -> None:
  app.add_middleware(RequestContext)
