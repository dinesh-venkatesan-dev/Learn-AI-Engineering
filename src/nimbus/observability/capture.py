"""TraceCapture: a local endpoint that records what a LangSmith client UPLOADS, without sending anything anywhere.

Point a Client at it (api_url=capture.url) and every run, every OTel span, exactly as it would leave the machine, is
kept here: for tests ("no email address reaches the trace"), for privacy reviews, and for the labs, which run without
a LangSmith account. It is not LangSmith: it stores, it doesn't answer queries.

  with TraceCapture() as capture:
    client = Client(api_url=capture.url, api_key="local")
    ... traced code ...
    client.flush()
    capture.runs      # {run_id: {"name", "run_type", "inputs", "outputs", "parent_run_id", "trace_id", ...}}
"""

import gzip
import json
import re
import threading
import time
from contextlib import suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

PART = re.compile(r"^(post|patch)\.([0-9a-f-]{36})(?:\.(inputs|outputs|events|error|extra|attachments.*))?$")


class TraceCapture:
  def __init__(self) -> None:
    self.runs: dict[str, dict] = {}
    self.spans: list[dict] = []
    self.requests: list[tuple[str, int]] = []  # (path, bytes): everything that arrived
    self.feedback: list[dict] = []
    self.url = ""
    self._server = self._thread = None

  # ── what arrives ──
  def _merge(self, run_id: str, data: dict) -> None:
    self.runs.setdefault(run_id, {}).update({k: v for k, v in data.items() if v is not None})

  async def _multipart(self, request: Request) -> None:
    form = await request.form()
    for name, value in form.multi_items():
      if name.startswith("attachment."):  # attachment.<run_id>.<name>: a file stored with the run
        _, run_id, label = name.split(".", 2)
        data = await value.read() if hasattr(value, "read") else value
        self.runs.setdefault(run_id, {}).setdefault("attachments", {})[label] = len(data)
        continue
      match = PART.match(name)
      if not match:
        continue
      raw = await value.read() if hasattr(value, "read") else value
      with suppress(ValueError):
        data = json.loads(raw)
        kind, run_id, field = match.groups()
        self._merge(run_id, {field: data} if field else data)

  def app(self) -> FastAPI:
    app = FastAPI()

    @app.get("/info")
    def info() -> dict:
      config = {
        "use_multipart_endpoint": True,
        "size_limit": 100,
        "size_limit_bytes": 20_000_000,
        "scale_up_qsize_trigger": 1000,
        "scale_up_nthreads_limit": 16,
        "scale_down_nempty_trigger": 4,
      }
      return {"version": "capture", "batch_ingest_config": config, "instance_flags": {}}

    @app.post("/runs/multipart")
    async def multipart(request: Request):
      self.requests.append(("/runs/multipart", int(request.headers.get("content-length", 0))))
      await self._multipart(request)
      return JSONResponse({}, status_code=202)

    @app.post("/runs/batch")
    async def batch(request: Request):
      body = await request.body()
      self.requests.append(("/runs/batch", len(body)))
      data = json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)
      for run in data.get("post", []) + data.get("patch", []):
        self._merge(str(run["id"]), run)
      return JSONResponse({}, status_code=202)

    @app.post("/runs")
    async def single(request: Request):
      run = await request.json()
      self.requests.append(("/runs", 0))
      self._merge(str(run["id"]), run)
      return JSONResponse({}, status_code=202)

    @app.patch("/runs/{run_id}")
    async def patch(run_id: str, request: Request):
      self._merge(run_id, await request.json())
      return JSONResponse({}, status_code=202)

    @app.post("/otel/v1/traces")
    async def otel(request: Request):
      from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

      body = await request.body()
      self.requests.append(("/otel/v1/traces", len(body)))
      message = ExportTraceServiceRequest()
      message.ParseFromString(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)
      for resource in message.resource_spans:
        for scope in resource.scope_spans:
          for span in scope.spans:
            attributes = {
              a.key: a.value.string_value or a.value.int_value or a.value.double_value for a in span.attributes
            }
            self.spans.append(
              {
                "name": span.name,
                "span_id": span.span_id.hex(),
                "parent": span.parent_span_id.hex(),
                "attributes": attributes,
              }
            )
      return JSONResponse({})

    @app.post("/feedback")
    async def feedback(request: Request):
      self.feedback.append(await request.json())
      return JSONResponse({"id": "captured"})

    return app

  # ── lifecycle ──
  def __enter__(self) -> "TraceCapture":
    import uvicorn
    from fastmcp.utilities.tests import find_available_port

    port = find_available_port()
    self._server = uvicorn.Server(uvicorn.Config(self.app(), host="127.0.0.1", port=port, log_level="critical"))
    self._thread = threading.Thread(target=self._server.run, daemon=True)
    self._thread.start()
    while not self._server.started:
      time.sleep(0.02)
    self.url = f"http://127.0.0.1:{port}"
    return self

  def __exit__(self, *exc) -> None:
    self._server.should_exit = True
    self._thread.join(timeout=10)

  # ── reading it ──
  def tree(self, trace_id: str | None = None) -> list[tuple[int, dict]]:
    """Runs as (depth, run), parents before children, ordered by start time."""
    runs = [r | {"id": k} for k, r in self.runs.items() if not trace_id or str(r.get("trace_id")) == trace_id]
    children: dict[str | None, list[dict]] = {}
    for run in runs:
      children.setdefault(str(run["parent_run_id"]) if run.get("parent_run_id") else None, []).append(run)
    out = []

    def walk(parent, depth):
      for run in sorted(children.get(parent, []), key=lambda r: str(r.get("start_time", ""))):
        out.append((depth, run))
        walk(run["id"], depth + 1)

    walk(None, 0)
    return out

  def text(self) -> str:
    return json.dumps(self.runs, default=str)
