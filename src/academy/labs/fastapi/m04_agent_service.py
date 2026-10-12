"""FastAPI · Module 4 labs: from agent to service. The support agent built once and served over HTTP, streamed as
typed events, protected from blocking calls and from bursts, with conversations that survive a restart, and slow
work moved to background jobs. Then the service, measured."""

import asyncio
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
import warnings

import httpx
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from academy.labs.fastapi.m02_data_streaming import served
from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, Settings, get_settings

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
CUSTOMER = {"X-Customer-ID": "c-1042"}  # Priya, signed in on our website; its backend calls with this header


def lab_app(chat_model=None, **settings):
  import tempfile

  from nimbus.api.app import create_app

  folder = tempfile.mkdtemp(prefix="nimbus-agent-lab-")
  config = Settings(api_database_url=f"sqlite:///{folder}/notes.db", **settings)
  return create_app(settings=config, chat_model=chat_model)


WORDS = [  # what the slow fake model streams, one word at a time
  "Your", "order", "NC-20417", "was", "delivered", "on", "the", "thirteenth", "of", "February", "and", "nothing",
  "more", "is", "due.",
]  # fmt: skip


class SlowModel(GenericFakeChatModel):
  """A fake model that takes `delay` seconds per answer (and streams word by word): load without a provider."""

  delay: float = 1.0
  chunks: list = []

  def bind_tools(self, tools, **kwargs):
    return self

  def _generate(self, *args, **kwargs):
    time.sleep(self.delay)  # a BLOCKING wait, like a sync HTTP call
    return super()._generate(*args, **kwargs)

  async def _agenerate(self, *args, **kwargs):
    await asyncio.sleep(self.delay)  # a non-blocking wait
    return super()._generate(*args, **kwargs)

  async def _astream(self, *args, **kwargs):
    for word in WORDS:
      await asyncio.sleep(self.delay / 10)
      self.chunks.append(word)
      yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def slow(delay: float = 1.0) -> SlowModel:
  return SlowModel(messages=itertools.repeat(AIMessage("Your order is on its way.")), delay=delay, chunks=[])


# ── Lab 1: the agent as an API ─────────────────────────────────────────────────
def serve_the_agent():
  from nimbus.api.app import create_app

  # The lifespan builds the agent ONCE, with a checkpointer: each thread_id keeps its own conversation
  app = create_app()
  with TestClient(app) as client:  # requests in-process: no server needed
    first = client.post(
      "/agent/chat", json={"message": "Where is my latest order?", "thread_id": "t-1"}, headers=CUSTOMER
    )
    follow = client.post(
      "/agent/chat", json={"message": "And how much did it cost?", "thread_id": "t-1"}, headers=CUSTOMER
    )
    history = client.get("/agent/threads/t-1", headers=CUSTOMER)
  return first, follow, history


@lab("lc.agent-api", "Serving an agent: the support agent as an HTTP API", uses_model=True, calls="~6 calls to Mistral")
def agent_api():
  """Two turns on one thread through POST /agent/chat (the follow-up needs the first turn), the thread read back,
  and the same request streamed."""
  first, follow, history = serve_the_agent()
  turns = table(
    ["turn", "status", "reply", "actions"],
    [
      [
        "Where is my latest order?",
        str(first.status_code),
        first.json().get("reply", first.text),
        ", ".join(first.json().get("actions", [])),
      ],
      [
        "And how much did it cost?",
        str(follow.status_code),
        follow.json().get("reply", follow.text),
        ", ".join(follow.json().get("actions", [])),
      ],
    ],
  )
  thread = table(["role", "text"], [[m["role"], m["text"]] for m in history.json()])
  return [
    turns,
    thread,
    callout(
      "The follow-up names no order: it was answered from the thread's checkpoint. The agent was built once at "
      "start-up; each request is one ainvoke on it with the thread id and the customer from the request.",
      "info",
      "One agent, many threads",
    ),
    example(
      "lc.l61.1",
      "Build once, with a checkpointer",
      serve_the_agent,
      "nimbus/api/agent_service.py AgentService.start",
      [thread],
    ),
    example("lc.l61.2", "POST /agent/chat", serve_the_agent, "nimbus/api/routers/agent.py chat()", [turns]),
    example("lc.l61.3", "POST /agent/chat/stream", serve_the_agent, "Lab 3 streams it from a real server.", [turns]),
    example("lc.l61.4", "Calling the API", serve_the_agent, "", [turns]),
  ]


# ── Lab 2: typed requests, meaningful errors ───────────────────────────────────
class DownModel(GenericFakeChatModel):
  def bind_tools(self, tools, **kwargs):
    return self

  async def _agenerate(self, *args, **kwargs):
    raise httpx.ConnectError("connection refused")


def built_once(settings_count: dict):
  import nimbus.api.agent_service as service

  original = service.build_support_agent

  def counted(*args, **kwargs):  # how many agents get built?
    settings_count["built"] += 1
    return original(*args, **kwargs)

  service.build_support_agent = counted
  try:
    with TestClient(lab_app(slow(0.01))) as client:
      for _ in range(5):
        client.post("/agent/chat", json={"message": "hi"}, headers=CUSTOMER)
  finally:
    service.build_support_agent = original
  return settings_count["built"]


def failures() -> list[list[str]]:
  rows = []
  with TestClient(lab_app(slow(0.01))) as client:
    for label, body, headers in (
      ("an empty message", {"message": ""}, CUSTOMER),
      ("an unknown field", {"message": "hi", "temperature": 2}, CUSTOMER),
      ("no customer header", {"message": "hi"}, {}),
      ("an unknown customer", {"message": "hi"}, {"X-Customer-ID": "c-9999"}),
    ):
      r = client.post("/agent/chat", json=body, headers=headers)
      rows.append([label, str(r.status_code), r.text[:140]])
    rows.append(["GET /health", str(client.get("/health").status_code), client.get("/health").text])
  with TestClient(lab_app(DownModel(messages=iter([])))) as client:
    r = client.post("/agent/chat", json={"message": "hi"}, headers=CUSTOMER)
    rows.append(
      ["the provider is down", str(r.status_code), f"{r.text[:120]} Retry-After: {r.headers.get('retry-after')}"]
    )
  return rows


async def without_a_server():
  from nimbus.api.app import create_app

  app = create_app(chat_model=slow(0.01))
  async with (
    app.router.lifespan_context(app),
    httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://nimbus") as client,  # in process
  ):
    response = await client.post("/agent/chat", json={"message": "hi"}, headers=CUSTOMER)
  return response.status_code, response.json()


@lab("prod.agent-api", "Your agent as an API: lifespan, types, errors with meaning", calls="No model calls")
def agent_as_api():
  """The agent built once for five requests, four bad requests refused before any model call, a provider outage as
  503 with Retry-After, the health check, and a test through ASGITransport."""
  built = built_once({"built": 0})
  rows = failures()
  status, body = asyncio.run(without_a_server())
  view = table(["request", "status", "response"], rows)
  return [
    table(
      ["", "value"],
      [["agents built for 5 requests", str(built)], ["ASGITransport test", f"{status} {json.dumps(body)}"]],
    ),
    view,
    callout(
      f"{built} agent built for 5 requests. Bad input was refused with 422/404 before the model was called; the "
      "outage became a 503 with Retry-After, which tells a client what to do, instead of a 500.",
      "decision",
      "A typed, honest agent API",
    ),
    example(
      "prod.l01.1",
      "Create expensive things once",
      built_once,
      "nimbus/api/app.py lifespan",
      [code(f"agents built: {built}", "text")],
    ),
    example(
      "prod.l01.2",
      "async def + ainvoke",
      without_a_server,
      "AgentService.ask awaits agent.ainvoke",
      [code(json.dumps(body), "json")],
    ),
    example(
      "prod.l01.3",
      "Pydantic in, Pydantic out",
      failures,
      "routers/agent.py ChatRequest, AgentReply",
      [table(["request", "status", "response"], rows[:4])],
    ),
    example("prod.l01.4", "Errors with meaning", failures, "", [table(["request", "status", "response"], rows[-1:])]),
    example("prod.l01.5", "A /health endpoint", failures, "", [table(["request", "status", "response"], rows[4:5])]),
    example(
      "prod.l01.6", "Tests without a server", without_a_server, "", [code(f"{status} {json.dumps(body)}", "text")]
    ),
  ]


# ── Lab 3: streaming ───────────────────────────────────────────────────────────
def read_events(base: str, message: str, thread: str, stop_after_tokens: int | None = None) -> dict:
  from httpx_sse import connect_sse

  started, events, first = time.perf_counter(), [], None
  with (
    httpx.Client(base_url=base, timeout=60) as client,
    connect_sse(
      client, "POST", "/agent/chat/stream", json={"message": message, "thread_id": thread}, headers=CUSTOMER
    ) as source,
  ):
    for sse in source.iter_sse():  # httpx-sse: one ServerSentEvent at a time, as it arrives
      first = first or time.perf_counter() - started
      events.append((sse.event, json.loads(sse.data), round(time.perf_counter() - started, 2)))  # data is JSON
      if stop_after_tokens and sum(e[0] == "token" for e in events) >= stop_after_tokens:
        break  # the user closed the tab
  return {"events": events, "first_s": first, "total_s": time.perf_counter() - started}


def disconnect_stops_work():
  model = slow(1.0)
  records = []
  handler = logging.Handler()
  handler.emit = records.append
  logger = logging.getLogger("nimbus.api.agent")
  logger.addHandler(handler)
  level, logger.level = logger.level, logging.INFO
  try:
    with served(lab_app(model)) as base:
      read_events(base, "Where is my order?", "t-gone", stop_after_tokens=2)
      time.sleep(1.5)  # time for the server to notice
  finally:
    logger.removeHandler(handler)
    logger.level = level
  return len(model.chunks), [r.getMessage() for r in records]


@lab("prod.streaming", "Streaming over HTTP: typed events as they happen", uses_model=True, calls="~3 calls to Mistral")
def streaming():
  """The support agent streamed from a real uvicorn server: tool, tool_result and token events, the first event
  against the whole answer; then a client that leaves after two tokens, and a server that stops."""
  with served(lab_app()) as base:
    run = read_events(base, "Where is my latest order, and when did it arrive?", "t-stream")
  produced, log = disconnect_stops_work()
  events = run["events"]
  shown = [[e[0], json.dumps(e[1], ensure_ascii=False)[:120], f"{e[2]} s"] for e in events if e[0] != "token"]
  tokens = [e for e in events if e[0] == "token"]
  view = table(
    ["event", "data", "at"], shown + [["token", f"{len(tokens)} events", f"{tokens[0][2] if tokens else '—'} s first"]]
  )
  timing = table(
    ["", "seconds"],
    [
      ["first event", f"{run['first_s']:.2f}"],
      ["first token", f"{tokens[0][2] if tokens else '—'}"],
      ["done", f"{run['total_s']:.2f}"],
    ],
  )
  stopped = table(
    ["", "value"],
    [
      ["words the model produced after a client left at token 2", str(produced)],
      ["server log", "; ".join(log) or "(nothing)"],
    ],
  )
  return [
    view,
    timing,
    answer("".join(e[1] for e in tokens), "The streamed answer"),
    stopped,
    callout(
      f"The first event arrived after {run['first_s']:.2f} s and the answer streamed from "
      f"{tokens[0][2] if tokens else '?'} s to {run['total_s']:.2f} s. When a client left after two tokens, the "
      f"model produced {produced} of {len(WORDS)} words before the server noticed and stopped: no tokens spent on an "
      "answer nobody reads.",
      "decision",
      "Stream the agent's work",
    ),
    example("prod.l02.1", "Server-Sent Events", read_events, "", [view]),
    example(
      "prod.l02.2",
      "Typed events from agent.astream",
      read_events,
      "AgentService.stream: updates → tool / tool_result, messages → token",
      [view],
    ),
    example("prod.l02.3", "The client side: httpx-sse", read_events, "", [view]),
    example("prod.l02.4", "Time to first event", read_events, "", [timing]),
    example("prod.l02.5", "Stop when the client leaves", disconnect_stops_work, "", [stopped]),
  ]


# ── Lab 4: async and concurrency ───────────────────────────────────────────────
def concurrency_app(model):
  from contextlib import asynccontextmanager

  from fastapi import FastAPI, HTTPException, Request
  from langchain.agents import create_agent

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    app.state.agent = create_agent(model, [])
    app.state.slots = asyncio.Semaphore(2)  # at most 2 model calls at a time, across all requests
    yield

  app = FastAPI(lifespan=lifespan)

  @app.post("/blocking")
  async def blocking(request: Request) -> dict:
    result = request.app.state.agent.invoke({"messages": [("user", "hi")]})  # ❌ freezes the event loop
    return {"answer": result["messages"][-1].text}

  @app.post("/async")
  async def non_blocking(request: Request) -> dict:
    result = await request.app.state.agent.ainvoke({"messages": [("user", "hi")]})  # ✅ yields while waiting
    return {"answer": result["messages"][-1].text}

  @app.post("/limited")
  async def limited(request: Request) -> dict:
    async with request.app.state.slots:  # wait for a free slot
      result = await request.app.state.agent.ainvoke({"messages": [("user", "hi")]})
    return {"answer": result["messages"][-1].text}

  @app.post("/with-timeout")
  async def with_timeout(request: Request, seconds: float = 0.5) -> dict:
    try:
      async with asyncio.timeout(seconds):  # give up after `seconds`
        result = await request.app.state.agent.ainvoke({"messages": [("user", "hi")]})
    except TimeoutError as error:
      raise HTTPException(504, f"No answer within {seconds} s") from error
    return {"answer": result["messages"][-1].text}

  return app


async def burst(path: str, n: int = 6) -> tuple[float, list[int]]:
  app = concurrency_app(slow(1.0))
  async with (
    app.router.lifespan_context(app),
    httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://nimbus", timeout=60) as client,
  ):
    started = time.perf_counter()
    responses = await asyncio.gather(*(client.post(path) for _ in range(n)))  # 6 users at once
  return time.perf_counter() - started, sorted({r.status_code for r in responses})


async def timeouts() -> list[list[str]]:
  app = concurrency_app(slow(1.0))
  rows = []
  async with (
    app.router.lifespan_context(app),
    httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://nimbus") as client,
  ):
    for seconds in (0.5, 3):
      r = await client.post("/with-timeout", params={"seconds": seconds})
      rows.append([f"{seconds} s budget", str(r.status_code), r.text])
  return rows


@lab("prod.concurrency", "Async and concurrency: many users, one event loop", calls="No model calls")
def concurrency():
  """Six requests at once, each needing a 1-second model call: a blocking endpoint, an async one, and an async one
  limited to two model calls at a time; then a time budget."""
  rows = []
  for label, path in (
    ("blocking (agent.invoke)", "/blocking"),
    ("async (agent.ainvoke)", "/async"),
    ("async, 2 slots", "/limited"),
  ):
    seconds, codes = asyncio.run(burst(path))
    rows.append({"variant": label, "seconds": round(seconds, 2), "status": ", ".join(map(str, codes))})
  budget = asyncio.run(timeouts())
  card = scorecard(
    rows, [Metric("seconds", "6 requests took", "lower", "text"), Metric("status", "status", "none", "text")]
  )
  limits = table(["request", "status", "response"], budget)
  return [
    card,
    limits,
    callout(
      f"Blocking: {rows[0]['seconds']} s, one after another, because invoke() froze the event loop. Async: "
      f"{rows[1]['seconds']} s, all together. Two slots: {rows[2]['seconds']} s in three waves, never more than two "
      "calls at the provider. The Nimbus agent service uses ainvoke, a semaphore (agent_concurrency) and a time "
      "budget (agent_timeout_s → 504).",
      "decision",
      "ainvoke, a semaphore and a budget",
    ),
    example("prod.l03.1", "Blocking calls freeze the loop", concurrency_app, "", [card]),
    example("prod.l03.2", "Measured: 6 at once", burst, "", [card]),
    example("prod.l03.3", "A semaphore protects the provider", concurrency_app, "", [card]),
    example("prod.l03.4", "Timeouts: 504 instead of hanging", timeouts, "", [limits]),
  ]


# ── Lab 5: conversations that survive restarts ─────────────────────────────────
def conversation_across_restart(checkpointer: str) -> dict:
  thread = f"restart-{checkpointer}-{int(time.time())}"
  with TestClient(lab_app(agent_checkpointer=checkpointer)) as client:  # server 1
    first = client.post(
      "/agent/chat", json={"message": "Where is my latest order?", "thread_id": thread}, headers=CUSTOMER
    )
  with TestClient(lab_app(agent_checkpointer=checkpointer)) as client:  # a restart: a new app, a new process's memory
    before = client.get(f"/agent/threads/{thread}", headers=CUSTOMER).json()
    follow = client.post(
      "/agent/chat", json={"message": "And how much did that order cost?", "thread_id": thread}, headers=CUSTOMER
    )
  return {
    "first": first.json().get("reply", first.text),
    "history_after_restart": len(before),
    "follow": follow.json().get("reply", follow.text),
  }


@lab("prod.persistence", "Conversations that survive restarts", uses_model=True, calls="~8 calls to Mistral")
def persistence():
  """One conversation, a restart in between, with the in-memory checkpointer and with AsyncPostgresSaver. Needs the
  course database: docker compose up -d."""
  rows = []
  for name in ("memory", "postgres"):
    result = conversation_across_restart(name)
    rows.append([name, result["first"], str(result["history_after_restart"]), result["follow"]])
  view = table(
    ["checkpointer", "turn 1 (before the restart)", "messages found after the restart", "turn 2 (after)"], rows
  )
  return [
    view,
    callout(
      f"After the restart the in-memory checkpointer had {rows[0][2]} messages of the thread and the follow-up "
      f"started from nothing; Postgres had {rows[1][2]}, and the follow-up knew which order. The client keeps the "
      "thread id; the conversation lives in the database.",
      "decision",
      "Postgres checkpoints in production",
    ),
    example(
      "prod.l04.1",
      "A thread id over HTTP",
      conversation_across_restart,
      "routers/agent.py: thread ids namespaced by customer",
      [view],
    ),
    example(
      "prod.l04.2",
      "InMemorySaver forgets",
      conversation_across_restart,
      "",
      [table(["checkpointer", "after restart"], [rows[0][::2]])],
    ),
    example(
      "prod.l04.3",
      "AsyncPostgresSaver in the lifespan",
      conversation_across_restart,
      "AgentService.start: from_conn_string + setup()",
      [view],
    ),
    example("prod.l04.4", "Restart, and continue", conversation_across_restart, "", [view]),
    example("prod.l04.5", "Reading a thread's history", conversation_across_restart, "GET /agent/threads/{id}", [view]),
  ]


# ── Lab 6: background jobs ─────────────────────────────────────────────────────
def start_worker(redis_url: str) -> subprocess.Popen:
  """A worker process. SimpleWorker runs jobs in-process (no fork, which macOS dislikes); --with-scheduler runs the
  retries that wait."""
  rq_cli = os.path.join(os.path.dirname(sys.executable), "rq")
  return subprocess.Popen(
    [
      rq_cli,
      "worker",
      "--url",
      redis_url,
      "--worker-class",
      "rq.worker.SimpleWorker",
      "--with-scheduler",
      "--quiet",
      "reports",
    ],
    cwd=PROJECT_DIR,
    env={**os.environ, "NIMBUS_REDIS_URL": redis_url},
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )


def wait_for(client, job_id: str, timeout: float = 120) -> tuple[dict, list[str]]:
  seen, started = [], time.perf_counter()
  while time.perf_counter() - started < timeout:
    status = client.get(f"/reports/{job_id}").json()
    if not seen or seen[-1] != status["status"]:
      seen.append(status["status"])
    if status["status"] in ("finished", "failed"):
      return status, seen
    time.sleep(0.5)
  return status, seen


@lab("prod.jobs", "Background jobs: accept now, work later", uses_model=True, calls="1 call to Mistral (in the worker)")
def jobs():
  """POST /reports answers 202 at once; an RQ worker builds the weekly digest; the client polls. Then a flaky job
  retried twice. Needs Redis (docker compose up -d)."""
  redis_url = get_settings().redis_url
  worker = start_worker(redis_url)
  try:
    with TestClient(lab_app(slow(0.01))) as client:
      started = time.perf_counter()
      accepted = client.post("/reports")
      accepted_s = time.perf_counter() - started
      if accepted.status_code != 202:
        return [callout(f"Background jobs need Redis at {redis_url}: {accepted.text}", "warning", "No Redis")]
      digest, digest_seen = wait_for(client, accepted.json()["job_id"])
      digest_s = time.perf_counter() - started
      flaky, flaky_seen = wait_for(client, client.post("/reports/sync").json()["job_id"])
  finally:
    worker.terminate()
    worker.wait(timeout=15)
  timeline = table(
    ["", "value"],
    [
      ["POST /reports", f"{accepted.status_code} after {accepted_s * 1000:.0f} ms: {accepted.text}"],
      ["statuses the client saw", " → ".join(digest_seen)],
      ["result after", f"{digest_s:.1f} s"],
    ],
  )
  retried = table(
    ["flaky job", "value"],
    [
      ["statuses", " → ".join(flaky_seen)],
      ["result", str(flaky["result"])],
      ["retries left", str(flaky["retries_left"])],
    ],
  )
  return [
    timeline,
    answer(digest.get("result") or "(no result)", "The weekly digest"),
    retried,
    callout(
      f"The API answered in {accepted_s * 1000:.0f} ms with a job id; the digest was ready after {digest_s:.1f} s in "
      f"a worker process. The flaky job failed twice and finished on its third attempt ({flaky['result']}). Jobs "
      "live in Redis, so they survive an API restart, and workers scale on their own.",
      "decision",
      "Slow work in RQ jobs",
    ),
    example("prod.l05.1", "202 + a job id, then poll", wait_for, "routers/reports.py", [timeline]),
    example("prod.l05.2", "RQ: jobs in Redis", start_worker, "", [timeline]),
    example(
      "prod.l05.3", "A worker is a separate process", start_worker, "nimbus/api/jobs.py weekly_digest", [timeline]
    ),
    example(
      "prod.l05.4",
      "Retries need the scheduler",
      wait_for,
      "Retry(max=3, interval=[1, 2]) + --with-scheduler",
      [retried],
    ),
    example(
      "prod.l05.5",
      "Limits: job_timeout, result_ttl",
      wait_for,
      "enqueue(..., job_timeout=180, result_ttl=3600)",
      [timeline],
    ),
  ]


# ── Production path: the service, measured ─────────────────────────────────────
def evaluate_service() -> list[dict]:
  rows = []
  for name in ("memory", "postgres"):
    turns, follow_ok, first_events = 0, 0, []
    for n, (question, follow, fact) in enumerate(
      (
        ("I have a question about my order NC-20417.", "How much did it cost?", "1300|1,300"),  # not the latest
        ("Let's talk about my oldest order.", "When was it delivered?", "13|February"),  # NC-20417, 2026-02-13
      )
    ):
      thread = f"eval-{name}-{n}-{int(time.time())}"
      with served(lab_app(agent_checkpointer=name)) as base:
        run = read_events(base, question, thread)
        first_events.append(run["first_s"])
      with TestClient(lab_app(agent_checkpointer=name)) as client:  # restarted
        reply = client.post("/agent/chat", json={"message": follow, "thread_id": thread}, headers=CUSTOMER).json()
      turns += 1
      follow_ok += re.search(fact, reply.get("reply", "")) is not None
    rows.append(
      {
        "variant": f"checkpointer: {name}",
        "follow_ups": round(follow_ok / turns, 3),
        "first_event_ms": round(sorted(first_events)[len(first_events) // 2] * 1000),
      }
    )
  return rows


@lab("prod.service-eval", "Evaluate: the agent service across a restart", uses_model=True, calls="~12 calls to Mistral")
def service_eval():
  """Two conversations whose second turn only makes sense with the first (which order?), each streamed on a real
  server, then continued after a restart: with the in-memory and the Postgres checkpointer. Needs the course
  database (docker compose up -d)."""
  rows = evaluate_service()
  memory, postgres = rows
  return [
    scorecard(
      rows,
      [
        Metric("follow_ups", "follow-ups right after a restart"),
        Metric("first_event_ms", "first event (p50)", "lower", "ms"),
      ],
      title="2 conversations, a restart in each",
    ),
    callout(
      f"Follow-ups right after a restart: {postgres['follow_ups']:.0%} with Postgres, {memory['follow_ups']:.0%} in "
      f"memory. The first streamed event took {postgres['first_event_ms']} ms with Postgres and "
      f"{memory['first_event_ms']} ms in memory (one model call dominates either way).",
      "decision",
      "Ship the service with Postgres checkpoints",
    ),
  ]
