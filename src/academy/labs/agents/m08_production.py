"""Agents with LangGraph · Module 8 labs: runtime context, async and concurrency, error handling per node, advanced
fault tolerance, testing graphs, and LangGraph Server. Then the capstone: the production support desk on every
evaluation the agents track built, plus an outage drill."""

import asyncio
import json
import operator
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model


def _thread(name: str) -> dict:
  return {"configurable": {"thread_id": name}}


# ── Lab 1: runtime context ─────────────────────────────────────────────────────
@dataclass
class Channel:
  """Per-run data from the app (read-only): who asks and where the reply goes. Defaults: callers pass what differs."""

  customer: str = "a customer"
  channel: Literal["app", "sms", "email"] = "app"
  language: str = "English"


class ReplyState(TypedDict, total=False):
  question: str
  reply: str


LIMITS = {
  "app": "two short sentences",
  "sms": "under 160 characters, no formatting",
  "email": "a short email with a greeting and a sign-off",
}


def context_graph():
  from langgraph.graph import START, StateGraph
  from langgraph.runtime import Runtime

  def write(state: ReplyState, runtime: Runtime[Channel]) -> dict:
    ctx = runtime.context  # typed, read-only, per run
    model = get_chat_model(role="fallback") if ctx.channel == "email" else get_chat_model()  # even the model
    prompt = f"You are Nimbus Coffee support writing to {ctx.customer} in {ctx.language}: {LIMITS[ctx.channel]}."
    return {"reply": model.invoke([("system", prompt), ("human", state["question"])]).text}

  return StateGraph(ReplyState, context_schema=Channel).add_node("write", write).add_edge(START, "write").compile()


def config_vs_context(graph):
  config = {"configurable": {"thread_id": "x"}, "recursion_limit": 5, "tags": ["lab"]}  # for LangGraph itself
  return graph.invoke({"question": "Do you deliver on Sundays?"}, config, context=Channel(channel="sms"))  # your data


@lab(
  "agents.runtime-context",
  "Runtime context: one graph, many customers and channels",
  uses_model=True,
  calls="~4 calls (Mistral, and Groq for email)",
)
def runtime_context():
  """One question answered by one graph for three channels: the context picks the prompt, the length, the language
  and the model."""
  graph = context_graph()
  runs = {}
  for ctx in (
    Channel(),
    Channel(customer="Priya", channel="sms"),
    Channel(customer="Farah", channel="email"),
    Channel(language="Hindi"),
  ):
    started = time.perf_counter()
    reply = graph.invoke({"question": "Do you deliver on Sundays?"}, context=ctx)["reply"]
    runs[f"{ctx.channel} · {ctx.language}"] = (reply, len(reply), (time.perf_counter() - started) * 1000)
  sms = config_vs_context(graph)
  return [
    table(["Context", "reply", "characters", "ms"], [[k, r, n, f"{ms:.0f}"] for k, (r, n, ms) in runs.items()]),
    callout(
      "context_schema declares typed, read-only data passed per run with invoke(context=...): the customer, the "
      "channel, the language. Nodes read it through runtime.context and choose behaviour from it, even the model "
      "(email went to Groq here). config is for LangGraph (thread_id, recursion_limit, tags); context is your app's "
      "data. Dataclass defaults let callers pass only what differs. The support desk takes the logged-in customer "
      "this way, so the model can never choose whose orders to read. This demo graph has no tools, so read the "
      "replies for their shape, not their facts.",
      "decision",
      "Runtime context",
    ),
  ] + [
    example(
      "lg.l45.1",
      "context_schema and invoke(context=...)",
      context_graph,
      "",
      [table(["context", "reply"], [[k, r] for k, (r, _, _) in runs.items()])],
    ),
    example(
      "lg.l45.2",
      "Behaviour from context: prompt, language, model",
      context_graph,
      "",
      [table(["context", "characters"], [[k, n] for k, (_, n, _) in runs.items()])],
    ),
    example("lg.l45.3", "config vs context", config_vs_context, "", [answer(sms["reply"], "sms")]),
    example("lg.l45.4", "Defaults", Channel, "", [code(str(Channel()), "python")]),
  ]


# ── Lab 2: async and concurrency ───────────────────────────────────────────────
REQUESTS = [
  ("c-1042", "When is my next coffee delivery?"),
  ("c-7007", "When will my order ship?"),
  ("c-6060", "Can I still return NC-90112? It's unopened."),
  ("c-5005", "Where is my order NC-78123?"),
]


async def one_request(desk, store, customer, text):
  from nimbus.support.agent import CustomerContext

  payload = {"messages": [HumanMessage(text)], "customer_id": customer, "actions": []}
  return await desk.ainvoke(payload, context=CustomerContext(customer, store.customers[customer]))  # async: ainvoke


async def sequential_vs_gather(desk, store):
  started = time.perf_counter()
  for customer, text in REQUESTS:
    await one_request(desk, store, customer, text)
  one_by_one = time.perf_counter() - started
  started = time.perf_counter()
  await asyncio.gather(*(one_request(desk, store, c, t) for c, t in REQUESTS))  # the waiting overlaps
  together = time.perf_counter() - started
  return one_by_one, together


def max_concurrency():
  from langgraph.graph import START, StateGraph
  from langgraph.types import Send

  class Fan(TypedDict, total=False):
    items: list[int]
    done: Annotated[list[float], operator.add]

  async def work(task: dict) -> dict:
    await asyncio.sleep(0.2)  # a slow API call
    return {"done": [time.perf_counter()]}

  graph = (
    StateGraph(Fan)
    .add_node("work", work)
    .add_conditional_edges(START, lambda s: [Send("work", {"i": i}) for i in s["items"]], ["work"])
    .compile()
  )
  out = {}
  for limit in (None, 2):
    started = time.perf_counter()
    asyncio.run(graph.ainvoke({"items": list(range(6)), "done": []}, {"max_concurrency": limit} if limit else {}))
    out["no limit" if limit is None else f"max_concurrency={limit}"] = time.perf_counter() - started
  return out


def sync_node_in_async_graph():
  from langgraph.graph import START, StateGraph

  class S(TypedDict, total=False):
    log: Annotated[list[str], operator.add]

  def blocking(state: S) -> dict:
    time.sleep(0.5)  # sync code: LangGraph runs it in a thread pool
    return {"log": ["sync done"]}

  async def ticker():
    ticks = 0
    while ticks < 10:
      await asyncio.sleep(0.05)
      ticks += 1
    return ticks

  graph = StateGraph(S).add_node("blocking", blocking).add_edge(START, "blocking").compile()

  async def main():
    started = time.perf_counter()
    _, ticks = await asyncio.gather(graph.ainvoke({"log": []}), ticker())  # the loop keeps ticking meanwhile
    return ticks, time.perf_counter() - started

  return asyncio.run(main())


@lab("agents.async", "Async graphs and concurrency", uses_model=True, calls="~24 calls to Mistral")
def async_lab():
  """Four customer requests through the support desk one after another and with asyncio.gather, a fan-out with and
  without max_concurrency, and a blocking sync node inside an async run."""
  from nimbus.llm.models import aclose_clients
  from nimbus.support.desk import build_support_desk
  from nimbus.support.tools import SupportStore

  store = SupportStore()

  async def run():
    try:
      desk = build_support_desk(get_chat_model(timeout_s=60), store)  # its own client: closed in this event loop
      return await sequential_vs_gather(desk, store)
    finally:
      await aclose_clients()

  one_by_one, together = asyncio.run(run())
  limits = max_concurrency()
  ticks, seconds = sync_node_in_async_graph()
  rps = get_settings().chat_requests_per_second
  return [
    table(
      ["Four desk requests", "seconds"],
      [["one after another", f"{one_by_one:.1f}"], ["asyncio.gather", f"{together:.1f}"]],
      title=f"Speed-up {one_by_one / together:.1f}×",
    ),
    table(["Six 0.2 s tasks fanned out", "seconds"], [[k, f"{v:.2f}"] for k, v in limits.items()]),
    table(
      ["A 0.5 s blocking sync node", "Result"],
      [["event-loop ticks meanwhile", ticks], ["total seconds", f"{seconds:.2f}"]],
    ),
    callout(
      f"Async nodes and ainvoke/astream are how graphs run inside web servers. asyncio.gather ran four desk "
      f"requests together: {one_by_one:.1f} s → {together:.1f} s ({one_by_one / together:.1f}×). "
      + (
        f"Barely faster, because the bottleneck here is the shared client-side rate limiter ({rps} requests/s): "
        "the same model calls take the same time however they're scheduled. Concurrency pays off when the time "
        "goes to waiting for slow responses, not when a limiter spaces the requests out. "
        if one_by_one / together < 1.3
        else "The waiting overlapped. "
      )
      + "max_concurrency caps one run's parallel tasks; a sync node runs in a thread pool and doesn't block the loop.",
      "decision",
      "Async",
    ),
  ] + [
    example(
      "lg.l46.1",
      "Async nodes and ainvoke",
      one_request,
      "",
      [table(["requests", "seconds"], [[4, f"{one_by_one:.1f}"]])],
    ),
    example(
      "lg.l46.2",
      "Many requests at once",
      sequential_vs_gather,
      "",
      [table(["one by one", "gather"], [[f"{one_by_one:.1f}", f"{together:.1f}"]])],
    ),
    example(
      "lg.l46.3",
      "max_concurrency",
      max_concurrency,
      "",
      [table(["run", "seconds"], [[k, f"{v:.2f}"] for k, v in limits.items()])],
    ),
    example(
      "lg.l46.4",
      "Sync code in async graphs",
      sync_node_in_async_graph,
      "",
      [table(["ticks", "seconds"], [[ticks, f"{seconds:.2f}"]])],
    ),
  ]


# ── Lab 3: error handling per node ─────────────────────────────────────────────
class Tracking(TypedDict, total=False):
  order_id: str
  status: str
  note: str


def retries_then_handler():
  from langgraph.errors import NodeError
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Command, RetryPolicy

  attempts = {"flaky": 0, "down": 0}

  def flaky(state: Tracking) -> dict:
    attempts["flaky"] += 1
    if attempts["flaky"] < 3:
      raise ConnectionError("tracking API unavailable")
    return {"status": "in transit"}

  def down(state: Tracking) -> dict:
    attempts["down"] += 1
    raise ConnectionError("courier API down")

  def last_known(state: Tracking, error: NodeError) -> Command:
    return Command(
      update={"status": "unknown", "note": f"{error.node} failed ({error.error}); showing the last known status"},
      goto=END,
    )

  retry = RetryPolicy(max_attempts=3, initial_interval=0.05, retry_on=ConnectionError)
  one = StateGraph(Tracking).add_node("flaky", flaky, retry_policy=retry).add_edge(START, "flaky").compile()
  two = (
    StateGraph(Tracking)
    .add_node("down", down, retry_policy=retry, error_handler=last_known)
    .add_edge(START, "down")
    .compile()
  )
  return one.invoke({"order_id": "NC-78123"}), two.invoke({"order_id": "NC-78123"}), attempts


def timeout_policy():
  from langgraph.errors import NodeError, NodeTimeoutError
  from langgraph.graph import END, START, StateGraph
  from langgraph.types import Command, TimeoutPolicy

  async def slow_courier(state: Tracking) -> dict:
    await asyncio.sleep(3)
    return {"status": "delivered"}

  def politely(state: Tracking, error: NodeError) -> Command:
    kind = "timed out" if isinstance(error.error, NodeTimeoutError) else "failed"
    return Command(update={"status": "unknown", "note": f"{error.node} {kind}; we'll email an update"}, goto=END)

  graph = StateGraph(Tracking).add_node(
    "slow_courier", slow_courier, timeout=TimeoutPolicy(run_timeout=0.5), error_handler=politely
  )
  graph = graph.add_edge(START, "slow_courier").compile()
  started = time.perf_counter()
  result = asyncio.run(graph.ainvoke({"order_id": "NC-78123"}))  # timeouts need async nodes
  return result, time.perf_counter() - started


def node_defaults_and_cache():
  from langgraph.cache.memory import InMemoryCache
  from langgraph.graph import START, StateGraph
  from langgraph.types import CachePolicy, RetryPolicy

  calls = []

  def eta(state: Tracking) -> dict:
    calls.append(state["order_id"])
    time.sleep(0.3)  # an expensive model or API call
    return {"note": f"ETA for {state['order_id']}: 2 days"}

  builder = StateGraph(Tracking).set_node_defaults(retry_policy=RetryPolicy(max_attempts=2))  # one policy for all
  graph = (
    builder.add_node("eta", eta, cache_policy=CachePolicy(ttl=300))
    .add_edge(START, "eta")
    .compile(cache=InMemoryCache())
  )
  times = []
  for order in ("NC-78123", "NC-78123", "NC-12500"):
    started = time.perf_counter()
    graph.invoke({"order_id": order})
    times.append((order, round(time.perf_counter() - started, 2)))
  return times, calls


@lab("agents.error-handling", "Error handling per node: retries, fallbacks, timeouts, caching", calls="No model calls")
def error_handling():
  """A tracking step that fails twice, a courier that's down, a courier that hangs, one policy for every node, and
  a cached ETA."""
  flaky, fallback, attempts = retries_then_handler()
  timed, seconds = timeout_policy()
  times, calls = node_defaults_and_cache()
  return [
    table(
      ["Node", "Result"],
      [
        ["retry_policy (fails twice)", f"{flaky['status']} after {attempts['flaky']} attempts"],
        ["retries run out → error_handler", f"{fallback['note']} ({attempts['down']} attempts)"],
        ["timeout (TimeoutPolicy 0.5 s on a 3 s call)", f"{timed['note']} after {seconds:.1f} s"],
        ["cache_policy: same order twice", f"{times} · computed for {calls}"],
      ],
    ),
    callout(
      "Per node: retry_policy retries chosen exceptions with backoff; error_handler runs when retries run out, gets "
      "the NodeError and decides what's next; a TimeoutPolicy stops an async node that takes too long; "
      "set_node_defaults applies one policy everywhere; cache_policy reuses a result for the same input. The "
      "production desk retries its agent step on temporary errors, switches to the Groq fallback from the second "
      "attempt, and hands the ticket to a person when every attempt fails.",
      "decision",
      "Error handling",
    ),
  ] + [
    example(
      "lg.l47.1",
      "retry_policy",
      retries_then_handler,
      "",
      [table(["status", "attempts"], [[flaky["status"], attempts["flaky"]]])],
    ),
    example("lg.l47.2", "error_handler", retries_then_handler, "", [answer(fallback["note"], "handled")]),
    example(
      "lg.l47.3", "timeout", timeout_policy, "", [table(["note", "seconds"], [[timed["note"], f"{seconds:.1f}"]])]
    ),
    example(
      "lg.l47.4",
      "set_node_defaults",
      node_defaults_and_cache,
      "",
      [table(["order", "seconds"], [list(t) for t in times])],
    ),
    example("lg.l47.5", "cache_policy", node_defaults_and_cache, "", [table(["computed for"], [[", ".join(calls)]])]),
  ]


# ── Lab 4: advanced fault tolerance ────────────────────────────────────────────
class Job(TypedDict, total=False):
  order_id: str
  result: str
  results: Annotated[list[str], operator.add]


class OrderNotFound(Exception):
  """A business error: retrying never fixes it."""


def retry_rule():
  from langgraph.graph import START, StateGraph
  from langgraph.types import RetryPolicy, default_retry_on

  def retry_on(error: BaseException) -> bool:
    return not isinstance(error, OrderNotFound) and (isinstance(error, ConnectionError) or default_retry_on(error))

  tries = {"n": 0}

  def lookup(state: Job) -> dict:
    tries["n"] += 1
    if state["order_id"] == "NC-99999":
      raise OrderNotFound(state["order_id"])
    if tries["n"] < 2:
      raise ConnectionError("connection reset")
    return {"result": f"{state['order_id']} found"}

  graph = (
    StateGraph(Job)
    .add_node("lookup", lookup, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05, retry_on=retry_on))
    .add_edge(START, "lookup")
    .compile()
  )
  out = {}
  for order in ("NC-78123", "NC-99999"):
    tries["n"] = 0
    try:
      out[order] = f"{graph.invoke({'order_id': order})['result']} ({tries['n']} attempts)"
    except OrderNotFound:
      out[order] = f"OrderNotFound, not retried ({tries['n']} attempt)"
  return out


def backup_on_retry():
  from langgraph.graph import START, StateGraph
  from langgraph.runtime import Runtime
  from langgraph.types import RetryPolicy

  def ship(state: Job, runtime: Runtime) -> dict:
    attempt = runtime.execution_info.node_attempt  # the same idea as the desk's fallback model
    if attempt == 1:
      raise ConnectionError("primary courier down")
    return {"result": f"booked with the backup courier on attempt {attempt}"}

  graph = (
    StateGraph(Job)
    .add_node("ship", ship, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05))
    .add_edge(START, "ship")
    .compile()
  )
  return graph.invoke({"order_id": "NC-12500"})["result"]


def timeouts_and_heartbeats():
  from langgraph.config import get_stream_writer
  from langgraph.errors import NodeTimeoutError
  from langgraph.graph import START, StateGraph
  from langgraph.runtime import Runtime
  from langgraph.types import TimeoutPolicy

  async def chatty(state: Job) -> dict:
    write = get_stream_writer()
    for page in range(6):  # 1.8 s, with output every 0.3 s
      await asyncio.sleep(0.3)
      write({"page": page})  # stream output counts as progress
    return {"result": "report done"}

  async def stuck(state: Job) -> dict:
    await asyncio.sleep(5)
    return {"result": "never"}

  async def quiet_batch(state: Job, runtime: Runtime) -> dict:
    for _ in range(5):
      await asyncio.sleep(0.3)
      runtime.heartbeat()  # in heartbeat mode, only this counts
    return {"result": "batch done"}

  async def run(node, policy):
    graph = StateGraph(Job).add_node("work", node, timeout=policy).add_edge(START, "work").compile()
    started = time.perf_counter()
    try:
      return f"{(await graph.ainvoke({'order_id': 'x'}))['result']} after {time.perf_counter() - started:.1f}s"
    except NodeTimeoutError as error:
      return f"NodeTimeoutError ({error.kind}) after {error.elapsed:.1f}s"

  async def main():
    heartbeat = TimeoutPolicy(idle_timeout=0.5, refresh_on="heartbeat")
    return {
      "chatty, idle_timeout 0.5 s": await run(chatty, TimeoutPolicy(idle_timeout=0.5)),
      "chatty, run_timeout 1.0 s": await run(chatty, TimeoutPolicy(run_timeout=1.0)),
      "stuck, idle_timeout 0.5 s": await run(stuck, TimeoutPolicy(idle_timeout=0.5)),
      "batch with heartbeats": await run(quiet_batch, heartbeat),
      "chatty in heartbeat mode": await run(chatty, heartbeat),
    }

  return asyncio.run(main())


def per_send_timeouts():
  from langgraph.errors import NodeTimeoutError
  from langgraph.graph import START, StateGraph
  from langgraph.types import RetryPolicy, Send, TimeoutPolicy

  async def quote(task: dict) -> dict:
    await asyncio.sleep(task["delay"])
    return {"results": [f"{task['supplier']}: quote"]}

  def fan_out(state: Job):
    suppliers = [("Green Valley", 0.2, 1.0), ("Hill Estates", 0.4, 1.0), ("Slow Farms", 2.0, 0.5)]
    return [
      Send("quote", {"supplier": s, "delay": d}, timeout=TimeoutPolicy(run_timeout=limit)) for s, d, limit in suppliers
    ]

  graph = (
    StateGraph(Job)
    .add_node("quote", quote, retry_policy=RetryPolicy(max_attempts=1))
    .add_conditional_edges(START, fan_out, ["quote"])
    .compile()
  )
  try:
    asyncio.run(graph.ainvoke({"results": []}))
    return "all quotes in"
  except NodeTimeoutError as error:
    return f"{error.node} timed out after {error.elapsed:.1f}s (its own limit 0.5 s; the others had 1.0 s)"


def graceful_shutdown():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.errors import GraphDrained
  from langgraph.graph import START, StateGraph
  from langgraph.runtime import RunControl

  def step(name):
    def node(state: Job) -> dict:
      time.sleep(0.4)
      return {"results": [name]}

    return node

  graph = (
    StateGraph(Job)
    .add_node("pack", step("pack"))
    .add_node("label", step("label"))
    .add_node("dispatch", step("dispatch"))
  )
  graph = (
    graph.add_edge(START, "pack")
    .add_edge("pack", "label")
    .add_edge("label", "dispatch")
    .compile(checkpointer=InMemorySaver())
  )
  control = RunControl()
  timer = threading.Timer(0.6, control.request_drain, args=("sigterm",))  # e.g. a deploy's SIGTERM
  timer.start()
  try:
    graph.invoke({"results": []}, _thread("drain"), control=control)
    drained = "finished before the drain"
  except GraphDrained as error:
    state = graph.get_state(_thread("drain"))
    drained = f"drained ({error.reason}) after {state.values['results']}, next {state.next}"
  finally:
    timer.cancel()
  return drained, graph.invoke(None, _thread("drain"))["results"]


@lab(
  "agents.fault-tolerance",
  "Advanced fault tolerance: smarter retries, idle timeouts, heartbeats, graceful shutdown",
  calls="No model calls",
)
def fault_tolerance():
  """A retry rule that never retries business errors, a backup on later attempts, hard and idle timeouts,
  heartbeats, per-task timeouts in a fan-out, and a run drained at a step boundary and resumed."""
  rule = retry_rule()
  backup = backup_on_retry()
  timeouts = timeouts_and_heartbeats()
  per_send = per_send_timeouts()
  drained, resumed = graceful_shutdown()
  return [
    table(
      ["Case", "Result"],
      [
        *[[f"retry_on: {k}", v] for k, v in rule.items()],
        ["backup on attempt 2", backup],
        *[[k, v] for k, v in timeouts.items()],
        ["per-Send timeouts", per_send],
        ["graceful shutdown", drained],
        ["resumed after the restart", str(resumed)],
      ],
    ),
    callout(
      "Retry what can succeed later (network errors) and never what can't (an order that doesn't exist). The retry "
      "attempt number can switch to a backup. run_timeout is a hard cap; idle_timeout only stops a node that stops "
      "making progress, and in heartbeat mode only runtime.heartbeat() counts. Each Send can carry its own timeout. "
      "RunControl.request_drain() stops at the next step boundary, so a deploy can stop runs and resume them later "
      "without repeating a step.",
      "decision",
      "Fault tolerance",
    ),
  ] + [
    example(
      "lg.l48.1", "retry_on with your own rule", retry_rule, "", [table(["order", "result"], list(rule.items()))]
    ),
    example("lg.l48.2", "Retry state: a backup on later attempts", backup_on_retry, "", [code(backup, "text")]),
    example(
      "lg.l48.3",
      "run_timeout vs idle_timeout",
      timeouts_and_heartbeats,
      "",
      [table(["case", "result"], [[k, v] for k, v in list(timeouts.items())[:3]])],
    ),
    example(
      "lg.l48.4",
      "Heartbeat mode",
      timeouts_and_heartbeats,
      "",
      [table(["case", "result"], [[k, v] for k, v in list(timeouts.items())[3:]])],
    ),
    example("lg.l48.5", "Per-Send timeouts", per_send_timeouts, "", [code(per_send, "text")]),
    example(
      "lg.l48.6", "Graceful shutdown", graceful_shutdown, "", [table(["drained", "resumed"], [[drained, str(resumed)]])]
    ),
  ]


# ── Lab 5: testing graphs ──────────────────────────────────────────────────────
def _test_module(name: str):
  """A test file loaded as a module (the tests folder isn't a package), to show its functions' code."""
  import importlib.util

  spec = importlib.util.spec_from_file_location(f"academy_lab_{name}", PROJECT_DIR / "tests" / f"{name}.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def run_tests(path: str) -> list[tuple[str, str, float]]:
  """Run a test file with pytest in a subprocess and read each test's outcome and duration."""
  report = PROJECT_DIR / ".cache" / "labs" / "agents" / "pytest-report.xml"
  report.parent.mkdir(parents=True, exist_ok=True)
  subprocess.run(
    [sys.executable, "-m", "pytest", "-q", path, f"--junitxml={report}", "-p", "no:cacheprovider"],
    cwd=PROJECT_DIR,
    capture_output=True,
    timeout=300,
    check=False,
  )
  import xml.etree.ElementTree as ET  # noqa: N817, S405 - our own pytest report

  rows = []
  for case in ET.parse(report).getroot().iter("testcase"):  # noqa: S314
    outcome = "failed" if case.find("failure") is not None or case.find("error") is not None else "passed"
    rows.append((case.get("name"), outcome, float(case.get("time", 0))))
  return rows


@lab("agents.testing", "Testing graphs: fast, free, deterministic", calls="No model calls")
def testing():
  """The support desk's tests: plain functions without a graph, the whole graph with a scripted model, one part of
  the graph from an injected state, an approval interrupt, and an outage."""
  desk_tests, graph_tests, approval_tests = (
    _test_module(n) for n in ("test_desk", "test_desk_graph", "test_approvals")
  )
  rows = run_tests("tests/test_desk.py") + run_tests("tests/test_desk_graph.py") + run_tests("tests/test_approvals.py")
  total = sum(t for _, _, t in rows)
  return [
    table(
      ["Test", "result", "seconds"],
      [[n, o, f"{t:.2f}"] for n, o, t in rows],
      title=f"{len(rows)} tests in {total:.1f} s, no API calls",
    ),
    callout(
      f"{sum(o == 'passed' for _, o, _ in rows)}/{len(rows)} passed in {total:.1f} s. Test node logic and routing "
      "as plain "
      "functions; test the whole graph with a fake chat model that returns scripted messages (the real tools still "
      "run against the order data); test one part by injecting state with update_state(as_node=...); test "
      "approvals by asserting the interrupt and resuming. One trap found here: routing that reads remaining_steps "
      "sees 2 inside update_state, so inject state as a node whose next step doesn't depend on it.",
      "decision",
      "Testing",
    ),
  ] + [
    example(
      "lg.l49.1",
      "Nodes and routing as plain functions",
      desk_tests.test_reply_language_is_decided_from_the_message,
      "",
      [table(["test", "result"], [[n, o] for n, o, _ in rows if "language" in n or "claims" in n])],
    ),
    example(
      "lg.l49.2",
      "The whole graph with a fake model",
      graph_tests.test_the_whole_graph_with_a_scripted_model,
      "",
      [table(["test", "result"], [[n, o] for n, o, _ in rows if "scripted" in n or "outage" in n])],
    ),
    example(
      "lg.l49.3",
      "One part, from an injected state",
      graph_tests.test_one_part_of_the_graph_from_an_injected_state,
      "",
      [table(["test", "result"], [[n, o] for n, o, _ in rows if "injected" in n])],
    ),
    example(
      "lg.l49.4",
      "Human-in-the-loop flows",
      graph_tests.test_an_approval_pauses_the_run_and_resume_completes_it,
      "Also test_only_sensitive_calls_are_asked_about_and_edits_keep_the_message_id: " + "tests/test_approvals.py",
      [table(["test", "result"], [[n, o] for n, o, _ in rows if "approval" in n or "sensitive" in n])],
    ),
  ]


# ── Lab 6: LangGraph Server ────────────────────────────────────────────────────
SERVER_DIR = PROJECT_DIR / "server"
PORT = 2025
URL = f"http://127.0.0.1:{PORT}"


def start_server() -> subprocess.Popen:
  import httpx

  cli = SERVER_DIR.parent / ".venv" / "bin" / "langgraph"
  log = (PROJECT_DIR / ".cache" / "langgraph-dev.log").open("w")
  process = subprocess.Popen(
    [str(cli), "dev", "--no-browser", "--port", str(PORT)], cwd=SERVER_DIR, stdout=log, stderr=subprocess.STDOUT
  )
  log.close()  # the child has its own handle
  for _ in range(90):
    try:
      if httpx.get(f"{URL}/ok", timeout=1).status_code == 200:
        return process
    except httpx.HTTPError:
      time.sleep(1)
  process.terminate()
  process.wait()
  raise RuntimeError("LangGraph server did not start: see .cache/langgraph-dev.log")


def use_the_api():
  from langgraph_sdk import get_sync_client

  client = get_sync_client(url=URL)
  out = {"assistants": [(a["graph_id"], a["assistant_id"][:8]) for a in client.assistants.search()]}
  thread = client.threads.create()
  context = {"customer_id": "c-1042", "name": "Priya S."}  # the logged-in customer, per run
  question = {
    "messages": [{"role": "user", "content": "When is my next coffee delivery?"}],
    "customer_id": "c-1042",
    "actions": [],
  }
  out["streamed"] = [
    list(part.data)
    for part in client.runs.stream(thread["thread_id"], "desk", input=question, context=context, stream_mode="updates")
    if part.event == "updates"
  ]
  out["reply"] = client.threads.get_state(thread["thread_id"])["values"]["ticket"]["reply"]
  refund = {
    "messages": [{"role": "user", "content": "Please refund NC-90112, it's unopened."}],
    "customer_id": "c-6060",
    "actions": [],
  }
  thread = client.threads.create()
  client.runs.wait(thread["thread_id"], "desk", input=refund, context={"customer_id": "c-6060", "name": "Farah Z."})
  state = client.threads.get_state(thread["thread_id"])
  out["paused at"] = state["next"]
  out["waiting for"] = state["interrupts"][0]["value"] if state["interrupts"] else None
  result = client.runs.wait(
    thread["thread_id"],
    "desk",
    command={"resume": {"decision": "approve"}},
    context={"customer_id": "c-6060", "name": "Farah Z."},
  )
  out["after approval"] = result["ticket"]["reply"] if result.get("ticket") else result
  out["actions"] = result.get("actions")
  return out


@lab(
  "agents.server",
  "LangGraph Server: the desk as an API with threads, runs and streaming",
  uses_model=True,
  calls="~8 calls to Mistral; starts and stops `langgraph dev`",
)
def server():
  """The production desk served by `langgraph dev` from server/langgraph.json, used through the Python SDK: a streamed
  run, the thread's saved state, and a refund approved over the API."""
  config = (SERVER_DIR / "langgraph.json").read_text()
  process = start_server()
  try:
    out = use_the_api()
  finally:
    process.terminate()
    process.wait()
  return [
    code(config, "json", title="server/langgraph.json"),
    table(
      ["Step", "Result"],
      [[k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v] for k, v in out.items()],
    ),
    callout(
      "langgraph.json names the graphs to host; `langgraph dev` serves them with persistence, a run queue and "
      "streaming built in (the desk is compiled without a checkpointer: the server brings its own). The SDK works "
      "with assistants (hosted graphs), threads (conversations) and runs (executions); the customer goes in each "
      "run's context. A refund paused at the review step until the API resumed it with a Command.",
      "decision",
      "LangGraph Server",
    ),
  ] + [
    example(
      "lg.l50.1",
      "langgraph.json",
      start_server,
      "server/langgraph.json and server/desk_graph.py.",
      [code(config, "json")],
    ),
    example("lg.l50.2", "`langgraph dev`", start_server, "", [table(["server"], [[URL]])]),
    example(
      "lg.l50.3", "Assistants, threads, runs", use_the_api, "", [table(["assistants"], [[str(out["assistants"])]])]
    ),
    example(
      "lg.l50.4",
      "Streaming a run; the thread's state",
      use_the_api,
      "",
      [table(["updates from", "reply"], [[str(out["streamed"]), out["reply"]]])],
    ),
    example(
      "lg.l50.5",
      "Human-in-the-loop over the API",
      use_the_api,
      "",
      [table(["paused at", "after approval"], [[str(out["paused at"]), str(out["after approval"])]])],
    ),
  ]


# ── Capstone: the production support desk on everything ───────────────────────
def outage_drill(scenarios) -> list[dict]:
  from langchain.chat_models import init_chat_model

  from nimbus.support.agent_eval import AgentRun, score
  from nimbus.support.desk import ask, build_support_desk
  from nimbus.support.tools import SupportStore

  s = get_settings()
  down = init_chat_model(s.chat_model, model_provider=s.chat_provider, endpoint="http://127.0.0.1:9", max_retries=0)
  rows = []
  for name, fallback in (
    ("primary down, no fallback", None),
    ("primary down, Groq fallback", get_chat_model(role="fallback")),
  ):
    for scenario in scenarios:
      store = SupportStore()
      desk = build_support_desk(down, store, fallback=fallback, retries=2)
      started = time.perf_counter()
      result = ask(desk, store, scenario["customer_id"], scenario["request"])
      ticket = result["ticket"]
      from nimbus.support.agent import SupportOutcome

      outcome = SupportOutcome(
        reply=ticket.reply, resolved=ticket.resolved, needs_human=ticket.needs_human, order_ids=ticket.order_ids
      )
      calls = [
        {"name": c["name"], "args": c["args"]}
        for m in result["messages"]
        for c in getattr(m, "tool_calls", [])
        if c["name"] != "final_answer"
      ]
      run = AgentRun(outcome, result.get("actions", []), calls, 0, None, (time.perf_counter() - started) * 1000)
      rows.append(
        {
          "variant": name,
          "id": scenario["id"],
          "ok": score(scenario, run)["success"],
          "human": ticket.needs_human,
          "ms": run.ms,
          "reply": ticket.reply,
        }
      )
  return rows


def evaluate_capstone() -> dict:
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.support import approval_eval, memory_eval
  from nimbus.support.agent_eval import load_scenarios, run_desk_scenario, score

  model, out = get_chat_model(), {}
  scenarios = load_scenarios(get_settings().eval_agent_scenarios) + load_scenarios(
    PROJECT_DIR / "data" / "eval" / "agent_scenarios_hard.json"
  )
  out["agent scenarios"] = [score(s, run_desk_scenario(model, s))["success"] for s in scenarios]
  embeddings = get_embeddings(get_settings())
  conversations = memory_eval.load_conversations(PROJECT_DIR / "data" / "eval" / "memory_conversations.json")
  out["memory conversations"] = [
    memory_eval.score(c, memory_eval.run_conversation(model, c, "threads+memory", embeddings)) for c in conversations
  ]
  approvals = approval_eval.load_scenarios(PROJECT_DIR / "data" / "eval" / "approval_scenarios.json")
  results = [(a, approval_eval.run_scenario(model, a, True)) for a in approvals]
  out["approvals"] = [
    (lambda sc: sc["actions_ok"] and sc["reply_ok"] and not sc["unrequested"])(approval_eval.score(a, r))
    for a, r in results
  ]
  out["drill"] = outage_drill([s for s in scenarios if s["id"] in ("s01", "s04", "s07", "s10", "h04")])
  return out


@lab(
  "agents.capstone-eval",
  "Capstone: the production support desk on every evaluation",
  uses_model=True,
  calls="~200 calls to Mistral and ~15 to Groq: about 12 minutes",
)
def capstone_eval():
  """The support desk as it ships: on the 15 agent scenarios (Module 4), the 12 memory conversations (Module 5), the 8
  approval scenarios (Module 6), and an outage drill with the primary model unreachable."""
  started = time.perf_counter()
  out = evaluate_capstone()
  seconds = time.perf_counter() - started
  suites = [
    {"variant": name, "passed": sum(results), "total": len(results), "rate": sum(results) / len(results)}
    for name, results in out.items()
    if name != "drill"
  ]
  drill = out["drill"]
  drill_rows = []
  for name in ("primary down, no fallback", "primary down, Groq fallback"):
    mine = [r for r in drill if r["variant"] == name]
    drill_rows.append(
      {
        "variant": name,
        "answered": sum(r["ok"] for r in mine) / len(mine),
        "to a person": sum(r["human"] for r in mine),
        "p50_ms": round(statistics.median(r["ms"] for r in mine)),
      }
    )
  passed = sum(s["passed"] for s in suites)
  total = sum(s["total"] for s in suites)
  return [
    scorecard(
      suites,
      [Metric("passed", "passed", "higher", "int"), Metric("total", "of", "none", "int"), Metric("rate", "rate")],
      title=f"The production desk · {seconds:.0f} s",
    ),
    scorecard(
      drill_rows,
      [
        Metric("answered", "answered correctly"),
        Metric("to a person", "handed to a person", "none", "int"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title="Outage drill: the primary model unreachable, 5 scenarios",
    ),
    table(
      ["Drill", "id", "ok", "reply"], [[r["variant"], r["id"], "✓" if r["ok"] else "✗", r["reply"]] for r in drill]
    ),
    callout(
      f"The desk as it ships passed {passed}/{total} across the three suites, and with the primary model unreachable "
      f"it answered {drill_rows[1]['answered']:.0%} of the drill through the Groq fallback (without the fallback every "
      "customer got the apology and a person took over). Each module's decision is still visible in it: the "
      "final_answer ending (4), per-turn state and customer memories (5), approvals before money moves (6), and "
      "retries with a fallback and a person as the last resort (8).",
      "decision",
      "Capstone",
    ),
  ]
