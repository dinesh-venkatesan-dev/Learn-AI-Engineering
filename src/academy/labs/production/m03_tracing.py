"""Production · Module 3 labs: tracing and feedback. Every lab runs the real LangSmith SDK, pointed at a local
TraceCapture: what the SDK would upload is recorded on this machine and nothing leaves it. Steps that READ from
LangSmith's servers (read_run, list_runs, feedback configs, annotation queues) run only when Nimbus has its own key
(NIMBUS_LANGSMITH_API_KEY + NIMBUS_LANGSMITH_TRACING=true in ai-course/.env); otherwise they show what they would do."""

import io
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, redirect_stdout, suppress

import langsmith as ls
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langsmith import Client, traceable

from academy.labs.runtime import Metric, callout, code, example, lab, scorecard, table
from nimbus.config import get_settings
from nimbus.observability.capture import TraceCapture
from nimbus.observability.tracing import langsmith_ready, pii_anonymizer, run_config

NEEDS_KEY = "needs LangSmith: set NIMBUS_LANGSMITH_API_KEY and NIMBUS_LANGSMITH_TRACING=true in ai-course/.env"


@contextmanager
def captured(**client_options):
  """A Client whose uploads land in a local TraceCapture, and a tracing context that uses it."""
  with TraceCapture() as capture:
    client = Client(api_url=capture.url, api_key="local-only", **client_options)
    with ls.tracing_context(client=client, enabled=True, project_name="nimbus-lab"):
      yield capture, client
    client.flush()


def fake(*replies: str) -> GenericFakeChatModel:
  return GenericFakeChatModel(messages=iter([AIMessage(r) for r in replies] * 10))


def tree_rows(capture, trace_id: str | None = None) -> list[list[str]]:
  return [
    [
      "  " * depth + str(run.get("name")),
      str(run.get("run_type")),
      json.dumps(run.get("outputs") or {}, default=str)[:90],
    ]
    for depth, run in capture.tree(trace_id)
  ]


def server(fn):
  """Run fn(client) against LangSmith when Nimbus has a key; otherwise say what's needed."""
  settings = get_settings()
  if not langsmith_ready(settings):
    return NEEDS_KEY
  from nimbus.observability.tracing import tracing_client

  return fn(tracing_client(settings))


# ── Lab 1: tracing basics (LangChain) ──────────────────────────────────────────
def a_chain():
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  prompt = ChatPromptTemplate.from_messages([("system", "Answer in five words."), ("human", "{question}")])
  return prompt | fake("Cold brew steeps for hours.") | StrOutputParser()  # prompt → model → parser


def console_trace():
  from langchain_core.tracers import ConsoleCallbackHandler

  output = io.StringIO()
  with redirect_stdout(output):  # the full trace, printed locally: every step's inputs and outputs
    a_chain().invoke({"question": "What is cold brew?"}, {"callbacks": [ConsoleCallbackHandler()]})
  return output.getvalue()


CHAIN_SCRIPT = (  # a plain script: no Nimbus imports, so nothing but the environment decides about tracing
  "from langchain_core.language_models.fake_chat_models import GenericFakeChatModel\n"
  "from langchain_core.messages import AIMessage\n"
  "from langchain_core.output_parsers import StrOutputParser\n"
  "from langchain_core.prompts import ChatPromptTemplate\n"
  "prompt = ChatPromptTemplate.from_messages([('system', 'Answer in five words.'), ('human', '{question}')])\n"
  "model = GenericFakeChatModel(messages=iter([AIMessage('Cold brew steeps for hours.')]))\n"
  "chain = prompt | model | StrOutputParser()\n"
  "print(chain.invoke({'question': 'What is cold brew?'}))\n"
)


def tracing_by_environment(capture_url: str) -> str:
  # No code changes: three environment variables, and every LangChain call is traced
  env = {**os.environ, "LANGSMITH_TRACING": "true", "LANGSMITH_ENDPOINT": capture_url, "LANGSMITH_API_KEY": "local"}
  script = CHAIN_SCRIPT + "from langchain_core.tracers.langchain import wait_for_all_tracers; wait_for_all_tracers()"
  result = subprocess.run(
    [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60, check=False
  )
  return result.stdout.strip()


@traceable(name="answer_question")
def answer_question(question: str) -> str:
  return a_chain().invoke({"question": question})  # the chain's runs nest under this function's run


@lab("lc.tracing", "Tracing: every step of every run", calls="No model calls")
def lc_tracing():
  """A prompt → model → parser chain traced three ways: printed by ConsoleCallbackHandler, sent by the SDK switched
  on with environment variables only (to the local capture), and nested under a @traceable function."""
  printed = console_trace()
  with TraceCapture() as capture:
    out = tracing_by_environment(capture.url)
    time.sleep(0.5)
    by_env = tree_rows(capture)
  with captured() as (capture, _):
    answer_question("What is cold brew?")
  nested = tree_rows(capture)
  view = table(["run", "type", "outputs"], nested)
  return [
    view,
    table(["run", "type", "outputs"], by_env, title=f"switched on by environment variables only → {out}"),
    code(printed[:1500], "text", "ConsoleCallbackHandler"),
    callout(
      f"One call, {len(nested)} runs: the function, the chain inside it, and the prompt, model and parser inside "
      "that. Setting LANGSMITH_TRACING and an endpoint traced the same chain with no code change.",
      "info",
      "A trace is a tree of runs",
    ),
    example("lc.l53.1", "A tree of runs", a_chain, "", [view]),
    example("lc.l53.2", "ConsoleCallbackHandler", console_trace, "", [code(printed[:800], "text")]),
    example(
      "lc.l53.3",
      "On with environment variables",
      tracing_by_environment,
      "",
      [table(["run", "type", "outputs"], by_env)],
    ),
    example("lc.l53.4", "@traceable", answer_question, "", [view]),
  ]


# ── Lab 2: your first trace ────────────────────────────────────────────────────
def run_id_up_front(chain, question: str):
  run_id = uuid.uuid4()  # chosen BEFORE the call, so this code can find its own trace
  answer = chain.invoke({"question": question}, {"run_id": run_id})
  return run_id, answer


def read_back(run_id) -> str:
  return server(lambda client: client.read_run(str(run_id)).outputs)  # with a key: the run, from LangSmith


@lab("smith.first-trace", "Your first trace", calls="No model calls")
def first_trace():
  """One chain call with its run id chosen up front, the runs it produced (as uploaded), flush, and reading it back
  from LangSmith when Nimbus has a key."""
  with captured() as (capture, client):
    run_id, answer = run_id_up_front(a_chain(), "What is a flat white?")
    pending = len(capture.runs)
    client.flush()  # uploads happen in a background thread: flush before reading back or exiting
    flushed = len(capture.runs)
  root = capture.runs.get(str(run_id), {})
  rows = tree_rows(capture)
  view = table(["run", "type", "outputs"], rows)
  return [
    view,
    table(
      ["", "value"],
      [
        ["run id chosen up front", str(run_id)],
        [
          "the root run, as uploaded",
          json.dumps({k: root.get(k) for k in ("name", "run_type", "inputs", "outputs")}, default=str)[:300],
        ],
        ["runs received before / after flush()", f"{pending} / {flushed}"],
        ["read_run from LangSmith", str(read_back(run_id))],
      ],
    ),
    example("smith.l01.1", "An API key and LANGSMITH_TRACING", tracing_by_environment, "", [view]),
    example("smith.l01.2", "A trace is a tree of runs", run_id_up_front, "", [view]),
    example("smith.l01.3", "Choosing the run id", run_id_up_front, "", [code(str(run_id), "text")]),
    example("smith.l01.4", "Reading it back", read_back, "", [code(str(read_back(run_id)), "text")]),
    example(
      "smith.l01.5",
      "Flush before reading or exiting",
      run_id_up_front,
      "client.flush()",
      [code(f"{pending} → {flushed} runs", "text")],
    ),
  ]


# ── Lab 3: tracing agents ──────────────────────────────────────────────────────
class ScriptedAgentModel(FakeMessagesListChatModel):
  def bind_tools(self, tools, **kwargs):
    return self


def traced_agent():
  from langchain.agents import create_agent

  from nimbus.support.tools import SupportStore, build_tools

  steps = [
    AIMessage("", tool_calls=[{"name": "get_order", "args": {"order_id": "NC-78123"}, "id": "c1"}]),
    AIMessage("NC-78123 has shipped."),
  ]
  agent = create_agent(ScriptedAgentModel(responses=steps), build_tools(SupportStore()))
  config = {
    "run_name": "support-agent",
    "tags": ["support", "v2"],
    "metadata": {"customer": "c-5005", "app_version": "2.3"},
  }
  return agent.invoke({"messages": [("user", "Where is NC-78123?")]}, config)  # tags and metadata reach every run


def find_by(capture, tag: str, key: str, value: str) -> list[str]:
  # In LangSmith: client.list_runs(project_name=..., filter='and(has(tags, "v2"), eq(metadata_key, ...))')
  return [
    r.get("name")
    for r in capture.runs.values()
    if tag in (r.get("tags") or []) and (r.get("extra") or {}).get("metadata", {}).get(key) == value
  ]


@lab("smith.agents", "Tracing agents, and making traces easy to find", calls="No model calls")
def agents():
  """A support agent with a scripted model and the real order tool: its whole loop in one trace, the run types, and
  the run name, tags and metadata on every run."""
  with captured() as (capture, _):
    traced_agent()
  rows = tree_rows(capture)
  types = sorted({str(r.get("run_type")) for r in capture.runs.values()})
  found = find_by(capture, "v2", "customer", "c-5005")
  view = table(["run", "type", "outputs"], rows)
  return [
    view,
    table(
      ["", "value"],
      [
        ["run types in this trace", ", ".join(types)],
        ["runs tagged v2 for customer c-5005", f"{len(found)} of {len(capture.runs)}"],
      ],
    ),
    callout(
      f"The agent's loop is one trace of {len(rows)} runs: the graph, each model call, the tool call with its "
      "result. The run name, tags and metadata given once in config were copied to every run, so a filter on "
      "them finds the whole trace.",
      "info",
      "Why it answered, in one tree",
    ),
    example("smith.l02.1", "The tool loop in one trace", traced_agent, "", [view]),
    example("smith.l02.2", "Run types", traced_agent, "", [code(", ".join(types), "text")]),
    example("smith.l02.3", "run_name, tags, metadata", traced_agent, "", [view]),
    example("smith.l02.4", "Finding traces by tag or metadata", find_by, "", [code(f"{len(found)} runs", "text")]),
  ]


# ── Lab 4: @traceable ──────────────────────────────────────────────────────────
@traceable(run_type="retriever", name="find_policy")
def find_policy(question: str) -> list[dict]:
  return [
    {"page_content": "Refunds within 14 days of delivery, unopened bags only.", "metadata": {"source": "refunds.html"}}
  ]


@traceable(run_type="tool", name="refund_check", process_inputs=lambda inputs: {"order_id": inputs["order"]["id"]})
def refund_check(order: dict) -> str:
  return "eligible" if order["opened"] is False else "not eligible"  # the whole order isn't recorded: just its id


@traceable(name="handle_ticket", process_outputs=lambda output: {"length": len(output)})
def handle_ticket(question: str, order: dict) -> str:
  ls.get_current_run_tree().add_metadata({"priority": "high"})  # added from inside the running function
  find_policy(question)
  verdict = refund_check(order)
  with ls.trace("draft_reply", run_type="chain", inputs={"verdict": verdict}) as block:  # a block, not a function
    reply = f"Your order {order['id']} is {verdict} for a refund."
    block.end(outputs={"reply": reply})
  return reply


@traceable(name="fails")
def fails() -> None:
  raise ValueError("the order service timed out")


@traceable(name="with_photo")
def with_photo(photo: ls.schemas.Attachment) -> str:
  return "a torn bag"  # the photo is stored with the run, not just the text read from it


@lab("smith.traceable", "@traceable: your own code in the trace", calls="No model calls")
def traceable_lab():
  """A ticket handler traced with @traceable: a retriever and a tool step, inputs and outputs shaped before
  recording, metadata added from inside, a traced block, an error, and a photo attached to a run."""
  with captured() as (capture, _):
    handle_ticket("Can I get a refund?", {"id": "NC-90112", "opened": False, "address": "12 MG Road", "total": 650})
    with suppress(ValueError):
      fails()
    with_photo(ls.schemas.Attachment(mime_type="image/png", data=b"\x89PNG" + b"0" * 2048))
  rows = [
    [
      "  " * d + str(r.get("name")),
      str(r.get("run_type")),
      json.dumps(r.get("inputs"), default=str)[:70],
      json.dumps(r.get("outputs"), default=str)[:60],
      (r.get("error") or "")[:40],
      json.dumps(r.get("attachments") or {}),
    ]
    for d, r in capture.tree()
  ]
  view = table(["run", "type", "inputs", "outputs", "error", "attachments"], rows)
  meta = next(
    (r.get("extra", {}).get("metadata", {}) for r in capture.runs.values() if r.get("name") == "handle_ticket"), {}
  )
  return [
    view,
    callout(
      "Every function became a run nested under its caller, with its type. refund_check recorded only the order "
      "id (not the address), handle_ticket only the reply's length, the failing run kept its error, and the photo "
      f"travelled with its run. Metadata added at run time: {json.dumps({k: meta.get(k) for k in ('priority',)})}.",
      "info",
      "@traceable",
    ),
    example("smith.l03.1", "@traceable nests", handle_ticket, "", [view]),
    example("smith.l03.2", "run_type", find_policy, "", [view]),
    example("smith.l03.3", "process_inputs / process_outputs", refund_check, "", [view]),
    example(
      "smith.l03.4", "get_current_run_tree()", handle_ticket, "", [code(json.dumps(meta, default=str)[:300], "json")]
    ),
    example("smith.l03.5", "with trace(...)", handle_ticket, "", [view]),
    example("smith.l03.6", "Errors are recorded", fails, "", [view]),
    example("smith.l03.7", "Attachments", with_photo, "", [view]),
  ]


# ── Lab 5: threads ─────────────────────────────────────────────────────────────
def conversation(thread_id: str):
  chain = a_chain()
  for question in ("What is cold brew?", "And how long does it steep?"):  # each turn is its own trace...
    chain.invoke({"question": question}, {"metadata": {"thread_id": thread_id}})  # ...grouped by thread_id


def langgraph_thread():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, MessagesState, StateGraph

  def reply(state: MessagesState) -> dict:
    return {"messages": [AIMessage("Here you go.")]}

  graph = (
    StateGraph(MessagesState).add_node("reply", reply).add_edge(START, "reply").compile(checkpointer=InMemorySaver())
  )
  graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "t-graph-1"}})  # thread_id → metadata


@lab("smith.threads", "Threads: a conversation, not just a call", calls="No model calls")
def threads():
  """Two turns tagged with the same thread_id, a LangGraph run whose checkpointer thread_id lands in the metadata,
  and the thread read back from LangSmith when Nimbus has a key."""
  with captured() as (capture, _):
    conversation("t-42")
    langgraph_thread()
  roots = [r for r in capture.runs.values() if not r.get("parent_run_id")]
  rows = [
    [
      str(r.get("name")),
      str((r.get("extra") or {}).get("metadata", {}).get("thread_id")),
      json.dumps(r.get("inputs"), default=str)[:80],
    ]
    for r in roots
  ]
  remote = server(
    lambda client: [
      r.name
      for r in client.list_runs(
        project_name=get_settings().langsmith_project, filter='eq(metadata_key, "thread_id")', limit=5
      )
    ]
  )
  view = table(["trace", "thread_id", "inputs"], rows)
  return [
    view,
    table(["reading a thread from LangSmith"], [[str(remote)]]),
    example("smith.l04.1", "One thread_id for every turn", conversation, "", [view]),
    example("smith.l04.2", "LangGraph copies it for you", langgraph_thread, "", [view]),
    example("smith.l04.3", "Reading a thread back", conversation, "", [code(str(remote), "text")]),
  ]


# ── Lab 6: controlling tracing ─────────────────────────────────────────────────
def switched_off(capture_client):
  with ls.tracing_context(enabled=False):  # e.g. a health check, or a block with private data
    answer_question("ping")


def another_project(client):
  with ls.tracing_context(project_name="nimbus-staging", client=client):  # this block's traces: another project
    answer_question("What is a cortado?")


def local_only():
  with ls.tracing_context(enabled="local"):  # the run tree is built in memory and NOT uploaded
    with ls.trace("checked_locally") as run:
      answer_question("What is a mocha?")
    return run


def sampled(capture_url: str, n: int = 40) -> int:
  client = Client(api_url=capture_url, api_key="local-only", tracing_sampling_rate=0.25)  # keep a quarter
  with ls.tracing_context(client=client, enabled=True):
    for _ in range(n):
      answer_question("hi")
  client.flush()
  return n


@lab("smith.control", "Controlling tracing: on, off, where, how much", calls="No model calls")
def control():
  """A block with tracing off, a block sent to another project, a run tree built locally and not uploaded, and 40
  traces through a client that samples a quarter of them."""
  with captured() as (capture, client):
    switched_off(client)
    client.flush()
    off = len(capture.runs)
    another_project(client)
    client.flush()
    projects = sorted({str(r.get("session_name")) for r in capture.runs.values()})
    before = len(capture.runs)
    from langsmith.run_trees import configure

    with TraceCapture() as default_endpoint:  # stands in for the default endpoint: did anything go there?
      fallback = Client(api_url=default_endpoint.url, api_key="local-only")
      configure(client=fallback)
      try:
        run = local_only()
        fallback.flush()
      finally:
        configure(client=None)
      leaked = len(default_endpoint.runs)
    local_children = len(run.child_runs)
    client.flush()
    after = len(capture.runs)
  with TraceCapture() as sampler:
    n = sampled(sampler.url)
    kept = sum(1 for r in sampler.runs.values() if r.get("name") == "answer_question")
  view = table(
    ["", "result"],
    [
      ["tracing_context(enabled=False): runs uploaded", str(off)],
      ["tracing_context(project_name=...): projects", ", ".join(projects)],
      ["enabled='local': runs in memory / uploaded", f"{local_children} / {after - before}"],
      ["… LangChain runs inside it sent by the default client anyway", str(leaked)],
      [f"sampling 0.25: traces kept of {n}", str(kept)],
      ["priority", "tracing_context(...) > configure(...) > environment variables"],
    ],
  )
  return [
    view,
    callout(
      "Off means off, a project name moves a block, and sampling kept "
      f"{kept} of {n}. enabled='local' kept the @traceable run in memory"
      + (
        f", but the {leaked} LangChain runs inside it were still sent by the default client: don't rely on it to keep "
        "data private. Nimbus pins the default endpoint to this machine unless it has its own key (guard_uploads)."
        if leaked
        else ", and nothing inside it was sent."
      ),
      "warning" if leaked else "info",
      "Controlling tracing",
    ),
    example("smith.l05.1", "Off for one block", switched_off, "", [view]),
    example("smith.l05.2", "Another project", another_project, "", [view]),
    example("smith.l05.3", "enabled='local'", local_only, "", [view]),
    example("smith.l05.4", "Sampling", sampled, "", [code(f"{kept} of {n} kept", "text")]),
    example(
      "smith.l05.5",
      "Priority",
      another_project,
      "tracing_context beats ls.configure beats LANGSMITH_* variables",
      [view],
    ),
  ]


# ── Lab 7: private data ────────────────────────────────────────────────────────
PRIVATE = "I'm priya.s@example.com, call me on +91 98765 43210. My card 4111 1111 1111 1111 was charged twice."


def with_client(capture_url: str, **options) -> str:
  client = Client(api_url=capture_url, api_key="local-only", **options)
  with ls.tracing_context(client=client, enabled=True):
    answer_question(PRIVATE)
  client.flush()


def secret_rules():
  from langsmith.anonymizer import create_secret_anonymizer

  return create_secret_anonymizer()  # API keys, tokens, passwords: a ready-made rule set


@traceable(name="lookup_customer", process_inputs=lambda inputs: {"customer_id": inputs["customer_id"]})
def lookup_customer(customer_id: str, password: str) -> str:
  return "found"  # the password argument is dropped before anything is recorded


@lab("smith.privacy", "Keeping private data out of traces", calls="No model calls")
def privacy():
  """The same message with an email, a phone number and a card number traced four ways: as is, with
  hide_inputs/hide_outputs, with Nimbus's anonymizer, and a function that drops an argument."""
  rows = []
  for label, options in (
    ("as is", {}),
    ("hide_inputs + hide_outputs", {"hide_inputs": True, "hide_outputs": True}),
    ("Nimbus's anonymizer", {"anonymizer": pii_anonymizer()}),
  ):
    with TraceCapture() as capture:
      with_client(capture.url, **options)
      text = capture.text()
    rows.append(
      [
        label,
        *(
          "leaked" if marker in text else "—"
          for marker in ("priya.s@example.com", "98765 43210", "4111 1111 1111 1111")
        ),
      ]
    )
  with TraceCapture() as capture:
    client = Client(api_url=capture.url, api_key="local-only")
    with ls.tracing_context(client=client, enabled=True):
      lookup_customer("c-1042", "hunter2-secret")
    client.flush()
    dropped = "hunter2" not in capture.text()
  secrets = secret_rules()
  samples = [
    "key sk-proj-AbCdEf1234567890AbCdEf1234567890abcd",
    "password=hunter2",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc",
    "AWS key AKIAIOSFODNN7EXAMPLE",
  ]
  view = table(["traced", "email", "phone", "card"], rows)
  return [
    view,
    table(
      ["", "result"],
      [
        ["process_inputs dropped the password", str(dropped)],
        *[[f"secret rules on: {t}", str(secrets(t))] for t in samples],
      ],
    ),
    callout(
      "Traced as is, the email, phone and card number all reached the upload. hide_inputs/hide_outputs keep the "
      "run but none of its data; the anonymizer keeps the text and replaces what matches its rules, inside LangChain "
      "messages too. Nimbus's tracing client uses the anonymizer (tracing_client).",
      "decision",
      "Anonymize before upload",
    ),
    example("smith.l06.1", "hide_inputs / hide_outputs", with_client, "", [view]),
    example("smith.l06.2", "An anonymizer", pii_anonymizer, "nimbus/observability/tracing.py PII_RULES", [view]),
    example(
      "smith.l06.3",
      "create_secret_anonymizer()",
      secret_rules,
      "",
      [code("\n".join(str(secrets(t)) for t in samples), "text")],
    ),
    example(
      "smith.l06.4",
      "process_inputs drops a field",
      lookup_customer,
      "",
      [code(f"password recorded: {not dropped}", "text")],
    ),
  ]


# ── Lab 8: distributed tracing ─────────────────────────────────────────────────
def menu_service(capture_url: str):
  from fastapi import FastAPI, Request

  app = FastAPI()
  client = Client(api_url=capture_url, api_key="local-only")

  @traceable(name="price_lookup")
  def price_lookup(item: str) -> int:
    time.sleep(0.2)  # the slow step, in the OTHER service
    return {"cold brew": 200}.get(item, 0)

  @app.get("/price")
  def price(item: str, request: Request) -> dict:
    with ls.tracing_context(parent=request.headers, client=client, enabled=True):  # continue the caller's trace
      result = {"price": price_lookup(item)}
    client.flush()
    return result

  return app


@traceable(name="support_reply")
def call_the_menu_service(base: str) -> str:
  import httpx

  headers = ls.get_current_run_tree().to_headers()  # langsmith-trace (+ baggage for tags and metadata)
  price = httpx.get(f"{base}/price", params={"item": "cold brew"}, headers=headers).json()["price"]
  return f"A cold brew is ₹{price}."


@lab("smith.distributed", "Distributed tracing: one trace across two services", calls="No model calls")
def distributed():
  """The support service calls a menu service over HTTP with tracing headers; the menu service continues the trace;
  the capture holds one tree with runs from both processes."""
  from langsmith.run_trees import configure

  from academy.labs.fastapi.m02_data_streaming import served

  with captured() as (capture, client):
    configure(client=client)  # the default client in this process: a trace continued from headers uses it too
    try:
      with served(menu_service(capture.url)) as base:
        call_the_menu_service(base)
      time.sleep(0.5)
    finally:
      configure(client=None)
  trace_ids = {str(r.get("trace_id")) for r in capture.runs.values()}
  rows = [["  " * d + str(r.get("name")), _ms(r)] for d, r in capture.tree()]
  view = table(["run", "duration"], rows)
  return [
    view,
    callout(
      f"{len(capture.runs)} runs, {len(trace_ids)} trace: support_reply in one service and price_lookup in the "
      "other, nested. The slow step is the menu service's, and it's visible in the caller's trace.",
      "info",
      "One tree, two processes",
    ),
    example("smith.l08.1", "Headers from the caller", call_the_menu_service, "", [view]),
    example(
      "smith.l08.2",
      "The service continues the trace",
      menu_service,
      "TracingMiddleware does the same for FastAPI",
      [view],
    ),
    example("smith.l08.3", "One tree", call_the_menu_service, "", [view]),
  ]


def _ms(run: dict) -> str:
  from datetime import datetime

  try:
    start, end = (datetime.fromisoformat(str(run[k]).replace("Z", "+00:00")) for k in ("start_time", "end_time"))
    return f"{(end - start).total_seconds() * 1000:.0f} ms"
  except (KeyError, ValueError):
    return "—"


# ── Lab 9: querying traces ─────────────────────────────────────────────────────
QUERIES = {
  "failed root runs, last day": "client.list_runs(project_name='nimbus', is_root=True, error=True)",
  "slow model calls": "client.list_runs(project_name='nimbus', run_type='llm', filter='gt(latency, \"5s\")')",
  "traces tagged v2 for one customer": (
    'filter=\'and(has(tags, "v2"), eq(metadata_key, "customer"), eq(metadata_value, "c-5005"))\''
  ),
  "traces whose root was rated badly": "trace_filter='and(eq(feedback_key, \"thumbs\"), eq(feedback_score, 0))'",
  "traces where ANY run used a tool": "tree_filter='eq(run_type, \"tool\")'",
  "only some fields": "select=['name', 'latency', 'total_tokens', 'error']",
  "the newer async API": "await client.runs.query(project_ids=[...], filters={'RUN_TYPE': 'llm'})",
}


def failures_and_slow_calls():
  def query(client):
    project = get_settings().langsmith_project
    failed = [r.name for r in client.list_runs(project_name=project, is_root=True, error=True, limit=5)]
    slow = [
      r.name
      for r in client.list_runs(project_name=project, filter='gt(latency, "5s")', select=["name", "latency"], limit=5)
    ]
    return {"failed": failed, "slow": slow}

  return server(query)


@lab("smith.query", "Querying traces with the SDK", calls="No model calls")
def query():
  """The queries that find failures, slow calls and expensive traces, and the filter language; they run against
  LangSmith when Nimbus has a key."""
  result = failures_and_slow_calls()
  view = table(["question", "query"], [[k, v] for k, v in QUERIES.items()])
  return [
    view,
    table(["run against LangSmith"], [[json.dumps(result, default=str) if isinstance(result, dict) else result]]),
    example("smith.l09.1", "list_runs", failures_and_slow_calls, "", [view]),
    example("smith.l09.2", "The filter language", failures_and_slow_calls, "", [view]),
    example("smith.l09.3", "select=", failures_and_slow_calls, "", [view]),
    example("smith.l09.4", "client.runs.query", failures_and_slow_calls, "", [view]),
  ]


# ── Lab 10: OpenTelemetry ──────────────────────────────────────────────────────
def otel_spans(capture_url: str) -> int:
  from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
  from opentelemetry.sdk.trace import TracerProvider
  from opentelemetry.sdk.trace.export import SimpleSpanProcessor

  # What langsmith.integrations.otel.configure(project_name=...) sets up: an exporter to LangSmith's OTel endpoint
  provider = TracerProvider()
  provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{capture_url}/otel/v1/traces", headers={"x-api-key": "local"}))
  )
  tracer = provider.get_tracer("nimbus")
  with tracer.start_as_current_span("support_reply") as span:  # ordinary OTel code
    span.set_attribute("langsmith.span.kind", "chain")
    span.set_attribute("langsmith.metadata.customer", "c-1042")
    with tracer.start_as_current_span("model_call") as child:
      child.set_attribute("langsmith.span.kind", "llm")
      child.set_attribute("gen_ai.request.model", "ministral-14b-latest")
      child.set_attribute("gen_ai.prompt", "Where is my order?")
      child.set_attribute("gen_ai.completion", "It has shipped.")
      child.set_attribute("gen_ai.usage.input_tokens", 12)
  provider.shutdown()
  return 2


def langchain_as_otel(capture_url: str) -> str:
  env = {
    **os.environ,
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_TRACING_MODE": "otel",  # LangChain itself emits OTel spans (set before the app starts; or "hybrid")
    "LANGSMITH_ENDPOINT": capture_url,
    "LANGSMITH_API_KEY": "local",
  }
  script = CHAIN_SCRIPT + (
    "from langchain_core.tracers.langchain import wait_for_all_tracers; wait_for_all_tracers()\n"
    "from opentelemetry import trace; getattr(trace.get_tracer_provider(), 'force_flush', lambda: None)()\n"
  )
  result = subprocess.run(
    [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60, check=False
  )
  return result.stdout.strip() or result.stderr.strip()[-200:]


@lab("smith.otel", "OpenTelemetry: standard spans to LangSmith", calls="No model calls")
def otel():
  """Two nested spans written with plain OpenTelemetry and the attributes LangSmith understands, exported to the
  local capture; then a LangChain chain emitting OTel spans itself."""
  with TraceCapture() as capture:
    otel_spans(capture.url)
    spans = list(capture.spans)
  with TraceCapture() as capture:
    out = langchain_as_otel(capture.url)
    time.sleep(1)
    from_langchain = list(capture.spans)
  view = table(
    ["span", "parent", "attributes"],
    [[s["name"], s["parent"] or "—", json.dumps(s["attributes"])[:140]] for s in spans],
  )
  lc = table(
    ["span from LangChain", "attributes"],
    [[s["name"], json.dumps(s["attributes"])[:120]] for s in from_langchain[:6]] or [["(none)", out]],
  )
  return [
    view,
    lc,
    example(
      "smith.l10.1",
      "configure(project_name=...)",
      otel_spans,
      "langsmith.integrations.otel.configure sets this exporter up",
      [view],
    ),
    example("smith.l10.2", "Ordinary OTel code", otel_spans, "", [view]),
    example("smith.l10.3", "Attributes LangSmith understands", otel_spans, "", [view]),
    example("smith.l10.4", "LangChain emitting OTel", langchain_as_otel, "", [lc]),
  ]


# ── Lab 11: feedback ───────────────────────────────────────────────────────────
def rate_an_answer(capture_url: str) -> list[dict]:
  client = Client(api_url=capture_url, api_key="local-only")
  run_id, config = run_config("notes-chat", user="priya")  # Nimbus returns this run_id with every answer
  a_chain().invoke({"question": "When do points expire?"}, config)
  client.create_feedback(run_id, "thumbs", score=0, comment="It didn't say when.")  # 👎 with a comment
  client.create_feedback(run_id, "issue", value="wrong_fact")  # a category
  client.create_feedback(run_id, "correction", score=0, correction={"answer": "Points expire after 12 months."})
  return run_id


def feedback_config():
  return server(
    lambda client: client.create_feedback_config(
      "helpfulness", feedback_config={"type": "continuous", "min": 0, "max": 1}
    )
  )


def presigned_url():
  return server(lambda client: client.create_presigned_feedback_token(str(uuid.uuid4()), "thumbs").url)


def nimbus_feedback() -> list:
  from fastapi.testclient import TestClient

  from academy.labs.fastapi.m02_data_streaming import fake_model, lab_app, sign_in

  with TestClient(lab_app(fake_model("Points expire after 12 months."))) as client:
    sign_in(client)
    client.post("/notes", json={"title": "Points", "content": "Points expire after 12 months."})
    answer = client.post("/chat", json={"question": "When do points expire?"}).json()
    client.post("/feedback", json={"run_id": answer["run_id"], "score": 1, "comment": "clear"})
    return client.get("/feedback", params={"run_id": answer["run_id"]}).json()


@lab("smith.feedback", "Feedback: users' ratings on the traces they're about", calls="No model calls")
def feedback():
  """An answer's run id chosen up front, three kinds of feedback on it (a score, a category, a correction), the
  Nimbus API storing a user's rating and forwarding it when tracing is on, and the server-side parts (configs,
  presigned URLs, reading back) when Nimbus has a key."""
  with TraceCapture() as capture:
    with ls.tracing_context(client=Client(api_url=capture.url, api_key="local-only"), enabled=True):
      run_id = rate_an_answer(capture.url)
    sent = list(capture.feedback)
  stored = nimbus_feedback()
  view = table(
    ["key", "score", "value", "comment / correction"],
    [
      [f.get("key"), str(f.get("score")), str(f.get("value")), json.dumps(f.get("comment") or f.get("correction"))]
      for f in sent
    ],
  )
  return [
    view,
    table(
      ["stored by the Nimbus API", "value"],
      [[str(f["run_id"])[:12] + "…", json.dumps({k: f[k] for k in ("key", "score", "comment")})] for f in stored],
    ),
    table(
      ["server side", "result"],
      [["feedback config", str(feedback_config())], ["presigned feedback URL", str(presigned_url())]],
    ),
    example(
      "smith.l11.1",
      "The run id of every answer",
      rate_an_answer,
      "nimbus/observability/tracing.py run_config",
      [code(str(run_id), "text")],
    ),
    example("smith.l11.2", "create_feedback: score, value, correction", rate_an_answer, "", [view]),
    example("smith.l11.3", "Feedback configs", feedback_config, "", [code(str(feedback_config()), "text")]),
    example("smith.l11.4", "Presigned URLs for the frontend", presigned_url, "", [code(str(presigned_url()), "text")]),
    example(
      "smith.l11.5",
      "Reading it back",
      nimbus_feedback,
      "Nimbus keeps every rating: GET /feedback",
      [table(["run", "rating"], [[str(f["run_id"])[:12], str(f["score"])] for f in stored])],
    ),
  ]


# ── Lab 12: annotation queues ──────────────────────────────────────────────────
def review_queue():
  def build(client):
    queue = client.create_annotation_queue(
      name="nimbus-review", description="Is the answer right and polite? Score 0 or 1 on each."
    )  # the rubric: instructions + the feedback keys reviewers fill in
    runs = list(client.list_runs(project_name=get_settings().langsmith_project, is_root=True, limit=3))
    client.add_runs_to_annotation_queue(queue.id, run_ids=[r.id for r in runs])
    return f"queue {queue.id} with {len(runs)} runs"

  return server(build)


@lab("smith.annotation", "Annotation queues: traces to human reviewers", calls="No model calls")
def annotation():
  """A review queue with a rubric, three fresh answers added to it, and the reviewers' scores read back as
  feedback; it runs against LangSmith when Nimbus has a key."""
  result = review_queue()
  steps = table(
    ["step", "SDK"],
    [
      ["a queue with a rubric", "client.create_annotation_queue(name=..., description=<instructions>)"],
      ["what to review", "list_runs filters: rated 👎, slow, or a random sample"],
      ["add runs", "client.add_runs_to_annotation_queue(queue.id, run_ids=[...])"],
      ["read the scores", "client.list_feedback(run_ids=[...]): reviewers' scores are ordinary feedback"],
    ],
  )
  return [
    steps,
    table(["run against LangSmith"], [[str(result)]]),
    example("smith.l12.1", "A queue with a rubric", review_queue, "", [steps]),
    example("smith.l12.2", "Choosing what to review", review_queue, "", [steps]),
    example("smith.l12.3", "Adding runs", review_queue, "", [code(str(result), "text")]),
    example("smith.l12.4", "Scores are feedback", review_queue, "", [steps]),
  ]


# ── Production path: tracing in production, measured ───────────────────────────
CONVERSATIONS = [
  f"I'm customer{n}@example.com, phone +91 98{n:03d}0 4{n:04d}. Where is order NC-{1000 + n}? "
  f"Card 4111 1111 1111 {1000 + n}."
  for n in range(10)
]


def pii_in_uploads(client_options: dict) -> tuple[int, int]:
  with TraceCapture() as capture:
    client = Client(api_url=capture.url, api_key="local-only", **client_options)
    with ls.tracing_context(client=client, enabled=True):
      for message in CONVERSATIONS:
        answer_question(message)
    client.flush()
    text = capture.text()
  emails = sum(f"customer{n}@example.com" in text for n in range(10))
  cards = sum(f"4111 1111 1111 {1000 + n}" in text for n in range(10))
  return emails, cards


def overhead(traced: bool, n: int = 40) -> float:
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate

  chain = ChatPromptTemplate.from_messages([("human", "{question}")]) | fake(*["ok"] * n) | StrOutputParser()
  timings = []
  with TraceCapture() as capture:
    client = Client(api_url=capture.url, api_key="local-only")
    with ls.tracing_context(client=client, enabled=traced):
      for _ in range(n):
        started = time.perf_counter()
        chain.invoke({"question": "hi"})
        timings.append((time.perf_counter() - started) * 1000)
    client.flush()
  return sorted(timings)[n // 2]


def evaluate_tracing() -> list[dict]:
  rows = []
  for name, options in (
    ("default client", {}),
    ("Nimbus tracing client", {"anonymizer": pii_anonymizer(), "tracing_sampling_rate": 0.25}),
  ):
    emails, cards = pii_in_uploads(options)
    with TraceCapture() as capture:
      sampled_client = Client(
        api_url=capture.url, api_key="local-only", **{k: v for k, v in options.items() if k != "anonymizer"}
      )
      with ls.tracing_context(client=sampled_client, enabled=True):
        for _ in range(40):
          answer_question("hi")
      sampled_client.flush()
      kept = sum(1 for r in capture.runs.values() if r.get("name") == "answer_question")
    rows.append({"variant": name, "emails_uploaded": emails, "cards_uploaded": cards, "traces_kept": kept})
  off, on = overhead(False), overhead(True)
  return rows, off, on


@lab("prod.tracing-eval", "Evaluate: tracing in production", calls="No model calls")
def tracing_eval():
  """Ten conversations with emails, phone and card numbers traced with the default client and with Nimbus's tracing
  client (anonymizer, sampling a quarter): what reached the upload, how many of 40 traces were kept, and what
  tracing adds to a call."""
  rows, off, on = evaluate_tracing()
  default, nimbus = rows
  return [
    scorecard(
      rows,
      [
        Metric("emails_uploaded", "emails uploaded (of 10)", "lower", "int"),
        Metric("cards_uploaded", "card numbers uploaded (of 10)", "lower", "int"),
        Metric("traces_kept", "traces kept (of 40)", "none", "int"),
      ],
      title="Captured locally: nothing was sent",
    ),
    table(
      ["tracing", "chain call p50"], [["off", f"{off:.2f} ms"], ["on (uploads in a background thread)", f"{on:.2f} ms"]]
    ),
    callout(
      f"With the default client {default['emails_uploaded']} emails and {default['cards_uploaded']} card numbers "
      f"reached the upload; with Nimbus's client {nimbus['emails_uploaded']} and {nimbus['cards_uploaded']}. Sampling "
      f"kept {nimbus['traces_kept']} of 40 traces. Tracing added {on - off:.2f} ms to a call (p50): uploads happen "
      "in the background.",
      "decision",
      "Trace with Nimbus's client",
    ),
  ]
