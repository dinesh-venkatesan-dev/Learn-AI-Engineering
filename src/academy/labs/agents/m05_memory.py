"""Agents with LangGraph · Module 5 labs: checkpointers and threads, durable savers, time travel, chat memory inside a
graph, long-term memory with a store, durability and fault tolerance, and managing checkpoints in production.
Then: the support desk with conversation memory and customer memories, against no memory and threads only."""

import asyncio
import operator
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import closing, suppress
from datetime import timedelta
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from academy.labs.runtime import Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model

WORK = PROJECT_DIR / ".cache" / "labs" / "agents" / "m05"


def _store():
  from nimbus.support.tools import SupportStore

  return SupportStore()


def _say(desk, store, customer_id, text, thread=None):
  from nimbus.support.agent import CustomerContext

  context = CustomerContext(customer_id, store.customers.get(customer_id, "the customer"))
  config = {"configurable": {"thread_id": thread}} if thread else {}
  payload = {"messages": [HumanMessage(text)], "customer_id": customer_id, "actions": []}
  return desk.invoke(payload, config, context=context)


# ── Lab 1: checkpointers and threads ───────────────────────────────────────────
FOLLOW_UP = ("c-1042", ["What's the status of NC-20417?", "How many loyalty points did it earn?"])  # two orders


def without_a_checkpointer(model, store):
  from nimbus.support.desk import build_support_desk

  desk = build_support_desk(model, store)  # no checkpointer: every invoke starts from nothing
  return [_say(desk, store, FOLLOW_UP[0], text)["ticket"].reply for text in FOLLOW_UP[1]]


def with_a_checkpointer(model, store):
  from nimbus.support.desk import build_support_desk
  from nimbus.support.memory import memory_checkpointer

  desk = build_support_desk(model, store, checkpointer=memory_checkpointer())  # InMemorySaver: state after every step
  replies = [_say(desk, store, FOLLOW_UP[0], text, thread="c-1042:a")["ticket"].reply for text in FOLLOW_UP[1]]
  fresh = _say(desk, store, FOLLOW_UP[0], FOLLOW_UP[1][1], thread="c-1042:b")["ticket"].reply  # a new thread
  return desk, replies, fresh


def latest_checkpoint(desk, thread):
  snapshot = desk.get_state({"configurable": {"thread_id": thread}})  # a StateSnapshot
  return {
    "messages": len(snapshot.values["messages"]),
    "next": snapshot.next,  # () when the run finished
    "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"][-12:],
    "step": snapshot.metadata["step"],
    "fields": sorted(snapshot.values),
  }


def every_checkpoint(desk, thread):
  history = desk.get_state_history({"configurable": {"thread_id": thread}})  # newest first
  return [(s.metadata["step"], s.metadata["source"], s.next, len(s.values.get("messages", []))) for s in history]


@lab(
  "agents.checkpointers",
  "Checkpointers and threads: graphs that remember",
  uses_model=True,
  calls="~12 calls to Mistral",
)
def checkpointers():
  """A follow-up question ("how many points will it earn?") to the support desk, without and with a checkpointer,
  in the same thread and in a new one; then what the checkpointer saved."""
  model, store = get_chat_model(), _store()
  stateless = without_a_checkpointer(model, store)
  desk, replies, fresh = with_a_checkpointer(model, store)
  latest = latest_checkpoint(desk, "c-1042:a")
  history = every_checkpoint(desk, "c-1042:a")
  # NC-20417: ₹1300 → 130 points; listing every order's points (72 and 130) isn't answering about "it"
  points = lambda reply: "130" in reply.replace(",", "") and "72" not in reply  # noqa: E731
  return [
    table(
      ["", "turn 1: " + FOLLOW_UP[1][0], "turn 2: " + FOLLOW_UP[1][1], "points right"],
      [
        ["no checkpointer", stateless[0], stateless[1], "✓" if points(stateless[1]) else "✗"],
        ["checkpointer, same thread", replies[0], replies[1], "✓" if points(replies[1]) else "✗"],
        ["checkpointer, new thread", "", fresh, "✓" if points(fresh) else "✗"],
      ],
    ),
    table(
      ["What", "Value"],
      [[k, str(v)] for k, v in latest.items()],
      title="get_state(): the latest checkpoint of thread c-1042:a",
    ),
    table(
      ["step", "source", "next", "messages"],
      [[s, src, str(n), m] for s, src, n, m in history],
      title=f"get_state_history(): {len(history)} checkpoints, newest first",
    ),
    callout(
      "Without a checkpointer every call starts from nothing, so 'it' in the follow-up means nothing. With one, the "
      "state is saved after every step under the thread_id and the next call continues from it; a new thread_id is a "
      f"new conversation. Two turns left {len(history)} checkpoints: every step is saved, not just every turn. "
      + (
        "Here the second turn needed the first one, and only the same thread got it right."
        if points(replies[1]) and not points(stateless[1])
        else "Read the replies: this run didn't show the expected difference."
      ),
      "decision",
      "Checkpointers",
    ),
  ] + [
    example(
      "lc.l25.1",
      "Without a checkpointer, every call starts fresh",
      without_a_checkpointer,
      "",
      [table(["turn", "reply"], [[1, stateless[0]], [2, stateless[1]]])],
    ),
    example(
      "lc.l25.2",
      "Checkpointer + thread_id",
      with_a_checkpointer,
      "",
      [table(["turn", "reply"], [[1, replies[0]], [2, replies[1]]])],
    ),
    example(
      "lc.l25.3",
      "Different thread_ids are separate",
      with_a_checkpointer,
      "",
      [answer(fresh, "the same question in a new thread")],
    ),
    example(
      "lc.l25.4",
      "Inspecting what's saved",
      latest_checkpoint,
      "",
      [table(["What", "Value"], [[k, str(v)] for k, v in latest.items()])],
    ),
    example(
      "lg.l27.1",
      "compile(checkpointer=...)",
      with_a_checkpointer,
      "",
      [table(["checkpoints after two turns"], [[len(history)]])],
    ),
    example(
      "lg.l27.2",
      "Same thread continues; a new one starts fresh",
      with_a_checkpointer,
      "",
      [table(["thread", "reply to the follow-up"], [["c-1042:a", replies[1]], ["c-1042:b", fresh]])],
    ),
    example("lg.l27.3", "get_state()", latest_checkpoint, "", [code(str(latest), "python")]),
    example(
      "lg.l27.4",
      "get_state_history()",
      every_checkpoint,
      "",
      [table(["step", "source", "next", "messages"], [[s, src, str(n), m] for s, src, n, m in history[:8]])],
    ),
    example(
      "lg.l27.5",
      "A chat agent with memory = agent + checkpointer",
      with_a_checkpointer,
      "The desk is Module 4's graph, compiled with a checkpointer.",
      [answer(replies[1], "the follow-up, answered")],
    ),
  ]


# ── Lab 2: durable checkpointers ───────────────────────────────────────────────
class Counter(TypedDict):
  visits: Annotated[list[str], operator.add]


def counter_graph(checkpointer):
  from langgraph.graph import START, StateGraph

  def visit(state: Counter) -> dict:
    return {"visits": [f"visit {len(state.get('visits', [])) + 1}"]}

  return StateGraph(Counter).add_node("visit", visit).add_edge(START, "visit").compile(checkpointer=checkpointer)


CHILD = """
import sqlite3, sys
from contextlib import closing
from langgraph.checkpoint.sqlite import SqliteSaver
sys.path.insert(0, {src!r})
from academy.labs.agents.m05_memory import counter_graph
with closing(sqlite3.connect({db!r}, check_same_thread=False)) as conn:
  graph = counter_graph(SqliteSaver(conn))
  print(graph.invoke({{"visits": []}}, {{"configurable": {{"thread_id": "asha"}}}})["visits"])
"""


def in_memory_forgets():
  from langgraph.checkpoint.memory import InMemorySaver

  graph = counter_graph(InMemorySaver())
  graph.invoke({"visits": []}, {"configurable": {"thread_id": "asha"}})
  restarted = counter_graph(InMemorySaver())  # a new process has a new, empty InMemorySaver
  return restarted.get_state({"configurable": {"thread_id": "asha"}}).values


def sqlite_survives_a_restart(db):
  from langgraph.checkpoint.sqlite import SqliteSaver

  db.unlink(missing_ok=True)
  with closing(sqlite3.connect(db, check_same_thread=False)) as conn:  # sqlite3's `with` alone doesn't close
    counter_graph(SqliteSaver(conn)).invoke({"visits": []}, {"configurable": {"thread_id": "asha"}})
  child = CHILD.format(src=str(PROJECT_DIR / "src"), db=str(db))  # a brand-new Python process
  out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=120, check=True)
  return out.stdout.strip()


def inside_the_file(db):
  with closing(sqlite3.connect(db)) as conn:
    tables = [r[0] for r in conn.execute("select name from sqlite_master where type='table' order by name")]
    rows = {t: conn.execute(f"select count(*) from {t}").fetchone()[0] for t in tables}  # noqa: S608
  return rows


def postgres_saver():
  from langgraph.checkpoint.postgres import PostgresSaver

  from nimbus.support.memory import postgres_url

  with PostgresSaver.from_conn_string(postgres_url(get_settings().database_url)) as saver:
    saver.setup()  # creates the tables once
    graph = counter_graph(saver)
    thread = {"configurable": {"thread_id": f"lab-{uuid.uuid4().hex[:6]}"}}
    graph.invoke({"visits": []}, thread)
    visits = graph.invoke({"visits": []}, thread)["visits"]
    saver.delete_thread(thread["configurable"]["thread_id"])  # a lab cleans up after itself
    with saver.conn.cursor() as cur:
      cur.execute("select table_name from information_schema.tables where table_name like 'checkpoint%' order by 1")
      tables = [r["table_name"] for r in cur.fetchall()]
  return visits, tables


def async_saver(db):
  from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

  async def run():
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:  # opens and closes its own connection
      graph = counter_graph(saver)
      return (await graph.ainvoke({"visits": []}, {"configurable": {"thread_id": "asha"}}))["visits"]

  return asyncio.run(run())


@lab("agents.durable-checkpointers", "Durable checkpointers: memory that survives a restart", calls="No model calls")
def durable_checkpointers():
  """A visit counter: InMemorySaver after a 'restart', SqliteSaver read by a brand-new process, the tables inside,
  PostgresSaver on the course database, and the async saver."""
  WORK.mkdir(parents=True, exist_ok=True)
  db = WORK / "durable.sqlite"
  forgotten = in_memory_forgets()
  child = sqlite_survives_a_restart(db)
  rows = inside_the_file(db)
  pg_visits, pg_tables = postgres_saver()
  async_visits = async_saver(db)
  return [
    table(
      ["Saver", "What happened"],
      [
        ["InMemorySaver, after a restart", f"state: {forgotten or '{} (gone)'}"],
        ["SqliteSaver, read by a NEW process", f"visits: {child}"],
        ["PostgresSaver (course database)", f"visits: {pg_visits}"],
        ["AsyncSqliteSaver + ainvoke, same file", f"visits: {async_visits}"],
      ],
    ),
    table(["SQLite table", "rows"], list(rows.items()), title="Inside the file"),
    table(["Postgres tables"], [[t] for t in pg_tables]),
    callout(
      "InMemorySaver is for tests: a restart loses every conversation. SqliteSaver keeps them in a file a new "
      "process can read; PostgresSaver keeps them in a database every server can share, which is what production "
      "uses. Async apps use the async savers with ainvoke. The tables are plain: checkpoints (one per step) and "
      "writes (pending outputs of a step), so you can inspect them with SQL.",
      "decision",
      "Durable savers",
    ),
  ] + [
    example("lg.l28.1", "InMemorySaver forgets", in_memory_forgets, "", [code(str(forgotten), "python")]),
    example("lg.l28.2", "SqliteSaver: a new process continues", sqlite_survives_a_restart, "", [code(child, "text")]),
    example("lg.l28.3", "Inside the file", inside_the_file, "", [table(["table", "rows"], list(rows.items()))]),
    example("lg.l28.4", "PostgresSaver", postgres_saver, "", [table(["tables"], [[t] for t in pg_tables])]),
    example("lg.l28.5", "Async savers", async_saver, "", [code(str(async_visits), "python")]),
  ]


# ── Lab 3: time travel ─────────────────────────────────────────────────────────
PRICES = {"Cold Brew": 200, "Masala Chai": 100, "Oat Latte": 220}


class DrinkState(TypedDict):
  mood: str
  drink: str
  price: int


def drink_graph(model):
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import END, START, StateGraph

  def suggest(state: DrinkState) -> dict:
    prompt = f"A customer feels: {state['mood']}. Pick exactly one of {list(PRICES)}. Reply with the name only."
    drink = model.invoke(prompt).text.strip().strip("*.")
    return {"drink": drink if drink in PRICES else "Oat Latte"}

  def price(state: DrinkState) -> dict:
    return {"price": PRICES[state["drink"]]}

  graph = StateGraph(DrinkState).add_node("suggest", suggest).add_node("price", price)
  graph.add_edge(START, "suggest").add_edge("suggest", "price").add_edge("price", END)
  return graph.compile(checkpointer=InMemorySaver())


def checkpoint_ids(graph, config):
  return [
    (s.config["configurable"]["checkpoint_id"][-8:], s.next, dict(s.values)) for s in graph.get_state_history(config)
  ]


def replay(graph, before_suggest):
  return graph.invoke(None, before_suggest.config)  # re-runs everything after that checkpoint (the model again)


def fork(graph, before_suggest):
  forked = graph.update_state(before_suggest.config, {"mood": "a hot afternoon, needs something cold"})
  return graph.invoke(None, forked)  # a new branch with the changed state


def as_node(graph, after_suggest):
  manual = graph.update_state(after_suggest.config, {"drink": "Masala Chai"}, as_node="suggest")
  return graph.get_state(manual).next, graph.invoke(None, manual)  # as if suggest had said it: price runs next


@lab(
  "agents.time-travel", "Time travel: replay and fork past checkpoints", uses_model=True, calls="~3 calls to Mistral"
)
def time_travel():
  """A drink suggestion (a model at temperature 1) then a price: listed checkpoints, a replay, a fork with a
  different mood, and an update written as if a node had produced it."""
  graph = drink_graph(get_chat_model(temperature=1.0))
  config = {"configurable": {"thread_id": "order-42"}}
  first = graph.invoke({"mood": "tired, it's raining", "drink": "", "price": 0}, config)
  history = list(graph.get_state_history(config))
  before = next(s for s in history if s.next == ("suggest",))
  after = next(s for s in history if s.next == ("price",))
  ids = checkpoint_ids(graph, config)
  replayed, forked = replay(graph, before), fork(graph, before)
  nxt, manual = as_node(graph, after)
  total = len(list(graph.get_state_history(config)))
  return [
    table(
      ["Run", "mood", "drink", "price"],
      [
        ["original", first["mood"], first["drink"], first["price"]],
        ["replay from before suggest", replayed["mood"], replayed["drink"], replayed["price"]],
        ["fork: mood changed", forked["mood"], forked["drink"], forked["price"]],
        ["as_node='suggest': drink written by hand", manual["mood"], manual["drink"], manual["price"]],
      ],
    ),
    table(
      ["checkpoint", "next", "values"], [[i, str(n), str(v)] for i, n, v in ids], title="The original run's checkpoints"
    ),
    callout(
      f"Every checkpoint has an id you can return to. Replay re-runs everything after it (the model ran again, at "
      f"temperature 1); a fork changes the state there and continues on a new branch; as_node writes a value as if "
      f"that node had produced it, so the graph continues with what follows it ({nxt}). Nothing is overwritten: the "
      f"thread now holds {total} checkpoints, the original ones included. Use it to debug a bad run and to test 'what "
      "if' without starting over.",
      "decision",
      "Time travel",
    ),
  ] + [
    example(
      "lc.l51.1",
      "A checkpoint after every node",
      checkpoint_ids,
      "",
      [table(["checkpoint", "next"], [[i, str(n)] for i, n, _ in ids])],
    ),
    example(
      "lc.l51.2",
      "get_state_history()",
      checkpoint_ids,
      "",
      [table(["checkpoint", "values"], [[i, str(v)] for i, _, v in ids])],
    ),
    example("lc.l51.3", "Replay", replay, "", [code(str(replayed), "python")]),
    example("lc.l51.4", "Fork with update_state", fork, "", [code(str(forked), "python")]),
    example(
      "lg.l29.1",
      "Checkpoint ids",
      checkpoint_ids,
      "",
      [table(["checkpoint", "next"], [[i, str(n)] for i, n, _ in ids])],
    ),
    example("lg.l29.2", "Replay: invoke(None, past_config)", replay, "", [code(str(replayed), "python")]),
    example("lg.l29.3", "Fork: update_state, then continue", fork, "", [code(str(forked), "python")]),
    example("lg.l29.4", "as_node", as_node, "", [table(["next", "result"], [[str(nxt), str(manual)]])]),
    example(
      "lg.l29.5",
      "History is never overwritten",
      checkpoint_ids,
      "",
      [table(["checkpoints in the thread now"], [[total]])],
    ),
  ]


# ── Lab 4: chat memory inside a graph ──────────────────────────────────────────
EARLY_FACT = "By the way, my daughter's birthday is on 14 April and she loves the Mango Frappe."
SMALL_TALK = [
  "Is the Bandra café open on Sundays?",
  "Do you have oat milk?",
  "What's in a cortado?",
  "Is the cold brew sweetened?",
  "Can I bring my dog to the café?",
  "Do you sell gift cards?",
  "How long do beans stay fresh?",
  "Do you roast in-house?",
  "Can I order ahead?",
  "Is there parking near Indiranagar?",
]
FINAL = "Which drink should I order for my daughter's birthday, and when is it?"


def long_history() -> list:
  messages = [HumanMessage(EARLY_FACT), AIMessage("Lovely! Noted.")]
  for question in SMALL_TALK:
    messages += [
      HumanMessage(question),
      AIMessage(f"Here's a full answer about that: {question.lower()} " + "Details. " * 30),
    ]
  return messages


def trimmed(messages, max_tokens=400):
  from langchain_core.messages import trim_messages
  from langchain_core.messages.utils import count_tokens_approximately

  # what the model sees; the saved history is untouched
  return trim_messages(
    messages, max_tokens=max_tokens, strategy="last", token_counter=count_tokens_approximately, start_on="human"
  )


def deleted(messages, keep=4):
  return [RemoveMessage(id=m.id) for m in messages[:-keep]]  # returned from a node: removed from the saved state


def summarised(model, messages, keep=4):
  old, recent = messages[:-keep], messages[-keep:]
  transcript = "\n".join(f"{m.type}: {m.text}" for m in old)
  summary = model.invoke(
    f"Summarise this conversation in two sentences. Keep names, dates and preferences.\n\n{transcript}"
  ).text
  return [SystemMessage(f"Summary of the earlier conversation: {summary}"), *recent], summary


@lab(
  "agents.chat-memory",
  "Managing chat memory inside a graph: trim, delete, summarise",
  uses_model=True,
  calls="~5 calls to Mistral",
)
def chat_memory():
  """A 22-message conversation whose first message has the fact the last question needs. The model sees all of it,
  a trimmed view, the state after deleting old messages, or a summary plus the recent messages."""
  from langchain_core.messages.utils import count_tokens_approximately
  from langgraph.graph import START, MessagesState, StateGraph

  model = get_chat_model()
  history = [m.model_copy(update={"id": str(i)}) for i, m in enumerate(long_history())]
  question = HumanMessage(FINAL)

  def delete_old(state: MessagesState) -> dict:
    return {"messages": deleted(state["messages"])}

  saved = StateGraph(MessagesState).add_node("delete_old", delete_old).add_edge(START, "delete_old").compile()
  after_delete = saved.invoke({"messages": history})["messages"]
  with_summary, summary = summarised(model, history)
  views = {
    "everything": history,
    "trimmed (400 tokens)": trimmed(history),
    "old messages deleted": after_delete,
    "summary + recent": with_summary,
  }
  rows = []
  for name, view in views.items():
    reply = model.invoke([*view, question]).text
    knows = "mango" in reply.casefold() and "14" in reply
    rows.append(
      {
        "variant": name,
        "messages": len(view),
        "tokens": count_tokens_approximately([*view, question]),
        "knows": 1.0 if knows else 0.0,
        "reply": reply,
      }
    )
  tokens = ", ".join(f"{r['variant']} {r['tokens']}" for r in rows)
  return [
    scorecard(
      rows,
      [
        Metric("messages", "messages sent", "lower", "int"),
        Metric("tokens", "tokens sent", "lower", "int"),
        Metric("knows", "remembers the early fact"),
      ],
    ),
    table(["View", "reply"], [[r["variant"], r["reply"]] for r in rows]),
    answer(summary, "The running summary"),
    callout(
      "Trimming changes what the model sees and keeps the saved history; RemoveMessage shrinks the saved history "
      f"itself; a summary replaces old messages with a few sentences (tokens sent: {tokens}). Which kept the fact "
      "from the first message: "
      + ", ".join(f"{r['variant']}: {'yes' if r['knows'] else 'no'}" for r in rows)
      + ". Cutting old messages cuts what they said; a summary or a long-term memory (lab 5) keeps it. The "
      + "production desk trims (3,000 tokens) and saves lasting facts as memories instead of summarising.",
      "decision",
      "Chat memory",
    ),
  ] + [
    example(
      "lg.l30.1",
      "Trim when calling the model",
      trimmed,
      "",
      [table(["messages", "tokens"], [[len(views["trimmed (400 tokens)"]), rows[1]["tokens"]]])],
    ),
    example(
      "lg.l30.2",
      "Delete with RemoveMessage",
      deleted,
      "",
      [table(["saved messages before", "after"], [[len(history), len(after_delete)]])],
    ),
    example("lg.l30.3", "Summarise", summarised, "", [answer(summary, "summary")]),
  ]


# ── Lab 5: long-term memory with a store ───────────────────────────────────────
STATEMENTS = [
  "I'm lactose intolerant.",
  "I prefer my beans whole, not ground.",
  "Where is my order NC-78123?",
  "Thanks, have a nice day!",
  "I'm annoyed my order hasn't shipped yet.",
  "Please deliver to my office this time.",
]


def store_basics(embeddings):
  from langgraph.store.memory import InMemoryStore

  store = InMemoryStore(index={"embed": embeddings, "dims": 384, "fields": ["fact"]})
  ns = ("customers", "c-1042", "memories")  # a namespace per customer
  for key, fact in (("diet", "Is lactose intolerant."), ("beans", "Prefers whole beans."), ("city", "Lives in Pune.")):
    store.put(ns, key, {"fact": fact})
  got = store.get(ns, "diet").value
  found = [
    (q, store.search(ns, query=q, limit=1)[0].value["fact"]) for q in ("What milk can I have?", "Grind setting?")
  ]
  return got, found  # get by key; search by meaning


def loose_extractor(model, text):
  from pydantic import BaseModel

  class Facts(BaseModel):
    facts: list[str]

  return model.with_structured_output(Facts, method="json_schema").invoke(f"Extract facts about the user: {text}").facts


def desk_memories(model, store, texts):
  from langgraph.store.memory import InMemoryStore

  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.support.desk import build_support_desk
  from nimbus.support.memory import all_memories

  memory = InMemoryStore(index={"embed": get_embeddings(get_settings()), "dims": 384, "fields": ["fact"]})
  desk = build_support_desk(model, store, memory=memory)  # compile(store=...): nodes and tools get runtime.store
  saved = {}
  for text in texts:
    before = {m["fact"] for m in all_memories(memory, "c-1042")}
    _say(desk, store, "c-1042", text, thread=None)
    saved[text] = sorted({m["fact"] for m in all_memories(memory, "c-1042")} - before)  # what this message added
  return desk, memory, saved


@lab(
  "agents.memory-store",
  "Long-term memory with a store: facts shared across threads",
  uses_model=True,
  calls="~20 calls to Mistral",
)
def memory_store():
  """A store on its own (put, get, semantic search), then the desk's strict remember tool against a loose extractor
  on six messages, and a memory saved in one conversation recalled in another."""
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.support.desk import build_support_desk
  from nimbus.support.memory import memory_checkpointer

  model, store = get_chat_model(), _store()
  got, found = store_basics(get_embeddings(get_settings()))
  loose = {t: loose_extractor(model, t) for t in STATEMENTS}
  _, memory, strict = desk_memories(model, store, STATEMENTS)
  recall_desk = build_support_desk(model, store, checkpointer=memory_checkpointer(), memory=memory)
  recalled = _say(recall_desk, store, "c-1042", "Which café drinks would suit me?", thread="c-1042:new")
  lasting = {STATEMENTS[0], STATEMENTS[1]}
  noise = lambda saved: sum(len(v) for k, v in saved.items() if k not in lasting)  # noqa: E731
  return [
    table(
      ["Message", "loose extractor saves", "desk's remember tool saves"],
      [[t, "; ".join(loose[t]) or "—", "; ".join(strict[t]) or "—"] for t in STATEMENTS],
      title="What counts as a memory",
    ),
    table(
      ["", "loose extractor", "remember tool"],
      [
        ["memories saved", sum(map(len, loose.values())), sum(map(len, strict.values()))],
        ["from messages with no lasting fact", noise(loose), noise(strict)],
      ],
    ),
    table(
      ["New conversation", "memories recalled", "reply"],
      [["Which café drinks would suit me?", "; ".join(recalled["memories"]), recalled["ticket"].reply]],
    ),
    callout(
      f"A store keeps JSON documents by namespace and key, searchable by meaning once it has an embedding model; "
      f"nodes and tools reach it through runtime.store. Be strict about what counts: the loose extractor saved "
      f"{noise(loose)} 'memories' from messages with nothing lasting in them (order numbers, a complaint, a one-off "
      f"delivery address), the remember tool {noise(strict)}. A memory saved in one conversation was recalled in a new "
      "one and shaped the reply.",
      "decision",
      "Long-term memory",
    ),
  ] + [
    example(
      "lc.l30.1",
      "A store: put, get, search",
      store_basics,
      "",
      [table(["What", "Result"], [["get", str(got)], *[[f"search: {q}", f] for q, f in found]])],
    ),
    example(
      "lc.l30.2",
      "Tools read and write runtime.store",
      desk_memories,
      "The remember tool is in nimbus/support/memory.py.",
      [table(["message", "saved"], [[t, "; ".join(v) or "—"] for t, v in strict.items()])],
    ),
    example(
      "lc.l30.3",
      "Memory shared across threads",
      desk_memories,
      "",
      [answer(recalled["ticket"].reply, "a new conversation")],
    ),
    example("lg.l31.1", "Documents by (namespace, key)", store_basics, "", [code(str(got), "python")]),
    example(
      "lg.l31.2", "Semantic search", store_basics, "", [table(["query", "best match"], [list(f) for f in found])]
    ),
    example(
      "lg.l31.3",
      "runtime.store; compile(store=...)",
      desk_memories,
      "",
      [table(["memories recalled"], [[m] for m in recalled["memories"]])],
    ),
    example(
      "lg.l31.4", "Thread A learns, thread B recalls", desk_memories, "", [answer(recalled["ticket"].reply, "thread B")]
    ),
    example(
      "lg.l31.5",
      "Be strict about what counts",
      loose_extractor,
      "",
      [table(["", "loose", "strict"], [["noise memories", noise(loose), noise(strict)]])],
    ),
  ]


# ── Lab 6: durability and fault tolerance ──────────────────────────────────────
class OrderRun(TypedDict):
  done: Annotated[list[str], operator.add]


def failing_steps(runs: dict, fail: set):
  def node(name):
    def run(state: OrderRun) -> dict:
      runs[name] = runs.get(name, 0) + 1
      if name in fail:
        fail.discard(name)  # fails once (an outage), works on the next attempt
        raise ConnectionError(f"{name}: the payment gateway timed out")
      return {"done": [name]}

    run.__name__ = name
    return run

  return node


def fail_then_resume():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph

  runs, fail = {}, {"pay"}
  node = failing_steps(runs, fail)
  graph = (
    StateGraph(OrderRun)
    .add_sequence([node("reserve_beans"), node("pay"), node("ship")])
    .add_edge(START, "reserve_beans")
  )
  graph = graph.compile(checkpointer=InMemorySaver())
  config = {"configurable": {"thread_id": "order-1"}}
  try:
    graph.invoke({"done": []}, config)
  except ConnectionError as error:
    failed = str(error)
  saved = graph.get_state(config)  # the last COMPLETED step survives
  result = graph.invoke(None, config)  # resume: finished steps aren't re-run
  return failed, saved.values, saved.next, result, dict(runs)


def parallel_failure():
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.graph import START, StateGraph

  runs, fail = {}, {"charge_card"}
  node = failing_steps(runs, fail)
  builder = StateGraph(OrderRun)
  for name in ("charge_card", "send_receipt", "update_stock"):
    builder.add_node(node(name)).add_edge(START, name)
  graph = builder.compile(checkpointer=InMemorySaver())
  config = {"configurable": {"thread_id": "order-2"}}
  with suppress(ConnectionError):
    graph.invoke({"done": []}, config)
  graph.invoke(None, config)
  return dict(runs)  # the branches that succeeded were kept as pending writes


def durability_modes(steps=30):
  from langgraph.checkpoint.sqlite import SqliteSaver
  from langgraph.graph import START, StateGraph

  WORK.mkdir(parents=True, exist_ok=True)
  db = WORK / "durability.sqlite"
  db.unlink(missing_ok=True)
  out = {}
  with closing(sqlite3.connect(db, check_same_thread=False)) as conn:
    saver = SqliteSaver(conn)
    names = [f"s{i}" for i in range(steps)]
    builder = StateGraph(OrderRun)
    for name in names:
      builder.add_node(name, lambda state, n=name: {"done": [n]})
    builder.add_edge(START, names[0])
    for a, b in zip(names, names[1:], strict=False):
      builder.add_edge(a, b)
    graph = builder.compile(checkpointer=saver)
    for mode in ("sync", "async", "exit"):
      config = {"configurable": {"thread_id": mode}}
      started = time.perf_counter()
      graph.invoke({"done": []}, config, durability=mode)
      out[mode] = ((time.perf_counter() - started) * 1000, len(list(graph.get_state_history(config))))
  return out


@lab("agents.durability", "Durability and fault tolerance: surviving crashes mid-run", calls="No model calls")
def durability():
  """An order flow whose payment step fails once: what's saved, what's re-run on resume, what happens to parallel
  branches that succeeded, and the cost of each durability mode on a 30-step run with SQLite."""
  failed, saved, nxt, result, runs = fail_then_resume()
  parallel = parallel_failure()
  modes = durability_modes()
  return [
    table(
      ["What", "Result"],
      [
        ["failed with", failed],
        ["saved state", str(saved)],
        ["next", str(nxt)],
        ["after resume", str(result)],
        ["executions per step", str(runs)],
      ],
      title="A step fails, then the run resumes",
    ),
    table(["Parallel branch", "executions"], list(parallel.items()), title="One of three parallel branches fails"),
    table(
      ["durability", "ms (30 steps, SQLite)", "checkpoints"], [[m, f"{ms:.0f}", n] for m, (ms, n) in modes.items()]
    ),
    callout(
      f"When a step fails, the checkpoint of the last completed step survives; invoke(None, config) resumes from it "
      f"and doesn't repeat finished work ({runs}). In a parallel step the branches that succeeded are kept as pending "
      f"writes, so only the failed one runs again ({parallel}). Durability decides when checkpoints are written: sync "
      f"and async save every step (async in the background), exit only the end: fastest "
      f"({modes['exit'][0]:.0f} ms vs {modes['sync'][0]:.0f} ms here) but nothing to resume from mid-run.",
      "decision",
      "Durability",
    ),
  ] + [
    example(
      "lg.l32.1",
      "The last completed step survives",
      fail_then_resume,
      "",
      [table(["saved", "next"], [[str(saved), str(nxt)]])],
    ),
    example(
      "lg.l32.2",
      "Resume without repeating work",
      fail_then_resume,
      "",
      [table(["step", "executions"], list(runs.items()))],
    ),
    example(
      "lg.l32.3", "Pending writes", parallel_failure, "", [table(["branch", "executions"], list(parallel.items()))]
    ),
    example(
      "lg.l32.4",
      "Durability modes",
      durability_modes,
      "",
      [table(["mode", "ms", "checkpoints"], [[m, f"{ms:.0f}", n] for m, (ms, n) in modes.items()])],
    ),
  ]


# ── Lab 7: managing checkpoints in production ─────────────────────────────────
def inspect_checkpoints(saver, thread):
  return [
    (item.metadata["step"], item.metadata["source"], item.config["configurable"]["checkpoint_id"][-8:])
    for item in saver.list({"configurable": {"thread_id": thread}}, limit=4)  # newest first
  ], len(list(saver.list({"configurable": {"thread_id": thread}}, filter={"source": "input"})))


def forget_me(saver, memory, customer_id):
  from nimbus.support.memory import forget_customer

  return forget_customer(saver, memory, customer_id)  # every conversation and every memory of the customer


def retention(saver, max_age):
  from nimbus.support.memory import expire_conversations

  return expire_conversations(saver, max_age)  # nightly: conversations inactive for longer than max_age


def encryption_at_rest(key: bytes):
  from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
  from langgraph.checkpoint.sqlite import SqliteSaver

  out = {}
  for label, serde in (("plain", None), ("encrypted", EncryptedSerializer.from_pycryptodome_aes(key=key))):
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
      saver = SqliteSaver(conn, serde=serde) if serde else SqliteSaver(conn)
      graph = counter_graph(saver)
      graph.invoke({"visits": ["card 4111-1111"]}, {"configurable": {"thread_id": "x"}})
      raw = b"".join(row[0] for row in conn.execute("select checkpoint from checkpoints"))
      out[label] = (b"4111-1111" in raw, graph.get_state({"configurable": {"thread_id": "x"}}).values["visits"][0])
  return out


def schema_change(saver, thread):
  from langgraph.graph import START, StateGraph

  class CounterV2(TypedDict, total=False):
    visits: Annotated[list[str], operator.add]
    channel: str  # a new field: old threads don't have it

  v2 = StateGraph(CounterV2).add_node("visit", lambda s: {}).add_edge(START, "visit").compile(checkpointer=saver)
  values = v2.get_state({"configurable": {"thread_id": thread}}).values
  return values, values.get("channel", "web")  # a default in code, not a crash


@lab(
  "agents.checkpoint-admin",
  "Managing checkpoints in production: inspect, forget, expire, encrypt, migrate",
  uses_model=True,
  calls="~6 calls to Mistral",
)
def checkpoint_admin():
  """The desk's conversations in the course Postgres (with the production helpers): inspect, list threads, a
  'forget me' request, retention, encryption at rest, and a state field added later."""
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.support.desk import build_support_desk
  from nimbus.support.memory import all_memories, conversations, desk_persistence, thread_id

  model, store, settings = get_chat_model(), _store(), get_settings()
  with desk_persistence(settings, get_embeddings(settings)) as (saver, memory):
    for cid in ("c-8080", "c-2210"):
      forget_me(saver, memory, cid)  # start clean
    desk = build_support_desk(model, store, checkpointer=saver, memory=memory)
    _say(desk, store, "c-8080", "I'm vegan, please remember that.", thread=thread_id("c-8080", "lab-a"))
    _say(desk, store, "c-8080", "Can I still return order NC-14002? It's sealed.", thread=thread_id("c-8080", "lab-b"))
    _say(desk, store, "c-2210", "Thanks, have a nice day!", thread=thread_id("c-2210", "lab-a"))
    latest, inputs = inspect_checkpoints(saver, thread_id("c-8080", "lab-a"))
    threads = {t: w for t, w in conversations(saver).items() if t.startswith(("c-8080:lab", "c-2210:lab"))}
    memories_before = all_memories(memory, "c-8080")
    forgotten = forget_me(saver, memory, "c-8080")
    after = {t for t in conversations(saver) if t.startswith("c-8080:")}
    time.sleep(1.5)
    expired = [t for t in retention(saver, timedelta(seconds=1)) if t.startswith("c-2210:lab")]  # days in production
    values, channel = schema_change(saver, thread_id("c-2210", "lab-a"))
  encrypted = encryption_at_rest(os.urandom(32))
  return [
    table(
      ["step", "source", "checkpoint"],
      [list(x) for x in latest],
      title=f"checkpointer.list(): newest first ({inputs} created by new input)",
    ),
    table(["Thread", "last active"], [[t, w.strftime("%H:%M:%S")] for t, w in threads.items()]),
    table(
      ["Request", "Result"],
      [
        ["memories of c-8080 before", "; ".join(m["fact"] for m in memories_before) or "—"],
        ["forget c-8080", str(forgotten)],
        ["c-8080 threads left", str(sorted(after)) or "[]"],
        ["retention (1 s here, days in production) deleted", str(expired)],
      ],
    ),
    table(
      ["Serializer", "card number readable in the database", "the app still reads it"],
      [[k, v[0], v[1]] for k, v in encrypted.items()],
      title="Encryption at rest",
    ),
    callout(
      "Checkpoints pile up forever unless you manage them. List them to inspect a thread; keep thread ids that "
      "start with the customer id, so a 'forget me' request finds every conversation (and delete the customer's "
      "memories too); run a retention job; encrypt checkpoints at rest when they hold personal data "
      "(NIMBUS_CHECKPOINT_KEY in production). When the state schema changes, old threads keep their old values: "
      "add fields with defaults in code and migrate renamed ones.",
      "decision",
      "Checkpoints in production",
    ),
  ] + [
    example(
      "lg.l33.1",
      "checkpointer.list()",
      inspect_checkpoints,
      "",
      [table(["step", "source", "checkpoint"], [list(x) for x in latest])],
    ),
    example(
      "lg.l33.2",
      "Every thread and when it was last active",
      conversations,
      "",
      [table(["thread", "last active"], [[t, w.isoformat()] for t, w in threads.items()])],
    ),
    example("lg.l33.3", "delete_thread(): forget me", forget_me, "", [code(str(forgotten), "python")]),
    example("lg.l33.4", "Retention", retention, "", [code(str(expired), "python")]),
    example(
      "lg.l33.5",
      "Encryption at rest",
      encryption_at_rest,
      "",
      [table(["serializer", "readable", "app reads"], [[k, v[0], v[1]] for k, v in encrypted.items()])],
    ),
    example(
      "lg.l33.6",
      "Changing the state schema",
      schema_change,
      "",
      [table(["old thread under the new schema", "new field"], [[str(values)[:200], channel]])],
    ),
  ]


# ── Production path: the desk with memory ──────────────────────────────────────
def evaluate_memory() -> list[dict]:
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.support.memory_eval import load_conversations, run_conversation, score

  model, embeddings = get_chat_model(), get_embeddings(get_settings())
  rows = []
  for conversation in load_conversations(PROJECT_DIR / "data" / "eval" / "memory_conversations.json"):
    for variant in ("none", "threads", "threads+memory"):
      run = run_conversation(model, conversation, variant, embeddings)
      rows.append({"variant": variant, "conversation": conversation, "run": run, "ok": score(conversation, run)})
  return rows


@lab(
  "agents.desk-memory-eval",
  "Evaluate: the support desk with memory",
  uses_model=True,
  calls="~90 calls to Mistral: 12 conversations × 3 variants, about 5 minutes",
)
def desk_memory_eval():
  """12 short conversations: follow-ups in the same conversation, facts recalled in a new one, and messages with
  nothing worth remembering. The desk with no memory, with a checkpointer, and with a checkpointer and a store."""
  from nimbus.support.memory_eval import unrequested_actions

  started = time.perf_counter()
  rows = evaluate_memory()
  seconds = time.perf_counter() - started
  out = []
  for variant in ("none", "threads", "threads+memory"):
    mine = [r for r in rows if r["variant"] == variant]
    by_kind = lambda kind, m=mine: [r for r in m if r["conversation"]["kind"] == kind]  # noqa: E731
    out.append(
      {
        "variant": {"none": "no memory", "threads": "checkpointer", "threads+memory": "checkpointer + store"}[variant],
        "follow_up": f"{sum(r['ok'] for r in by_kind('follow-up'))}/{len(by_kind('follow-up'))}",
        "recall": f"{sum(r['ok'] for r in by_kind('recall'))}/{len(by_kind('recall'))}",
        "noise": sum(len(r["run"]["memories"]) for r in by_kind("noise")),
        "unrequested": sum(len(unrequested_actions(r["conversation"], r["run"])) for r in mine),
        "calls": round(sum(r["run"]["calls"] for r in mine) / len(mine), 2),
        "tokens": round(sum(r["run"]["tokens"] for r in mine) / len(mine)),
      }
    )
  failures = [
    [
      r["variant"],
      r["conversation"]["id"],
      " → ".join(t for _, t in r["conversation"]["turns"]),
      r["run"]["reply"],
      "; ".join(r["run"]["memories"]),
      "; ".join(unrequested_actions(r["conversation"], r["run"])),
    ]
    for r in rows
    if not r["ok"]
  ]
  best = out[2]
  return [
    scorecard(
      out,
      [
        Metric("follow_up", "follow-ups right", "none", "text"),
        Metric("recall", "recalled in a new conversation", "none", "text"),
        Metric("noise", "memories saved from noise", "lower", "int"),
        Metric("unrequested", "actions nobody asked for", "lower", "int"),
        Metric("calls", "model calls / conversation", "lower", "text"),
        Metric("tokens", "tokens / conversation", "lower", "int"),
      ],
      title=f"12 conversations · {seconds:.0f} s",
    ),
    table(
      ["Variant", "Conversation", "Messages", "last reply", "memories", "actions nobody asked for"],
      failures,
      title="Every miss",
    ),
    callout(
      f"Follow-ups need the conversation (the checkpointer), recall across conversations needs the store: with both, "
      f"the desk got {best['follow_up']} follow-ups and {best['recall']} recalls, and saved {best['noise']} memories "
      "from the messages with nothing lasting in them. Actions nobody asked for: "
      + ", ".join(f"{o['variant']} {o['unrequested']}" for o in out)
      + ". Without a remember tool, a customer stating a preference got the nearest tool instead (a subscription "
      "pause). Production runs the desk on PostgresSaver and PostgresStore (nimbus/support/memory.py), with thread "
      "ids per customer for 'forget me' and a retention job.",
      "decision",
      "Ship the desk with memory",
    ),
  ]
