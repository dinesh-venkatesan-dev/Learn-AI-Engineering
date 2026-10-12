"""RAG · Module 7 labs: the RAG chain, prompting for grounded answers, citations, conversational follow-ups, and
when to query a database instead of searching text. Then: the production answerer against a naive one."""

import json
import re
import sqlite3
import time
from contextlib import closing
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from academy.labs import cache
from academy.labs.rag import corpus
from academy.labs.runtime import Input, Metric, answer, callout, code, example, lab, scorecard, table
from nimbus.config import PROJECT_DIR, get_settings
from nimbus.llm import get_chat_model
from nimbus.llm.usage import UsageRecorder
from nimbus.rag.answer import Answerer, format_sources, verify
from nimbus.rag.ingestion.embeddings import get_embeddings
from nimbus.rag.retriever import Retriever

DATA = get_settings().data_dir
UNANSWERABLE = [q["question"] for q in json.loads((PROJECT_DIR / "data" / "eval" / "unanswerable.json").read_text())]


@lru_cache
def _retriever() -> Retriever:
  s = get_settings()
  return Retriever(s, get_embeddings(s), get_chat_model())


def _hits(question: str, k: int = 5) -> list[Document]:
  return [d for d, _ in _retriever().search(question, k=k)[1]]


def answer_correct(text: str, evidence: tuple[str, ...]) -> bool:
  """Does an answer state the evidence? Numbers must appear as whole numbers; other evidence as words."""
  norm = " ".join(text.replace("**", "").replace(",", "").split()).casefold()
  numbers = set(re.findall(r"\d+(?:\.\d+)?", norm))
  for e in evidence:
    digits = re.findall(r"\d+(?:\.\d+)?", e.replace(",", ""))
    if digits:
      if not all(d in numbers for d in digits):
        return False
    elif e.casefold() not in norm:
      return False
  return True


def refused(text: str) -> bool:
  return bool(
    re.search(
      r"(don'?t|do not|doesn'?t|does not) (know|have|contain|mention|include|say|specify)"
      r"|not (available|mentioned|specified|provided|found)|no information|cannot (find|answer)"
      r"|isn'?t (mentioned|specified|available)",
      text.replace("’", "'"),
      re.I,
    )
  )


# ── Lab 1: the RAG chain and prompting ──────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def rag_by_hand(model, question):
  docs = _hits(question)  # 1. retrieve
  context = "\n\n".join(d.page_content for d in docs)  # 2. format the context
  prompt = f"Answer the question using this context.\n\nContext:\n{context}\n\nQuestion: {question}"  # 3. prompt
  reply = model.invoke(prompt)  # 4. generate
  return docs, reply.text


def rag_as_one_chain(model):
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.runnables import RunnableLambda, RunnablePassthrough

  prompt = ChatPromptTemplate.from_template(
    "Answer the question using this context.\n\nContext:\n{context}\n\nQuestion: {question}"
  )
  retrieve = RunnableLambda(_hits)
  join = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))
  chain = {"context": retrieve | join, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
  return chain


def with_sources(model):
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.runnables import RunnableLambda, RunnablePassthrough

  prompt = ChatPromptTemplate.from_template(
    "Answer the question using this context.\n\nContext:\n{context}\n\nQuestion: {question}"
  )
  chain = RunnablePassthrough.assign(docs=RunnableLambda(lambda x: _hits(x["question"]))).assign(
    answer=RunnableLambda(
      lambda x: {"context": "\n\n".join(d.page_content for d in x["docs"]), "question": x["question"]}
    )
    | prompt
    | model
    | RunnableLambda(lambda m: m.text)
  )  # the documents travel next to the answer, so the UI can show them
  return chain


GROUNDED = (
  "Answer ONLY from the context. If the context doesn't contain the answer, say \"I don't know\". "
  "Answer in at most two sentences."
)


def grounded_prompt(model, question):
  docs = _hits(question)
  context = format_sources(docs)  # numbered sources with their metadata, separated from the instructions
  reply = cache.text(model, [("system", f"{GROUNDED}\n\nSources:\n{context}"), ("human", question)])
  return reply


def position_test(model, question, answer_chunk, distractors, position):
  chunks = [*distractors]
  chunks.insert(position, answer_chunk)  # where in a long context the answer sits
  context = "\n\n".join(c.page_content for c in chunks)
  return cache.text(model, [("system", f"{GROUNDED}\n\nContext:\n{context}"), ("human", question)])


def token_budget(docs, max_tokens):
  from nimbus.rag.answer import fit_budget

  kept = fit_budget(docs, max_tokens)  # the best chunks that fit, in rank order; the rest are dropped
  return kept


@lab(
  "rag.rag-chain",
  "The RAG chain: retrieve, present the context, answer, and say when you don't know",
  uses_model=True,
  calls="~14 calls to Mistral (cached after the first run)",
)
def rag_chain():
  """The chain by hand and in LCEL; a naive prompt vs a grounded one on answerable and unanswerable questions; where
  the answer sits in a long context; a token budget."""
  model = get_chat_model()
  question = "Who approves remote work outside India?"
  docs, by_hand = rag_by_hand(model, question)
  chain = rag_as_one_chain(model)
  as_chain = chain.invoke(question)
  both = with_sources(model).invoke({"question": question})
  rows = []
  tests = [
    ("Who approves remote work outside India?", True),
    ("How many days of sick leave do employees get?", True),
    (UNANSWERABLE[0], False),
    (UNANSWERABLE[1], False),
  ]
  for q, answerable in tests:
    naive = cache.text(
      model,
      "Answer the question using this "
      f"context.\n\nContext:\n{chr(10).join(d.page_content for d in _hits(q))}\n\nQuestion: {q}",
    )
    grounded = grounded_prompt(model, q)
    rows.append([q, "answerable" if answerable else "not in the documents", naive, grounded])
  leave = corpus.question_for("How many days of paid annual leave does an employee get?")
  answer_chunk = next(
    c
    for c in corpus.chunks(300, 50, "v3-tesseract", "recursive", "headers")
    if leave and "24 days of paid" in c.page_content
  )
  distractors = [
    c
    for c in corpus.chunks(300, 50, "v3-tesseract", "recursive", "headers")
    if c.metadata["source"] not in ("handbook.md", "handbook_2024_archived.md")
  ][:24]
  positions = {}
  for position in (0, 12, 24):
    text = position_test(model, leave.question, answer_chunk, distractors, position)
    positions[position] = (text, "24" in text)
  ranked = _hits("What benefits do employees get?", k=10)
  kept = token_budget(ranked, 300)
  return [
    table(["Question", "", "naive prompt", "grounded prompt"], rows, title="Same retrieval, two prompts"),
    table(
      ["Answer chunk at position", "of", "answer", "right?"],
      [[p + 1, len(distractors) + 1, t[:120], "✓" if ok else "✗"] for p, (t, ok) in positions.items()],
      title="Lost in the middle? The answer at the start, middle and end of 25 chunks",
    ),
    callout(
      "Without grounding instructions the model fills gaps from its own knowledge; with them it says it doesn't "
      "know. Present sources numbered and labelled, separate from the instructions. Long contexts are read worst "
      "in the middle, "
      + (
        "though this model found the answer at every position here; "
        if all(ok for _, ok in positions.values())
        else "and here the answer in the middle was missed; "
      )
      + "production puts its best chunks at the start and the end anyway, and keeps only what fits a token budget.",
      "decision",
      "Prompting for RAG",
    ),
  ] + [
    example(
      "lc.l45.1",
      "RAG by hand",
      rag_by_hand,
      "Retrieve → format → prompt → generate.",
      [table(["sources", "answer"], [[", ".join(d.metadata["source"] for d in docs), by_hand]])],
    ),
    example(
      "lc.l45.2",
      "The same pipeline as one LCEL chain",
      rag_as_one_chain,
      "A dict of Runnables runs in parallel: retrieval fills `context`, the question passes through.",
      [answer(as_chain, title=question)],
    ),
    example(
      "lc.l45.3",
      "Returning the sources with the answer",
      with_sources,
      "`RunnablePassthrough.assign` adds `docs`, then `answer`, to the same dict.",
      [table(["keys", "sources"], [[str(list(both)), ", ".join(d.metadata["source"] for d in both["docs"])]])],
    ),
    example(
      "lc.l45.4",
      "Grounding: 'I don't know' when the context lacks it",
      grounded_prompt,
      "",
      [table(["Question", "naive", "grounded"], [[r[0], r[2][:100], r[3][:100]] for r in rows[2:]])],
    ),
    example("rag.l40.1", "A grounded prompt", grounded_prompt, "", [code(GROUNDED, "text")]),
    example(
      "rag.l40.2",
      "Numbered sources with metadata",
      grounded_prompt,
      "",
      [code(format_sources(_hits(question)[:2]), "text", "how the sources look")],
    ),
    example(
      "rag.l40.3",
      "Order matters: lost in the middle",
      position_test,
      "",
      [table(["position", "right?"], [[p + 1, "✓" if ok else "✗"] for p, (_, ok) in positions.items()])],
    ),
    example(
      "rag.l40.4",
      "A token budget",
      token_budget,
      f"{len(ranked)} ranked chunks, a 300-token budget: {len(kept)} kept.",
      [table(["kept"], [[d.page_content[:80]] for d in kept])],
    ),
  ]


# ── Lab 2: citations ────────────────────────────────────────────────────────────
INLINE = (
  "Answer ONLY from the numbered sources. Put [n] after each fact. "
  "If they don't contain the answer, say \"I don't know\"."
)


def inline_citations(model, question, docs):
  import re

  reply = cache.text(model, [("system", f"{INLINE}\n\nSources:\n{format_sources(docs)}"), ("human", question)])
  cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply)})  # markers → source numbers
  sources = {n: docs[n - 1].metadata["source"] for n in cited if 1 <= n <= len(docs)}  # → documents
  return reply, sources


class Quote(BaseModel):
  source: int = Field(description="The number of the source")
  quote: str = Field(description="The exact supporting words, copied verbatim")


class Cited(BaseModel):
  answer: str = Field(description="The answer, with [n] markers")
  quotes: list[Quote]


def structured_citations(model, question, docs):
  result = cache.structured(
    model, Cited, [("system", f"{INLINE}\n\nSources:\n{format_sources(docs)}"), ("human", question)]
  )
  return result  # typed data: the answer plus the exact quotes, ready to render and to check


def verify_quotes(result, docs):
  from nimbus.rag.answer import Citation

  verified, rejected = verify([Citation(source=q.source, quote=q.quote) for q in result.quotes], docs)
  return verified, rejected  # a quote that isn't in its source is a made-up citation


@lab(
  "rag.citations",
  "Citations: show where every fact came from, and check them",
  uses_model=True,
  calls="4 calls to Mistral (cached after the first run)",
)
def citations():
  """Inline [n] markers mapped back to documents, structured quotes, and verification, including a forged citation."""
  model = get_chat_model()
  question = "What are the payment terms of the supplier contract, and how much does a bag of Nilgiri Sunrise cost?"
  docs = _hits(question)
  reply, sources = inline_citations(model, question, docs)
  result = structured_citations(model, question, docs)
  verified, rejected = verify_quotes(result, docs)
  forged = result.model_copy(
    update={"quotes": [*result.quotes, Quote(source=1, quote="Nimbus refunds every opened bag in full.")]}
  )
  f_verified, f_rejected = verify_quotes(forged, docs)
  return [
    answer(reply, title="Inline citations", meta={f"[{n}]": s for n, s in sources.items()}),
    table(
      ["[n]", "quote", "source", "verified?"],
      [
        [
          q.source,
          q.quote,
          docs[q.source - 1].metadata["source"] if 1 <= q.source <= len(docs) else "—",
          "✓" if q in [Quote(**v.model_dump()) for v in verified] else "✗",
        ]
        for q in result.quotes
      ],
      title="Structured citations",
    ),
    callout(
      f"Markers turn an answer into something checkable: each [n] points at a document. Structured output adds the "
      f"exact quote, and code can then verify it: {len(verified)} of {len(result.quotes)} quotes appear in their "
      f"sources. A forged quote added for the test was rejected ({len(f_rejected)} rejected). Production drops "
      "unverified citations and counts them, so a made-up citation never reaches a user.",
      "decision",
      "Verified citations",
    ),
  ] + [
    example(
      "rag.l41.1",
      "Inline citations, mapped back",
      inline_citations,
      "",
      [answer(reply, title=question), table(["marker", "document"], [[f"[{n}]", s] for n, s in sources.items()])],
    ),
    example(
      "rag.l41.2",
      "Structured citations: answer + exact quotes",
      structured_citations,
      "",
      [code(result.model_dump_json(indent=2), "json")],
    ),
    example(
      "rag.l41.3",
      "Verify every quote",
      verify_quotes,
      "With a forged quote added: it isn't in source [1], so it's rejected.",
      [
        table(
          ["", "verified", "rejected"],
          [["model's quotes", len(verified), len(rejected)], ["+ a forged quote", len(f_verified), len(f_rejected)]],
        )
      ],
    ),
  ]


# ── Lab 3: conversational RAG ───────────────────────────────────────────────────
TURNS = [
  "How many days of paid annual leave do I get?",
  "What about sick days?",
  "And do I need a doctor's note for those?",
]


def follow_ups_alone():
  hits = _hits("What about sick days?", k=3)  # alone, the follow-up is barely a question
  return hits


def rewrite_with_history(model, history, question):
  from nimbus.rag.answer import REWRITE

  transcript = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text}" for m in history)
  standalone = cache.text(model, REWRITE.format(history=transcript, question=question))
  return standalone


def multi_turn(answerer, turns):
  from nimbus.rag.answer import remember

  history, results = [], []
  for question in turns:
    result = answerer.answer(question, history)  # rewrite → retrieve → answer
    history = remember(history, result)  # → remember
    results.append(result)
  return results


def agent_alternative(model, question_turns):
  from langchain.agents import create_agent
  from langchain_core.tools import tool
  from langgraph.checkpoint.memory import InMemorySaver

  @tool
  def search_documents(query: str) -> str:
    """Search Nimbus Coffee's documents. Pass a complete, standalone question."""
    return format_sources(_hits(query, k=3))

  agent = create_agent(model, tools=[search_documents], checkpointer=InMemorySaver(), system_prompt=GROUNDED)
  config = {"configurable": {"thread_id": "lab-conversation"}}
  replies = [agent.invoke({"messages": [HumanMessage(q)]}, config)["messages"] for q in question_turns]
  return replies  # the agent writes its own search queries from the conversation it remembers


@lab(
  "rag.conversational",
  "Conversational RAG: follow-up questions that depend on the history",
  uses_model=True,
  calls="~12 calls to Mistral",
)
def conversational():
  """Three turns where the 2nd and 3rd only make sense with the 1st: the production answerer, and an agent."""
  model = get_chat_model()
  alone = follow_ups_alone()
  answerer = Answerer(get_settings(), _retriever(), model)
  results = multi_turn(answerer, TURNS)
  history = [HumanMessage(TURNS[0]), AIMessage(results[0].answer)]
  standalone = rewrite_with_history(model, history, TURNS[1])
  agent_messages = agent_alternative(model, TURNS)

  def this_turn(msgs):  # the thread holds every turn: keep what came after the last question
    last = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    return [c["args"]["query"] for m in msgs[last:] if isinstance(m, AIMessage) for c in m.tool_calls]

  searches = [this_turn(msgs) for msgs in agent_messages]
  agent_answers = [msgs[-1].text for msgs in agent_messages]
  return [
    table(
      ["Turn", "question", "standalone question", "answer"],
      [[i + 1, r.question, r.standalone, r.answer] for i, r in enumerate(results)],
      title="The production answerer: rewrite → retrieve → answer → remember",
    ),
    table(
      ["Turn", "the agent searched for", "answer"],
      [
        [i + 1, " | ".join(s) or "(answered from memory, no search)", a]
        for i, (s, a) in enumerate(zip(searches, agent_answers, strict=True))
      ],
      title="The agent alternative: a retriever tool + a checkpointer",
    ),
    callout(
      f"'What about sick days?' means nothing to a search engine; rewritten with the history it became "
      f"'{results[1].standalone}'. The pipeline does that with one extra call per follow-up and stays predictable. "
      "An agent with a search tool and memory decides for itself when to search"
      + (
        f": here it searched {sum(1 for x in searches[1:] if not x)} follow-up(s) not at all and answered from "
        "the first search's results still in its memory: right this time, only because that chunk also held "
        "the answer. "
        if any(not x for x in searches[1:])
        else ", writing its own queries from the conversation. "
      )
      + "Production's answerer rewrites; the agent tracks build the second shape with checks around it.",
      "decision",
      "Follow-ups",
    ),
  ] + [
    example(
      "rag.l42.1",
      "The problem: a follow-up on its own",
      follow_ups_alone,
      "",
      [table(["result for 'What about sick days?'"], [[d.page_content[:90]] for d in alone])],
    ),
    example(
      "rag.l42.2",
      "Rewrite into a standalone question",
      rewrite_with_history,
      "",
      [table(["follow-up", "standalone"], [[TURNS[1], standalone]])],
    ),
    example(
      "rag.l42.3",
      "A multi-turn loop",
      multi_turn,
      "",
      [table(["turn", "standalone", "answer"], [[i + 1, r.standalone, r.answer[:120]] for i, r in enumerate(results)])],
    ),
    example(
      "rag.l42.4",
      "The agent alternative",
      agent_alternative,
      "",
      [table(["turn", "searches"], [[i + 1, str(s)] for i, s in enumerate(searches)])],
    ),
  ]


# ── Lab 4: structured data — SQL vs search ──────────────────────────────────────
SQL_QUESTIONS = {
  "How many menu items are vegan?": "select count(*) from menu where vegan = 'yes'",
  "What is the average price of a hot drink?": "select round(avg(price_rupees), 2) from menu where category = 'hot'",
  "What is the total value of delivered orders?": "select sum(total) from orders where status = 'delivered'",
}


def load_tables(path):
  import csv
  import json
  import sqlite3
  from contextlib import closing

  with closing(sqlite3.connect(path)) as db:  # sqlite3's own `with` commits but doesn't close
    db.execute("create table if not exists menu (item text, category text, price_rupees integer, vegan text)")
    with open(DATA / "menu.csv", newline="") as f:
      db.executemany("insert into menu values (?, ?, ?, ?)", [tuple(r.values()) for r in csv.DictReader(f)])
    db.execute(
      "create table if not exists orders (order_id text, customer text, total integer, status text, date text)"
    )
    orders = json.loads((DATA / "orders.json").read_text())
    db.executemany(
      "insert into orders values (?, ?, ?, ?, ?)",
      [(o["order_id"], o["customer"], o["total"], o["status"], o["date"]) for o in orders],
    )
    db.commit()
  return path


class SQL(BaseModel):
  sql: str = Field(description="One SQLite SELECT statement")


def text_to_sql(model, path, question):
  import sqlite3
  from contextlib import closing

  with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:  # read-only: nothing can be changed
    tables = [r[0] for r in db.execute("select name from sqlite_master where type = 'table'")]
    schema = "\n\n".join(  # CREATE TABLE + 3 sample rows: the model sees 'yes'/'no' and 'hot', not guesses
      db.execute("select sql from sqlite_master where name = ?", (t,)).fetchone()[0]
      + f"\n/* sample rows: {db.execute(f'select * from {t} limit 3').fetchall()} */"
      for t in tables
    )
    sql = cache.structured(model, SQL, f"Schema:\n{schema}\n\nWrite one SQLite SELECT for: {question}").sql
    if not sql.lstrip().lower().startswith("select"):
      raise ValueError("only SELECT statements are allowed")
    rows = db.execute(sql).fetchmany(50)  # a row limit
  reply = cache.text(model, f"Question: {question}\nSQL: {sql}\nRows: {rows}\nAnswer in one sentence.")
  return sql, rows, reply


def search_cannot_count(model, question):
  return cache.text(
    model, [("system", GROUNDED + "\n\nSources:\n" + format_sources(_hits(question))), ("human", question)]
  )


@lab(
  "rag.sql-vs-text",
  "Structured data: query a database instead of searching text",
  uses_model=True,
  calls="~9 calls to Mistral (cached after the first run)",
)
def sql_vs_text():
  """Counting, averaging and summing over all rows: vector search vs text-to-SQL on the same tables."""
  model = get_chat_model()
  path = PROJECT_DIR / ".cache" / "labs" / "m07.sqlite"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.unlink(missing_ok=True)
  load_tables(path)
  rows = []
  for question, truth_sql in SQL_QUESTIONS.items():
    with closing(sqlite3.connect(path)) as db:
      truth = db.execute(truth_sql).fetchone()[0]
    via_search = search_cannot_count(model, question)
    sql, result, via_sql = text_to_sql(model, path, question)
    rows.append([question, truth, via_search, via_sql, sql])
  path.unlink(missing_ok=True)
  hit = lambda text, truth: str(truth).rstrip("0").rstrip(".") in text.replace(",", "")  # noqa: E731
  score = [
    {"variant": "vector search + RAG prompt", "right": sum(hit(r[2], r[1]) for r in rows)},
    {"variant": "text-to-SQL", "right": sum(hit(r[3], r[1]) for r in rows)},
  ]
  return [
    scorecard(
      score, [Metric("right", f"right answers (of {len(rows)})", "higher", "int")], title="Aggregations over every row"
    ),
    table(["Question", "true value", "search", "SQL", "query"], rows),
    callout(
      "Search returns a few similar chunks, so 'how many', 'average' and 'total' over all rows are answered from a "
      "sample (or refused). A database answers them exactly. Route aggregate questions about structured data to SQL "
      "with a read-only connection, SELECT only, and a row limit; the Data track builds that into a guarded agent.",
      "decision",
      "Search or SQL",
    ),
  ] + [
    example(
      "rag.l43.1",
      "Load the tables into SQLite",
      load_tables,
      "",
      [table(["table", "rows"], [["menu", 8], ["orders", 4]])],
    ),
    example(
      "rag.l43.2",
      "Why search fails at aggregation",
      search_cannot_count,
      "",
      [table(["Question", "true", "search answer"], [[r[0], r[1], r[2][:120]] for r in rows])],
    ),
    example(
      "rag.l43.3",
      "Text-to-SQL: schema → query → rows → answer",
      text_to_sql,
      "",
      [table(["Question", "SQL", "answer"], [[r[0], r[4], r[3][:120]] for r in rows])],
    ),
    example(
      "rag.l43.4",
      "Safety: read-only, SELECT only, a row limit",
      text_to_sql,
      "`mode=ro` makes the connection read-only; a non-SELECT is refused before running; fetchmany(50) caps rows.",
      [
        table(
          ["guard", "how"],
          [
            ["read-only", "file:...?mode=ro"],
            ["SELECT only", "checked before executing"],
            ["row limit", "fetchmany(50)"],
          ],
        )
      ],
    ),
  ]


# ── Production path: the answerer ───────────────────────────────────────────────
NAIVE = "Answer the question using this context.\n\nContext:\n{context}\n\nQuestion: {question}"


def evaluate_answers(limit: int | None = None) -> dict:
  model = get_chat_model()
  answerer = Answerer(get_settings(), _retriever(), model)
  questions = list(corpus.questions())[:limit] if limit else list(corpus.questions())
  results = {"naive prompt": [], "answerer (production)": []}
  for q in [*[(q.question, q.evidence) for q in questions], *[(u, None) for u in UNANSWERABLE]]:
    question, evidence = q
    recorder = UsageRecorder()
    started = time.perf_counter()
    docs = _hits(question)
    naive = model.invoke(
      NAIVE.format(context="\n\n".join(d.page_content for d in docs), question=question), {"callbacks": [recorder]}
    ).text
    results["naive prompt"].append(
      (question, evidence, naive, [], [], recorder, (time.perf_counter() - started) * 1000)
    )
    r = answerer.answer(question)
    results["answerer (production)"].append(
      (
        question,
        evidence,
        r.answer if r.answerable else f"I don't know. {r.answer}",
        r.citations,
        r.rejected,
        r.recorder,
        r.ms,
      )
    )
  return results


@lab(
  "rag.answer-eval",
  "Evaluate: a naive RAG prompt vs the production answerer",
  uses_model=True,
  calls="~100 calls to Mistral (45 answerable + 6 unanswerable questions, two ways): about 4 minutes",
  inputs=[Input("limit", "Answerable questions to use (45 = all)", "number", default=45, min=5, max=45)],
)
def answer_eval(limit: int):
  """Both use the same production retriever; only the prompt, structure and checks differ."""
  started = time.perf_counter()
  results = evaluate_answers(limit)
  seconds = time.perf_counter() - started
  rows = []
  for name, items in results.items():
    answerable = [i for i in items if i[1] is not None]
    unanswerable = [i for i in items if i[1] is None]
    correct = sum(answer_correct(i[2], i[1]) for i in answerable)
    refusals = sum(refused(i[2]) for i in unanswerable)
    cited = [c for i in answerable for c in i[3]]
    rejected = sum(len(i[4]) for i in items)
    usage = [i[5].usage() for i in items]
    rows.append(
      {
        "variant": name,
        "correct": round(correct / len(answerable), 3),
        "refused_right": f"{refusals}/{len(unanswerable)}",
        "invented": len(unanswerable) - refusals,
        "citations": len(cited) if "production" in name else "—",
        "rejected": rejected if "production" in name else "—",
        "tokens": round(sum(u.input_tokens + u.output_tokens for u in usage) / len(items)),
        "p50_ms": round(sorted(i[6] for i in items)[len(items) // 2]),
      }
    )
  sample = [
    [i[0], i[2][:140], "✓" if (answer_correct(i[2], i[1]) if i[1] else refused(i[2])) else "✗"]
    for i in results["answerer (production)"][-8:]
  ]
  return [
    scorecard(
      rows,
      [
        Metric(
          "correct",
          "correct (answerable)",
          "higher",
          "ratio",
          "The answer states the evidence (numbers as whole numbers)",
        ),
        Metric("refused_right", "said 'I don't know' (unanswerable)", "none", "text"),
        Metric("invented", "answered the unanswerable", "lower", "int"),
        Metric("citations", "verified citations", "none", "int"),
        Metric("rejected", "citations rejected", "lower", "int", "Quote not found in the cited source"),
        Metric("tokens", "tokens per question", "lower", "int"),
        Metric("p50_ms", "latency p50", "lower", "ms"),
      ],
      title=f"{limit} answerable + {len(UNANSWERABLE)} unanswerable questions · {seconds:.0f} s",
    ),
    table(["Question", "production answer", "✓"], sample, title="The last questions (unanswerable ones at the end)"),
    callout(
      f"The production answerer was correct on {rows[1]['correct']:.0%} of answerable questions (naive: "
      f"{rows[0]['correct']:.0%}) and answered {rows[1]['invented']} of the unanswerable ones (naive: "
      f"{rows[0]['invented']}). Every fact carries a verified citation; rejected quotes were dropped. It costs "
      "more tokens per question (structured output, numbered sources, rules): the price of answers you can check.",
      "decision",
      "Ship the answerer",
    ),
  ]
