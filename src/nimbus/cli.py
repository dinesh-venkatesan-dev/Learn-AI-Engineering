"""Command line for the Nimbus service.

uv run nimbus rag ingest [--force]      index every document in the data folder (incremental)
uv run nimbus rag search "question"     the production retriever's top chunks (hybrid, reranked, valid today)
uv run nimbus rag ask "question"        a grounded answer with verified citations [--audience public|staff]
uv run nimbus rag eval [-k 5]           hit/precision/recall/nDCG@k, MRR, coverage and latency on the evaluation set
uv run nimbus rag synth [-n 24]         generate the synthetic evaluation set (data/eval/synthetic.json)
uv run nimbus rag stats                 what's in the index
uv run nimbus assistant ask --customer c-1042 "question"   the Nimbus assistant (any question: account, policies,
                                        cafés; --architecture single|router|supervisor)
uv run nimbus support triage "message"  route one customer message (JSON), then stream a reply draft
uv run nimbus support eval [--prompt v2] score triage on the labelled tickets
uv run nimbus support desk --customer c-1042 [--conversation a] "message"   the support desk (remembers conversations
                                        and customer facts in Postgres): progress as it works, then the ticket
uv run nimbus support approve --customer c-6060 --conversation r1 --decision approve|edit|reject [--args JSON]
                                        [--reason ...]   answer the approval the conversation is waiting on
uv run nimbus support forget --customer c-1042   delete the customer's conversations and memories
"""

import argparse
import json
import sys

from nimbus.config import get_settings
from nimbus.logging_setup import setup_logging
from nimbus.rag.evaluation import evaluate, load_questions
from nimbus.rag.ingestion.embeddings import get_embeddings
from nimbus.rag.ingestion.pipeline import IngestionPipeline, StepResult
from nimbus.rag.ingestion.store import VectorIndex
from nimbus.rag.retriever import Retriever


def print_step(result: StepResult) -> None:
  stats = "  ".join(f"{k}={v}" for k, v in result.stats.items())
  print(f"  {result.step:<6} {result.ms:>6} ms   {stats}")


def main() -> None:
  parser = argparse.ArgumentParser(prog="nimbus", description="The Nimbus AI service.")
  areas = parser.add_subparsers(dest="area", required=True)
  rag = areas.add_parser("rag", help="retrieval-augmented generation").add_subparsers(dest="command", required=True)
  ingest = rag.add_parser("ingest", help="index every document in the data folder (incremental)")
  ingest.add_argument("--force", action="store_true", help="re-embed and rewrite every chunk, even unchanged ones")
  search = rag.add_parser("search", help="the chunks closest in meaning to a question")
  search.add_argument("query")
  search.add_argument("-k", type=int, default=3)
  evaluate_cmd = rag.add_parser("eval", help="score retrieval on the evaluation set")
  evaluate_cmd.add_argument("-k", type=int, default=5)
  rag.add_parser("stats", help="what's in the index")
  ask = rag.add_parser("ask", help="a grounded answer with verified citations")
  ask.add_argument("question")
  ask.add_argument("--audience", default="staff", choices=["public", "staff"])
  synth = rag.add_parser("synth", help="generate the synthetic evaluation set from the corpus")
  synth.add_argument("-n", type=int, default=24, help="chunks to write questions from (one per source first)")
  assistant = areas.add_parser("assistant", help="the Nimbus assistant").add_subparsers(dest="command", required=True)
  assistant_ask = assistant.add_parser("ask", help="ask the assistant as a logged-in customer")
  assistant_ask.add_argument("--customer", required=True)
  assistant_ask.add_argument("--architecture", default=None, choices=["single", "router", "supervisor"])
  assistant_ask.add_argument("question")
  support = areas.add_parser("support", help="customer support").add_subparsers(dest="command", required=True)
  triage_cmd = support.add_parser("triage", help="route one customer message, then draft a reply")
  triage_cmd.add_argument("message")
  desk_cmd = support.add_parser("desk", help="the support desk agent for one logged-in customer")
  desk_cmd.add_argument("--customer", required=True, help="customer id, as the login would set it (e.g. c-1042)")
  desk_cmd.add_argument("--conversation", default="cli", help="conversation id: the same id continues it")
  desk_cmd.add_argument("message")
  approve_cmd = support.add_parser("approve", help="answer the approval a desk conversation is waiting on")
  approve_cmd.add_argument("--customer", required=True)
  approve_cmd.add_argument("--conversation", default="cli")
  approve_cmd.add_argument("--decision", required=True, choices=["approve", "edit", "reject"])
  approve_cmd.add_argument(
    "--args", default=None, help="for edit: the corrected arguments as JSON, e.g. '{\"weeks\": 4}'"
  )
  approve_cmd.add_argument("--reason", default=None, help="for reject: told to the customer")
  forget_cmd = support.add_parser("forget", help="delete a customer's conversations and memories")
  forget_cmd.add_argument("--customer", required=True)
  support_eval = support.add_parser("eval", help="score triage on the labelled tickets")
  support_eval.add_argument("--prompt", default=None, help="prompt version (default: NIMBUS_TRIAGE_PROMPT_VERSION)")
  support_eval.add_argument("--method", default=None, choices=["function_calling", "json_schema"])
  args = parser.parse_args()

  settings = get_settings()
  setup_logging(settings.log_level)
  if args.area == "assistant":
    sys.exit(run_assistant(args, settings))
  if args.area == "support":
    sys.exit(run_support(args, settings))
  embeddings = get_embeddings(settings)
  if args.command in ("ask", "synth"):
    sys.exit(run_rag_with_model(args, settings, embeddings))

  if args.command == "ingest":
    report = IngestionPipeline(settings, embeddings).run(print_step, force=args.force)
    print(json.dumps(report.__dict__, indent=2))
    sys.exit(1 if report.failed and not report.chunks else 0)

  with VectorIndex(settings, embeddings) as index:
    retriever = Retriever(settings, embeddings)
    if args.command == "search":
      prepared, hits = retriever.search(args.query, args.k)
      print("  " + " · ".join(prepared.steps))
      for document, score in hits:
        print(
          f"{score:.3f}  {document.metadata['source']} p{document.metadata.get('page')}  "
          f"{document.page_content[:100]!r}"
        )
    elif args.command == "eval":
      report = evaluate(
        lambda q, k: [d for d, _ in retriever.search(q, k)[1]],
        load_questions(settings.eval_questions),
        index.texts(),
        k=args.k,
        indexed_sources=index.sources(),
      )
      print(json.dumps({**report.summary(), "by_kind": report.by_kind}, indent=2))
    else:
      print(json.dumps(index.stats(), indent=2, default=str))


if __name__ == "__main__":
  main()


def entry() -> None:
  """The console script: run, then close the cached model clients' connections."""
  import asyncio

  from nimbus.llm.models import aclose_clients

  try:
    main()
  finally:
    asyncio.run(aclose_clients())


def run_rag_with_model(args: argparse.Namespace, settings, embeddings) -> int:
  from nimbus.llm import get_chat_model

  model = get_chat_model(settings)
  retriever = Retriever(settings, embeddings, model)
  if args.command == "ask":
    from nimbus.rag.answer import Answerer

    result = Answerer(settings, retriever, model).answer(args.question, audience=args.audience)
    print(result.answer)
    for c in result.citations:
      print(f"  [{c.source}] {result.sources[c.source - 1].metadata['source']}: {c.quote!r}")
    print("  " + " · ".join(result.steps), file=sys.stderr)
    return 0
  return _synth(settings, model, args.n)


def _synth(settings, model, n: int) -> int:
  """One chunk per source first, then more, skipping tiny chunks and code; questions checked and saved."""
  from langchain_core.documents import Document
  from sqlalchemy import create_engine, text

  from nimbus.rag.quality import generate_questions, save_questions

  engine = create_engine(settings.database_url)
  try:
    with engine.connect() as db:
      rows = db.execute(
        text(
          f"select content, source from {settings.table_name} "
          "where length(content) > 150 and file_type <> 'code' order by source, langchain_id"
        )
      ).all()
  finally:
    engine.dispose()
  by_source: dict[str, list[Document]] = {}
  for content, source in rows:
    by_source.setdefault(source, []).append(Document(content, metadata={"source": source}))
  picked = [docs[0] for docs in by_source.values()]
  picked += [d for docs in by_source.values() for d in docs[1:]][: max(0, n - len(picked))]
  kept, rejected = generate_questions(picked[:n], model)
  path = settings.eval_questions.parent / "synthetic.json"
  save_questions(kept, path)
  print(json.dumps({"written": len(kept), "rejected": rejected, "path": str(path)}, indent=2, ensure_ascii=False))
  return 0


def run_assistant(args: argparse.Namespace, settings) -> int:
  from nimbus.assistant.architectures import build_router, build_single, build_supervisor
  from nimbus.assistant.specialists import build_specialists
  from nimbus.llm import get_chat_model
  from nimbus.rag.retriever import Retriever

  model = get_chat_model(settings)
  retriever = Retriever(settings, get_embeddings(settings), model)
  specialists = build_specialists(settings, model, retriever)
  architecture = args.architecture or settings.assistant_architecture
  if architecture == "single":
    _, ask = build_single(model, specialists, retriever)
  elif architecture == "router":
    _, ask = build_router(model, specialists)
  else:
    _, ask = build_supervisor(model, specialists)
  result = ask(args.customer, args.question)
  print(json.dumps({"reply": result.reply, "used": result.specialists, "architecture": architecture}, indent=2))
  return 0


def run_forget(settings, customer_id: str) -> int:
  from nimbus.support.memory import desk_persistence, forget_customer

  with desk_persistence(settings, get_embeddings(settings)) as (checkpointer, memories):
    print(json.dumps(forget_customer(checkpointer, memories, customer_id)))
  return 0


def run_desk(
  model, settings, customer_id: str, conversation: str, message: str | None, decision: dict | None = None
) -> int:
  """One message to the desk, or (decision given) a reviewer's answer to the question it's waiting on."""
  from langchain_core.messages import HumanMessage
  from langgraph.types import Command

  from nimbus.support.agent import CustomerContext
  from nimbus.support.desk import build_support_desk
  from nimbus.support.memory import desk_persistence, thread_id
  from nimbus.support.tools import SupportStore

  store = SupportStore()
  if customer_id not in store.customers:
    print(f"unknown customer {customer_id}", file=sys.stderr)
    return 1
  context = CustomerContext(customer_id, store.customers[customer_id])
  config = {"configurable": {"thread_id": thread_id(customer_id, conversation)}}
  with desk_persistence(settings, get_embeddings(settings)) as (checkpointer, memories):
    desk = build_support_desk(
      model,
      store,
      checkpointer=checkpointer,
      memory=memories,
      history_tokens=settings.desk_history_tokens,
      refunds=True,
      review=True,  # the customer confirms subscription changes, a teammate approves refunds
    )
    if decision is None:
      payload = {"messages": [HumanMessage(message)], "customer_id": customer_id, "actions": []}
    elif desk.get_state(config).interrupts:
      payload = Command(resume=decision)
    else:
      print("nothing is waiting for approval in this conversation", file=sys.stderr)
      return 1
    for mode, data in desk.stream(payload, config, stream_mode=["custom", "values"], context=context):
      if mode == "custom":
        print(f"… {data['progress']}", file=sys.stderr, flush=True)  # progress while the agent works
    snapshot = desk.get_state(config)
  if snapshot.interrupts:  # paused: show what's waiting; `nimbus support approve` answers it
    print(json.dumps({"waiting": snapshot.interrupts[0].value}, ensure_ascii=False, indent=2))
  else:
    print(snapshot.values["ticket"].model_dump_json(indent=2))
  return 0


def run_support(args: argparse.Namespace, settings) -> int:
  from nimbus.llm import get_chat_model, model_name
  from nimbus.support.evaluation import evaluate as evaluate_triage
  from nimbus.support.evaluation import load_tickets
  from nimbus.support.schema import Ticket
  from nimbus.support.triage import Triager, stream_reply

  model = get_chat_model(settings)
  if args.command == "desk":
    return run_desk(model, settings, args.customer, args.conversation, args.message)
  if args.command == "approve":
    decision = {"decision": args.decision, "args": json.loads(args.args) if args.args else None, "reason": args.reason}
    return run_desk(model, settings, args.customer, args.conversation, None, decision)
  if args.command == "forget":
    return run_forget(settings, args.customer)
  method = getattr(args, "method", None) or settings.triage_method
  triager = Triager(model, getattr(args, "prompt", None) or settings.triage_prompt_version, method)
  if args.command == "triage":
    ticket = Ticket(id="cli", text=args.message)
    result = triager.triage(ticket)
    if result.triage is None:
      print(f"failed ({result.action}): {result.error}", file=sys.stderr)
      return 1
    print(result.triage.model_dump_json(indent=2))
    for piece in stream_reply(model, ticket, result.triage):
      print(piece, end="", flush=True)
    print()
    return 0
  report = evaluate_triage(triager, load_tickets(settings.eval_tickets), model_name(model), settings.triage_concurrency)
  print(json.dumps(report.summary(), indent=2))
  return 0
