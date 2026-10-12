"""Memory evaluation: short conversations, each turn in a named conversation ("a", "b"), scored on the last reply.

  follow-up  the second message only makes sense with the first (same conversation): needs the checkpointer
  recall     a fact stated in conversation a is needed in conversation b: needs the memory store
  noise      nothing worth remembering: any memory saved is noise

Variants: "none" (every message on its own), "threads" (a checkpointer), "threads+memory" (checkpointer and store).
"""

import json
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from nimbus.llm.usage import UsageRecorder
from nimbus.support.agent import CustomerContext
from nimbus.support.desk import build_support_desk
from nimbus.support.memory import all_memories, memory_checkpointer, thread_id
from nimbus.support.tools import SupportStore


def load_conversations(path: Path) -> list[dict]:
  return json.loads(path.read_text())


def run_conversation(model, conversation: dict, variant: str, embeddings) -> dict:
  """Fresh checkpointer and store per conversation (in memory: the scores don't depend on earlier runs)."""
  store = SupportStore()
  checkpointer = memory_checkpointer() if variant != "none" else None
  memory = (
    InMemoryStore(index={"embed": embeddings, "dims": 384, "fields": ["fact"]}) if variant == "threads+memory" else None
  )
  desk = build_support_desk(model, store, checkpointer=checkpointer, memory=memory)
  cid = conversation["customer_id"]
  context = CustomerContext(cid, store.customers.get(cid, "the customer"))
  recorder, started, result = UsageRecorder(), time.perf_counter(), None
  actions: dict[str, list[str]] = {}  # per thread: an action taken anywhere in the conversation counts
  for n, (conv, text) in enumerate(conversation["turns"]):
    thread = thread_id(cid, conv if checkpointer else f"{conv}{n}")
    config = {"callbacks": [recorder], "configurable": {"thread_id": thread}}
    result = desk.invoke({"messages": [HumanMessage(text)], "customer_id": cid, "actions": []}, config, context=context)
    actions[thread] = result.get("actions", [])
  ticket = result["ticket"]
  return {
    "reply": ticket.reply,
    "actions": [a for thread_actions in actions.values() for a in thread_actions],
    "memories": [m["fact"] for m in all_memories(memory, cid)] if memory is not None else [],
    "calls": len(recorder.calls),
    "tokens": recorder.usage().input_tokens + recorder.usage().output_tokens,
    "ms": (time.perf_counter() - started) * 1000,
  }


def unrequested_actions(conversation: dict, run: dict) -> list[str]:
  """Actions taken in a conversation that asked for none (seen: a delivery preference 'handled' by pausing the
  subscription for 6 weeks, when no tool fitted)."""
  return [] if conversation.get("actions") else run["actions"]


def score(conversation: dict, run: dict) -> bool:
  if unrequested_actions(conversation, run):
    return False
  reply = run["reply"].casefold().replace("’", "'")
  numbers = {int(n) for n in re.findall(r"\d+", reply.replace(",", ""))}
  if conversation["kind"] == "noise":
    return not run["memories"]
  return (
    all(n in numbers for n in conversation.get("answer_numbers", []))
    and not any(n in numbers for n in conversation.get("not_numbers", []))  # e.g. every order's points listed
    and (not conversation.get("answer_any") or any(a in reply for a in conversation["answer_any"]))
    and all(a in " | ".join(run["actions"]) for a in conversation.get("actions", []))
  )
