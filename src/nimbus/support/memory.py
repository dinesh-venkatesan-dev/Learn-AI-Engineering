"""Support desk memory, in the course Postgres.

  conversations  a checkpointer (PostgresSaver): every step of every conversation, per thread_id. A thread id is
                 "<customer_id>:<conversation>", so everything about one customer can be found and deleted.
  memories       a store (PostgresStore, with pgvector search): lasting facts a customer stated about themselves,
                 under ("customers", <customer_id>, "memories"), recalled by meaning in later conversations.

    with desk_persistence(settings, embeddings) as (checkpointer, memories):
      desk = build_support_desk(model, store, checkpointer=checkpointer, memory=memories)

Privacy: forget_customer() deletes a customer's conversations and memories; expire_conversations() is the nightly
retention job; set NIMBUS_CHECKPOINT_KEY (32 bytes, hex) to encrypt checkpoints at rest.
"""

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

ORDER_ID = re.compile(r"NC-\d+", re.IGNORECASE)
DUPLICATE_SCORE = 0.9  # a new memory this close to an existing one is the same fact again
# The desk's own types saved in checkpoints. Unlisted classes load with a warning today and will be blocked by a
# future LangGraph (seen as "Deserializing unregistered type ... from checkpoint").
DESK_TYPES = [("nimbus.support.agent", "SupportOutcome"), ("nimbus.support.desk", "Ticket")]


def desk_serializer(key: bytes | None = None):
  """Checkpoint serializer that may load the desk's types; encrypted with AES when a key is given."""
  from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

  serde = JsonPlusSerializer(allowed_msgpack_modules=DESK_TYPES)
  if key is None:
    return serde
  from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

  return EncryptedSerializer.from_pycryptodome_aes(serde=serde, key=key)


def memory_checkpointer():
  """An in-process checkpointer for the desk (tests, evaluations)."""
  from langgraph.checkpoint.memory import InMemorySaver

  return InMemorySaver(serde=desk_serializer())


def namespace(customer_id: str) -> tuple[str, ...]:
  return ("customers", customer_id, "memories")


def thread_id(customer_id: str, conversation: str) -> str:
  return f"{customer_id}:{conversation}"


def postgres_url(database_url: str) -> str:
  return database_url.replace("postgresql+psycopg://", "postgresql://")  # psycopg's own URL form


@contextmanager
def desk_persistence(settings, embeddings) -> Iterator[tuple]:
  """(checkpointer, memory store) on the course database, tables created if missing; connections closed on exit."""
  from langgraph.checkpoint.postgres import PostgresSaver
  from langgraph.store.postgres import PostgresStore

  url = postgres_url(settings.database_url)
  key = bytes.fromhex(settings.checkpoint_key.get_secret_value()) if settings.checkpoint_key else None
  index = {"embed": embeddings, "dims": settings.embedding_dims, "fields": ["fact"]}
  with (
    PostgresSaver.from_conn_string(url) as checkpointer,
    PostgresStore.from_conn_string(url, index=index) as memories,
  ):
    checkpointer.serde = desk_serializer(key)
    checkpointer.setup()
    memories.setup()
    yield checkpointer, memories


def remember_fact(kind: Literal["diet", "taste", "delivery", "contact"], fact: str, runtime: ToolRuntime) -> str:
  """Save a lasting fact the customer stated about THEMSELVES that will matter in future conversations: dietary needs
  or allergies (diet), taste preferences (taste), delivery preferences (delivery), how they want to be contacted
  (contact). fact: one short third-person sentence, e.g. 'Is lactose intolerant.'. Never save questions, one-off
  requests, order details, complaints, or anything a tool told you."""
  if runtime.store is None:
    return "Memory is not available."
  if ORDER_ID.search(fact) or len(fact) > 140:  # order details belong to the order system, not to memory
    return "Not saved: memories are lasting facts about the customer, not order details."
  ns = namespace(runtime.context.customer_id)
  same = runtime.store.search(ns, query=fact, limit=1)
  if same and (same[0].score or 0) >= DUPLICATE_SCORE:
    return "Already known."
  runtime.store.put(ns, str(uuid.uuid4()), {"fact": fact, "kind": kind, "saved": datetime.now(UTC).isoformat()})
  return "Saved."


remember = tool("remember")(remember_fact)  # the tool the agent sees; the function above is its code


def recall(store: BaseStore, customer_id: str, query: str, limit: int = 3) -> list[str]:
  """The customer's memories most related to the message (by meaning)."""
  return [item.value["fact"] for item in store.search(namespace(customer_id), query=query, limit=limit)]


def all_memories(store: BaseStore, customer_id: str) -> list[dict]:
  return [item.value for item in store.search(namespace(customer_id), limit=100)]


def conversations(checkpointer) -> dict[str, datetime]:
  """thread_id → when it was last active (list() returns newest checkpoints first)."""
  seen: dict[str, datetime] = {}
  for item in checkpointer.list(None):
    seen.setdefault(item.config["configurable"]["thread_id"], datetime.fromisoformat(item.checkpoint["ts"]))
  return seen


def forget_customer(checkpointer, store: BaseStore, customer_id: str) -> dict[str, int]:
  """A "forget me" request: every conversation and every memory of the customer, deleted."""
  threads = [t for t in conversations(checkpointer) if t.startswith(f"{customer_id}:")]
  for t in threads:
    checkpointer.delete_thread(t)
  items = store.search(namespace(customer_id), limit=1000)
  for item in items:
    store.delete(item.namespace, item.key)
  return {"conversations": len(threads), "memories": len(items)}


def expire_conversations(checkpointer, max_age: timedelta) -> list[str]:
  """Retention: delete conversations inactive for longer than max_age (run nightly). Memories are kept."""
  now = datetime.now(UTC)
  old = [t for t, when in conversations(checkpointer).items() if now - when > max_age]
  for t in old:
    checkpointer.delete_thread(t)
  return old
