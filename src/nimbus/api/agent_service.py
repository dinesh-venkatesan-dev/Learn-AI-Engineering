"""The support agent as a service: built ONCE at start-up, with a checkpointer so each conversation (thread) keeps
its memory, and guarded for production traffic.

  AgentService.start(settings, model)   in the lifespan: the agent, the checkpointer (memory or Postgres)
  ask(...)                               one turn: a slot (semaphore), a time budget (504), errors → 503/502
  stream(...)                            typed events: tool, token, done; stops when the client disconnects
  history(thread_id)                     the conversation so far, read from the checkpointer
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from nimbus.api.errors import ApiError, ModelError, ModelUnavailable
from nimbus.api.streaming import in_own_task
from nimbus.llm.errors import classify
from nimbus.support.agent import CustomerContext, build_support_agent
from nimbus.support.tools import SupportStore

log = logging.getLogger("nimbus.api.agent")


class AgentTimeout(ApiError):
  status, code = 504, "timeout"


class UnknownCustomer(ApiError):
  status, code = 404, "unknown_customer"


@dataclass
class Turn:
  reply: str
  actions: list[str]
  needs_human: bool


class AgentService:
  def __init__(self, agent, store: SupportStore, concurrency: int, timeout_s: float, checkpointer):
    self.agent, self.store, self.timeout_s, self.checkpointer = agent, store, timeout_s, checkpointer
    self.slots = asyncio.Semaphore(concurrency)  # at most this many agent runs at once; the rest queue
    self.stack: AsyncExitStack | None = None

  @classmethod
  async def start(cls, settings, model, store: SupportStore) -> "AgentService":
    stack = AsyncExitStack()
    if settings.agent_checkpointer == "postgres":
      from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

      url = settings.database_url.replace("+psycopg", "")
      checkpointer = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(url))
      await checkpointer.setup()  # creates its tables once
    else:
      from langgraph.checkpoint.memory import InMemorySaver

      checkpointer = InMemorySaver()  # fine for tests; forgets everything on restart
    agent = build_support_agent(model, store, checkpointer=checkpointer, output="text")
    service = cls(agent, store, settings.agent_concurrency, settings.agent_timeout_s, checkpointer)
    service.stack = stack
    return service

  async def close(self) -> None:
    if self.stack:
      await self.stack.aclose()

  def context(self, customer_id: str) -> CustomerContext:
    name = self.store.customers.get(customer_id)
    if name is None:
      raise UnknownCustomer(f"No customer {customer_id}.")
    return CustomerContext(customer_id=customer_id, name=name)

  @staticmethod
  def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

  async def ask(self, customer_id: str, thread_id: str, message: str) -> Turn:
    context = self.context(customer_id)
    try:
      async with self.slots, asyncio.timeout(self.timeout_s):
        before = await self.agent.aget_state(self.config(thread_id))
        done = len(before.values.get("actions", [])) if before.values else 0
        state = await self.agent.ainvoke({"messages": [HumanMessage(message)]}, self.config(thread_id), context=context)
    except TimeoutError as error:
      raise AgentTimeout(f"No answer within {self.timeout_s:.0f} s. Please try again.") from error
    except ApiError:
      raise
    except Exception as error:  # noqa: BLE001 - provider failures become 503 / 502
      raise (ModelUnavailable() if classify(error).action == "retry" else ModelError("The agent failed.")) from error
    actions = state.get("actions", [])[done:]  # this turn's actions only
    return Turn(state["messages"][-1].text, actions, any(a.startswith("escalate") for a in actions))

  async def stream(self, customer_id: str, thread_id: str, message: str, disconnected):
    """Yields (event, data). `disconnected()` is awaited between chunks: no tokens spent on an answer nobody reads."""
    context = self.context(customer_id)
    run = self.agent.astream(
      {"messages": [HumanMessage(message)]},
      self.config(thread_id),
      context=context,
      stream_mode=["updates", "messages"],
    )
    finished = False
    try:
      async with self.slots:
        async for mode, chunk in in_own_task(run):  # the client leaves → the run is cancelled at once (streaming.py)
          if await disconnected():
            log.info("client left thread %s: stopped", thread_id)
            return
          if mode == "messages":
            token, _ = chunk  # a chunk while the model streams (a whole message from models that don't)
            calls = token.tool_call_chunks if isinstance(token, AIMessageChunk) else getattr(token, "tool_calls", [])
            if isinstance(token, AIMessage) and token.text and not calls:
              yield "token", token.text
          else:
            for update in chunk.values():
              for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                if isinstance(m, AIMessage):
                  for call in m.tool_calls:
                    yield "tool", {"name": call["name"], "args": call["args"]}
                elif isinstance(m, ToolMessage):
                  yield "tool_result", {"name": m.name, "text": m.text[:200]}
      finished = True
    finally:
      if not finished:
        log.info("client left thread %s: run cancelled", thread_id)
    yield "done", {"thread_id": thread_id}

  async def history(self, thread_id: str) -> list[dict]:
    state = await self.agent.aget_state(self.config(thread_id))
    messages = state.values.get("messages", []) if state.values else []
    return [{"role": m.type, "text": m.text} for m in messages if m.type in ("human", "ai") and m.text]
