"""The support agent over HTTP. The calling app (our website's backend, with an API key) says which customer is
signed in; thread ids are namespaced by that customer, so nobody can read another customer's conversation.

  POST /agent/chat            one turn → the reply, this turn's actions, needs_human
  POST /agent/chat/stream     Server-Sent Events: tool, tool_result, token, done
  GET  /agent/threads/{id}    the conversation so far (from the checkpointer)
"""

import uuid
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from nimbus.api.agent_service import AgentService
from nimbus.api.deps import audit

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(audit)])
CustomerId = Annotated[str, Header(alias="X-Customer-ID", pattern=r"^c-\d{4}$", description="The signed-in customer")]


def service(request: Request) -> AgentService:
  return request.app.state.agent_service


ServiceDep = Annotated[AgentService, Depends(service)]


class ChatRequest(BaseModel):
  model_config = {"extra": "forbid"}

  message: str = Field(min_length=1, max_length=2000)
  thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], pattern=r"^[A-Za-z0-9_-]{1,40}$")


class AgentReply(BaseModel):
  reply: str
  thread_id: str
  actions: list[str]
  needs_human: bool


def thread_key(customer_id: str, thread_id: str) -> str:
  return f"{customer_id}:{thread_id}"  # a customer can only ever reach their own threads


@router.post("/chat")
async def chat(body: ChatRequest, customer_id: CustomerId, agent: ServiceDep) -> AgentReply:
  turn = await agent.ask(customer_id, thread_key(customer_id, body.thread_id), body.message)
  return AgentReply(reply=turn.reply, thread_id=body.thread_id, actions=turn.actions, needs_human=turn.needs_human)


@router.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
  body: ChatRequest, customer_id: CustomerId, agent: ServiceDep, request: Request
) -> AsyncIterable[ServerSentEvent]:
  events = agent.stream(customer_id, thread_key(customer_id, body.thread_id), body.message, request.is_disconnected)
  async for event, data in events:
    yield ServerSentEvent(event=event, data={**data, "thread_id": body.thread_id} if event == "done" else data)


@router.get("/threads/{thread_id}")
async def thread(thread_id: str, customer_id: CustomerId, agent: ServiceDep) -> list[dict]:
  return await agent.history(thread_key(customer_id, thread_id))
