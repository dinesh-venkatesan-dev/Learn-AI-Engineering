"""The Nimbus assistant as ONE MCP tool: any MCP host (another team's agent, an IDE) can ask it anything.

  ask_nimbus(question, customer_id)   the router assistant (Agents Module 7) answers

The customer id is an argument here; MCP Module 4 takes it from the caller's token instead.
"""

from functools import lru_cache

from fastmcp import FastMCP

mcp = FastMCP(
  "Nimbus Assistant", instructions="Ask Nimbus Coffee's assistant about orders, policies, cafés and the menu."
)


@lru_cache
def _assistant():
  from nimbus.assistant.architectures import build_router
  from nimbus.assistant.specialists import build_specialists
  from nimbus.config import get_settings
  from nimbus.llm import get_chat_model
  from nimbus.rag.ingestion.embeddings import get_embeddings
  from nimbus.rag.retriever import Retriever

  settings, model = get_settings(), get_chat_model()
  retriever = Retriever(settings, get_embeddings(settings), model)
  return build_router(model, build_specialists(settings, model, retriever))[1]


@mcp.tool(annotations={"readOnlyHint": True})
async def ask_nimbus(question: str, customer_id: str = "c-1042") -> str:
  """Ask Nimbus Coffee's assistant: orders, deliveries, subscriptions, refund rules, cafés, the café menu."""
  import asyncio

  return (await asyncio.to_thread(_assistant(), customer_id, question)).reply  # the assistant is synchronous
