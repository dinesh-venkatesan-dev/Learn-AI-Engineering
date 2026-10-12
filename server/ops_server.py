"""The ops deep agent for LangGraph Server (Agents Module 10): the server provides the checkpointer and the store,
so the agent is built without them; its files live in each thread's state on the server."""

from deepagents import create_deep_agent

from nimbus.llm import get_chat_model
from nimbus.ops.agents import SALES_ANALYST, VOICE_OF_CUSTOMER, list_cafes, weekly_numbers

graph = create_deep_agent(
  model=get_chat_model(),
  tools=[list_cafes, weekly_numbers],
  subagents=[SALES_ANALYST, VOICE_OF_CUSTOMER],
  system_prompt=(
    "You are Nimbus's operations assistant. Use the tools for every number; keep notes in /notes/. Be concise."
  ),
)
