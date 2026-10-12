"""A slow, thorough researcher for async subagents (Agents Module 9): a quarter's trend for every café.

Served by `langgraph dev` from this folder; the ops agent starts it with start_async_task and checks on it later.
"""

from langchain.agents import create_agent

from nimbus.llm import get_chat_model
from nimbus.ops.agents import list_cafes, weekly_numbers

researcher = create_agent(
  get_chat_model(),
  [list_cafes, weekly_numbers],
  system_prompt=(
    "You research Nimbus cafés' quarter (2026-07-01 to 2026-09-30). For each café call weekly_numbers with "
    "start 2026-07-01 and end 2026-09-30, then report each café's quarter revenue exactly as given and the top "
    "and bottom café. Be brief."
  ),
  name="researcher",
)
