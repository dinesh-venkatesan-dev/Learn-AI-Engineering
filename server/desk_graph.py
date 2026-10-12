"""The support desk for LangGraph Server (`uv run langgraph dev` in this folder).

The server brings its own checkpointer (threads) and queue; the graph is compiled without one. Callers pass the
logged-in customer as the run's context: {"customer_id": "c-6060", "name": "Farah Z."}.
"""

from nimbus.llm import get_chat_model
from nimbus.support.desk import build_support_desk
from nimbus.support.tools import SupportStore

graph = build_support_desk(
  get_chat_model(),
  SupportStore(),
  refunds=True,
  review=True,  # refunds and subscription pauses wait for approval: resume with Command(resume=...)
  fallback=get_chat_model(role="fallback"),
  retries=2,
)
