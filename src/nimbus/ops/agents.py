"""The weekly ops review, written by an agent from the warehouse's numbers.

  ops_tools()        cafe_names, weekly_numbers, low_ratings: every number computed in code (nimbus/ops/data.py)
  plain_agent()      create_agent with those tools: one context does everything
  deep_agent()       create_deep_agent: a todo plan, the review written to /review.md, and two subagents with clean
                     contexts (sales-analyst for one café's numbers, voice-of-customer for one café's complaints)

Both get the same tools and the same request; the evaluation checks the review against ground_truth().
"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import ToolException, tool

from nimbus.ops.data import COVERED, WEEK, cafe_names, cafe_week, low_reviews

REQUEST = (
  "Write this week's ops review ({start} to {end}) for every Nimbus café: revenue and its change on the week before, "
  "the best seller, the average rating, and any reviews rated 2 or lower (with their ids). End with the two cafés "
  "that need attention and why."
)
LIMITS = {"recursion_limit": 80}  # small models can loop on tools: always cap a run


@tool("cafe_names")
def list_cafes() -> list[str]:
  """Every Nimbus café's name."""
  return cafe_names()


def check_dates(start: str, end: str) -> None:
  """Measured: asked about "this week" with no date, the model invented November 2023, got ₹0 everywhere and kept
  re-querying until the recursion limit. A clear error tells it what the data covers."""
  if not (COVERED[0] <= start <= end <= COVERED[1]):
    raise ToolException(
      f"The warehouse covers {COVERED[0]} to {COVERED[1]}. 'This week' means {WEEK[0]} to {WEEK[1]}; omit the "
      "dates to use it."
    )


@tool
def weekly_numbers(cafe: str, start: str = WEEK[0], end: str = WEEK[1]) -> dict:
  """One café's numbers for a period (ISO days, inclusive; leave the dates out for this week, 2026-09-24 to
  2026-09-30): revenue in rupees and its change on the period before, completed and refunded orders, average rating,
  best and weakest seller. Quote these numbers exactly."""
  check_dates(start, end)
  try:
    return cafe_week(cafe, start, end)
  except ValueError as error:  # an unknown café: a message the model can correct, not a crash
    raise ToolException(str(error)) from error


def cafe_low_ratings(cafe: str, start: str = WEEK[0], end: str = WEEK[1]) -> list[dict] | str:
  """One café's reviews rated 2 or lower in a period (leave the dates out for this week), with review ids and comments
  (evidence for complaints)."""
  check_dates(start, end)
  try:
    reviews = low_reviews(cafe, start, end)
  except ValueError as error:
    raise ToolException(str(error)) from error
  # measured: an empty list came back as three invented "low reviews"; say plainly that there are none
  return reviews or f"No reviews rated 2 or lower for {cafe} in this period."


low_ratings = tool("low_ratings")(cafe_low_ratings)  # the tool the agents see; the function above is its code


for _tool in (list_cafes, weekly_numbers, low_ratings):
  _tool.handle_tool_error = True  # a ToolException becomes a message for the model, wherever the tool is used


def ops_tools() -> list:
  return [list_cafes, weekly_numbers, low_ratings]


PLAIN_PROMPT = (
  "You are Nimbus Coffee's operations analyst. Use the tools for every number and quote them exactly; never compute "
  "or guess numbers yourself. Write the review in Markdown, one section per café."
)
DEEP_PROMPT = (
  "You are Nimbus Coffee's operations lead. Plan with write_todos first. Get the café list, then delegate: "
  "sales-analyst once per café for its numbers, voice-of-customer once per café for complaints (call several task "
  "tools in one turn when you can). Write the finished review in Markdown to /review.md with write_file, one section "
  "per café, quoting numbers exactly as the analysts report them, then reply with one sentence saying it's done."
)
SALES_ANALYST = {
  "name": "sales-analyst",
  "description": "Gets ONE café's weekly numbers (revenue, change, orders, refunds, rating, best and weakest seller). "
  "Give it the café name and the dates.",
  "system_prompt": "Call weekly_numbers once and reply with its numbers exactly as given, as a short list. Never "
  "round or compute anything.",
  "tools": [weekly_numbers],
}
VOICE_OF_CUSTOMER = {
  "name": "voice-of-customer",
  "description": "Lists ONE café's reviews rated 2 or lower in a period, with review ids. Give it the café name and "
  "the dates.",
  "system_prompt": "Call low_ratings once. Reply with each review's id, rating and comment, or 'none'.",
  "tools": [low_ratings],
}


def plain_agent(model: BaseChatModel):
  return create_agent(model, ops_tools(), system_prompt=PLAIN_PROMPT)


def deep_agent(model: BaseChatModel, checkpointer=None, **options):
  from deepagents import create_deep_agent
  from langchain.agents.middleware import TodoListMiddleware

  return create_deep_agent(
    model=model,  # always pass it: without one, deepagents falls back to an Anthropic model
    tools=[list_cafes],
    system_prompt=DEEP_PROMPT,
    subagents=[SALES_ANALYST, VOICE_OF_CUSTOMER],
    middleware=[TodoListMiddleware()],  # planning is opt-in since deepagents 0.7
    checkpointer=checkpointer,
    **options,
  )


def review_text(result: dict) -> str:
  """The review: /review.md if the agent wrote it, otherwise its final message."""
  files = result.get("files") or {}
  review = files.get("/review.md")
  if review:
    content = review.get("content") if isinstance(review, dict) else review
    return "\n".join(content) if isinstance(content, list) else str(content)
  return result["messages"][-1].text
