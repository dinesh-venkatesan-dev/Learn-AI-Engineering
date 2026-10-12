"""The Nimbus ops MCP server: the warehouse's weekly numbers for staff tools (internal).

weekly_numbers, low_ratings   the same tools the ops agents use (Agents Modules 9–10)
weekly_report                 every café, with progress; a background TASK (MCP tasks extension) the client can
                              start, check on, collect or cancel
review_digest                 summarises a café's low reviews with the CALLER's model (sampling): no model here
"""

import asyncio

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp_tasks import TasksExtension
from mcp.types import (
  CreateMessageRequest,
  CreateMessageRequestParams,
  CreateMessageResult,
  InputRequiredResult,
  SamplingMessage,
  TextContent,
)

from nimbus.ops.data import WEEK, cafe_names, cafe_week, low_reviews

mcp = FastMCP("Nimbus Ops", instructions="Weekly café numbers from the Nimbus warehouse. Internal: staff only.")
mcp.add_extension(TasksExtension())  # in memory, one process; TasksExtension(url="redis://...") + workers to scale


@mcp.tool(tags={"internal"}, annotations={"readOnlyHint": True})
def weekly_numbers(cafe: str, start: str = WEEK[0], end: str = WEEK[1]) -> dict:
  """One café's numbers for a period (default: this week): revenue and its change, orders, rating, sellers."""
  try:
    return cafe_week(cafe, start, end)
  except ValueError as error:
    raise ToolError(str(error)) from error


@mcp.tool(tags={"internal"}, annotations={"readOnlyHint": True}, task=True)
async def weekly_report(ctx: Context) -> list[dict]:
  """Every café's numbers for this week, with progress as each café is done."""
  cafes = cafe_names()
  rows = []
  for done, cafe in enumerate(cafes, start=1):
    rows.append(await asyncio.to_thread(cafe_week, cafe))  # the query runs off the event loop
    await ctx.report_progress(progress=done, total=len(cafes), message=f"{cafe} done")
  return rows


@mcp.tool(tags={"internal"}, annotations={"readOnlyHint": True})
async def review_digest(cafe: str, ctx: Context) -> str | InputRequiredResult:
  """One sentence on what this week's low reviews of a café complain about, written by the caller's model."""
  reviews = low_reviews(cafe)
  if not reviews:
    return f"No reviews rated 2 or lower for {cafe} this week."
  if ctx.input_responses is None:  # round 1: ask the client's model (their provider, their key, their bill)
    text = "\n".join(f"#{r['review_id']} ({r['rating']}★): {r['comment']}" for r in reviews)
    request = CreateMessageRequest(
      method="sampling/createMessage",
      params=CreateMessageRequestParams(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=text))],
        system_prompt="Summarise what these café reviews complain about in one sentence, citing review ids.",
        max_tokens=80,
        temperature=0.2,
      ),
    )
    return InputRequiredResult(result_type="input_required", input_requests={"digest": request})
  answer = ctx.input_responses["digest"]  # round 2: the client's completion
  if isinstance(answer, CreateMessageResult) and isinstance(answer.content, TextContent):
    return answer.content.text
  return "The client did not provide a summary."
