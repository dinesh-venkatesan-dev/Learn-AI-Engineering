"""The Nimbus support MCP server: the support desk's lookups, for any MCP host (agents, IDEs, desktop apps).

  get_order, cafe_menu_item    typed results (Order, MenuItem)
  check_refund, loyalty_points the refund policy and the points rule, from the same code the agents use
  escalate_to_human            hands the conversation to a person
  cancel_order                 asks the USER to confirm first (elicitation), then cancels; destructive
Resources (the APP reads them): the refund policy, the café menu, one order by id, the public documents.
Prompts (the USER picks them): a support reply in the house style; completions suggest order ids and menu items.

Run it on its own:  uv run fastmcp run src/nimbus/mcp/server.py        (stdio, for a host that starts it)
                    uv run fastmcp list src/nimbus/mcp/server.py       (what a client sees)
                    uv run python src/nimbus/mcp/server.py --http 8765 (a shared server at /mcp, health at /health)
"""

import html
import re
from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from langchain_core.tools import ToolException
from mcp.types import (
  ElicitRequest,
  ElicitRequestFormParams,
  InputRequiredResult,
  PromptReference,
  ResourceTemplateReference,
)
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from nimbus.config import PROJECT_DIR
from nimbus.support.records import MenuItem, Order
from nimbus.support.tools import SupportStore, build_tools

store = SupportStore()
_lookups = {t.name: t for t in build_tools(store)}  # one source of truth: the same tools the agents use


def _run(name: str, args: dict) -> str:
  """Call one of the agents' tools; its ToolException becomes an MCP ToolError (a message the client's model reads)."""
  try:
    return str(_lookups[name].func(**args))
  except ToolException as error:
    raise ToolError(str(error)) from error


mcp = FastMCP(
  "Nimbus Support",
  instructions="Look up Nimbus Coffee orders (numbers look like NC-12093) and café menu items. Prices are in rupees.",
)


@mcp.tool(annotations={"readOnlyHint": True})
def get_order(
  order_id: Annotated[str, Field(pattern=r"^NC-\d{4,6}$", description="The order number, like NC-12093")],
) -> Order:
  """Look up a Nimbus order: status, items, total, and order and delivery dates."""
  order = store.order(order_id)
  if order is None:
    raise ToolError(f"No order {order_id}. Ask the customer to check the number.")  # the client sees this text
  return Order.model_validate(order)  # extra fields (customer id, opened) stay out of the reply


@mcp.tool(annotations={"readOnlyHint": True})
def cafe_menu_item(name: Annotated[str, Field(description="The item, e.g. 'Oat Latte'")]) -> MenuItem:
  """Price in rupees and whether a café drink or snack is vegan."""
  row = store.menu.get(name.strip().casefold())
  if row is None:
    raise ToolError(f"'{name}' isn't on the café menu.")
  return MenuItem.from_row(row)


@mcp.tool(annotations={"readOnlyHint": True})
def check_refund(
  order_id: Annotated[str, Field(pattern=r"^NC-\d{4,6}$", description="The order number, like NC-12093")],
  opened: Annotated[bool, Field(description="Whether the customer opened the bag")],
) -> str:
  """Check whether an order can be refunded under Nimbus policy. Call this before promising any refund."""
  return _run("check_refund", {"order_id": order_id, "opened": opened})


@mcp.tool(annotations={"readOnlyHint": True})
def loyalty_points(order_id: str | None = None, amount_rupees: int | None = None) -> str:
  """Nimbus Beans points earned (1 point per ₹10). For an order pass order_id (its total is looked up); for any other
  amount pass amount_rupees."""
  return _run("loyalty_points", {"order_id": order_id, "amount_rupees": amount_rupees})


@mcp.tool
def escalate_to_human(
  reason: Annotated[str, Field(description="Why a person is needed")],
  urgency: Annotated[str, Field(pattern="^(normal|high|urgent)$", description="normal, high or urgent")],
) -> str:
  """Hand the conversation to a person: safety issues, legal threats, refunds of opened faulty bags, or anything the
  other tools can't resolve."""
  return _run("escalate_to_human", {"reason": reason, "urgency": urgency})


# ── resources: what an application puts in context ─────────────────────────────
DOCS = PROJECT_DIR / "data" / "nimbus"
PUBLIC_DOCS = {"faq.html", "products.txt", "refunds_and_returns.html", "subscriptions.txt", "privacy_notice.md"}


def page_text(path) -> str:
  """An HTML page's readable text (the documents are the website's own pages)."""
  page = re.sub(r"<(nav|footer|header)\b.*?</\1>", " ", path.read_text(), flags=re.DOTALL | re.IGNORECASE)
  text = re.sub(r"<[^>]+>", " ", page)
  return re.sub(r"\s+", " ", html.unescape(text)).strip()


@mcp.resource("nimbus://policies/refunds", mime_type="text/plain")
def refund_policy() -> str:
  """Refunds, returns, cancellations and café purchases: the policy customers see on the website."""
  return page_text(DOCS / "refunds_and_returns.html")


@mcp.resource("nimbus://menu", mime_type="application/json")
def menu() -> dict:
  """The café menu: item → price in rupees and whether it's vegan."""
  return {
    row["item"]: {"price_rupees": int(row["price_rupees"]), "vegan": row["vegan"] == "yes"}
    for row in store.menu.values()
  }


@mcp.resource("nimbus://orders/{order_id}{?fields}")
def order_resource(order_id: str, fields: str = "") -> dict:
  """One order by id. ?fields=status,total returns only some fields."""
  found = store.order(order_id)
  if found is None:
    raise ResourceError(f"No order {order_id}")
  wanted = [f for f in fields.split(",") if f]
  public = Order.model_validate(found).model_dump()
  return {k: v for k, v in public.items() if not wanted or k in wanted}


@mcp.resource("docs://{name}")
def document(name: str) -> str:
  """A public Nimbus document by file name (faq.html, products.txt, ...). Staff documents are not served."""
  if name not in PUBLIC_DOCS:
    raise ResourceError(f"No public document {name!r}. Public documents: {', '.join(sorted(PUBLIC_DOCS))}.")
  path = DOCS / name
  return page_text(path) if path.suffix == ".html" else path.read_text()


# ── prompts: templates the user picks ──────────────────────────────────────────
@mcp.prompt
def support_reply(issue: str, tone: Literal["warm", "formal"] = "warm", max_sentences: int = 3) -> str:
  """Draft a reply to a customer, following the Nimbus refund policy."""
  return (
    f"Write a {tone} reply in at most {max_sentences} sentences from Nimbus Coffee support. The customer's issue: "
    f"{issue}. Follow this policy: {refund_policy()} Don't assume anything about the order that isn't stated; ask "
    "for the order number if you need it. Reply with the message only."
  )


@mcp.completion
def complete(ref, argument, context):
  """Suggestions while the user types: order ids for nimbus://orders/{order_id}, items for the menu."""
  if isinstance(ref, ResourceTemplateReference) and argument.name == "order_id":
    return [o for o in store.orders if o.startswith(argument.value.upper())]
  if isinstance(ref, ResourceTemplateReference) and argument.name == "name":
    return [d for d in sorted(PUBLIC_DOCS) if d.startswith(argument.value.lower())]
  if isinstance(ref, PromptReference) and argument.name == "issue":
    return [i for i in ("damaged bag", "late delivery", "wrong item", "refund") if i.startswith(argument.value.lower())]
  return None


# ── a destructive tool that asks the user first ────────────────────────────────
MODERN = "2026-07-28"


@mcp.tool(annotations={"destructiveHint": True, "idempotentHint": False})
async def cancel_order(
  order_id: Annotated[str, Field(pattern=r"^NC-\d{4,6}$", description="The order number, like NC-12093")],
  ctx: Context,
) -> str | InputRequiredResult:
  """Cancel an order that hasn't been packed. The customer confirms first."""
  found = store.order(order_id)
  if found is None:
    raise ToolError(f"No order {order_id}.")
  if found["status"] != "processing":
    raise ToolError(f"{order_id} is {found['status']}: only orders that haven't been packed can be cancelled.")
  question = f"Really cancel {order_id} (₹{found['total']})?"
  if ctx.request_context.protocol_version == MODERN:  # modern: the tool is a guard, no state held between rounds
    if ctx.input_responses is None:
      form = ElicitRequestFormParams(
        message=question,
        requested_schema={"type": "object", "properties": {"confirm": {"type": "boolean"}}, "required": ["confirm"]},
      )
      return InputRequiredResult(
        result_type="input_required",
        input_requests={"confirm": ElicitRequest(method="elicitation/create", params=form)},
      )
    answer = ctx.input_responses["confirm"]
    confirmed = answer.action == "accept" and answer.content["confirm"]
  else:  # legacy: pause mid-tool over the session's back-channel
    answer = await ctx.elicit(question, response_type=bool)
    confirmed = answer.action == "accept" and answer.data
  if not confirmed:
    return f"{order_id} was NOT cancelled."
  found["status"] = "cancelled"
  return f"Cancelled {order_id}. ₹{found['total']} goes back to the original payment method in 5 to 7 business days."


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
  return JSONResponse({"status": "ok"})


if __name__ == "__main__":
  import sys

  if "--http" in sys.argv:  # a shared web service: many clients, one server
    port = int(sys.argv[sys.argv.index("--http") + 1])
    mcp.run(transport="http", host="127.0.0.1", port=port, show_banner=False)
  else:
    mcp.run()  # stdio by default: the host starts this file and talks over stdin/stdout
