"""The Nimbus account server: the customer-facing MCP server, behind sign-in (MCP Module 4).

Every HTTP request carries a JWT (nimbus.mcp.auth). Who the caller is comes from the token, so a customer's agent can
only ever see that customer's orders, whatever the model puts in the arguments.

  whoami, my_orders           the caller's identity and orders
  get_order, check_refund     the caller's own orders (staff with orders:read_all: any order)
  cancel_order                one of the caller's orders that hasn't been packed; destructive
  issue_refund                orders:refund scope only; destructive
  weekly_sales                managers only (the "manager" role)
  remember_note, my_notes     per-user notes kept on the server (UserSession), across connections
Resources and prompts: the refund policy and the support_reply template, the same as the support server.

Run it:  NIMBUS_JWT_PUBLIC_KEY="$(cat key.pub)" uv run python src/nimbus/mcp/account.py --http 8766
"""

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AuthProvider, require_roles, require_scopes
from fastmcp.server.sessions import UserSession
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from nimbus.mcp import server as support
from nimbus.mcp.auth import caller, caller_scopes, roles
from nimbus.ops.data import cafe_names, cafe_week
from nimbus.support.records import Order

OrderId = Annotated[str, Field(pattern=r"^NC-\d{4,6}$", description="The order number, like NC-12093")]


def _visible_order(order_id: str) -> dict:
  """The order if the caller may see it. Someone else's order looks exactly like a missing one: no leak."""
  found = support.store.order(order_id)
  if found is None or (found["customer_id"] != caller() and "orders:read_all" not in caller_scopes()):
    raise ToolError(f"No order {order_id} on your account.")
  return found


def build_account_server(auth: AuthProvider | None) -> FastMCP:
  """auth=None only for in-memory tests of the tools themselves; over HTTP it is always a verifier."""
  mcp = FastMCP(
    "Nimbus Account",
    instructions="The signed-in customer's Nimbus orders, refunds and notes. Order numbers look like NC-12093.",
    auth=auth,
  )

  @mcp.tool(annotations={"readOnlyHint": True})
  def whoami() -> dict:
    """Who you are signed in as, and what you may do."""
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    return {"user": token.claims["sub"], "scopes": token.scopes, "roles": roles(token.claims)}

  @mcp.tool(annotations={"readOnlyHint": True})
  def my_orders() -> list[Order]:
    """Your orders, oldest first."""
    return [Order.model_validate(o) for o in support.store.orders_of(caller())]

  @mcp.tool(annotations={"readOnlyHint": True})
  def get_order(order_id: OrderId) -> Order:
    """One of your orders: status, items, total, and order and delivery dates."""
    return Order.model_validate(_visible_order(order_id))

  @mcp.tool(annotations={"readOnlyHint": True})
  def check_refund(order_id: OrderId, opened: Annotated[bool, Field(description="Was the bag opened?")]) -> str:
    """Whether one of your orders can be refunded under Nimbus policy."""
    _visible_order(order_id)
    return support._run("check_refund", {"order_id": order_id, "opened": opened})

  @mcp.tool(annotations={"destructiveHint": True, "idempotentHint": False})
  def cancel_order(order_id: OrderId) -> str:
    """Cancel one of your orders that hasn't been packed yet."""
    found = _visible_order(order_id)
    if found["status"] != "processing":
      raise ToolError(f"{order_id} is {found['status']}: only orders that haven't been packed can be cancelled.")
    found["status"] = "cancelled"
    return f"Cancelled {order_id}. ₹{found['total']} goes back to the original payment method in 5 to 7 business days."

  @mcp.tool(auth=require_scopes("orders:refund"), annotations={"destructiveHint": True, "idempotentHint": True})
  def issue_refund(order_id: OrderId, reason: Annotated[str, Field(description="One sentence")]) -> str:
    """Refund an order in full (staff with the orders:refund scope)."""
    found = _visible_order(order_id)
    if found.get("refunded"):
      return f"{order_id} was already refunded."
    found["refunded"] = True
    return f"{caller()} refunded ₹{found['total']} on {order_id}: {reason}"

  @mcp.tool(auth=require_roles("manager", extract=roles), annotations={"readOnlyHint": True})
  def weekly_sales() -> dict:
    """This week's revenue for every café (managers only)."""
    return {cafe: cafe_week(cafe)["revenue_rupees"] for cafe in cafe_names()}

  @mcp.tool
  async def remember_note(
    note: Annotated[str, Field(description="e.g. 'oat milk, no sugar'")], session: UserSession
  ) -> str:
    """Save a note about your preferences, kept for your next conversations."""
    notes = await session.get("notes", default=[])
    await session.set("notes", [*notes, note])
    return f"Saved: {note}"

  @mcp.tool(annotations={"readOnlyHint": True})
  async def my_notes(session: UserSession) -> list[str]:
    """Your saved preference notes."""
    return await session.get("notes", default=[])

  mcp.resource("nimbus://policies/refunds", mime_type="text/plain")(support.refund_policy)
  mcp.prompt(support.support_reply)

  @mcp.custom_route("/health", methods=["GET"])
  async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})

  return mcp


if __name__ == "__main__":
  import sys

  from nimbus.mcp.auth import verifier

  port = int(sys.argv[sys.argv.index("--http") + 1])
  build_account_server(verifier()).run(transport="http", host="127.0.0.1", port=port, show_banner=False)
