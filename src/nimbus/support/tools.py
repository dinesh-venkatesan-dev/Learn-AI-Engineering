"""Support tools: what the support model can look up and do.

Each tool is a Python function with a schema the model reads (name, description, argument types and descriptions).
The model only *asks* for a tool; this code runs it. Failures come back as messages the model can act on
(ToolException), full records stay in the app as artifacts, and escalation ends the conversation directly.
"""

import csv
import json
from datetime import date
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import BaseModel, Field

from nimbus.config import PROJECT_DIR

REFUND_WINDOW_DAYS = 14  # from refunds_and_returns.html and products.txt: unopened bags, within 14 days


class SupportStore:
  """The order system and the café menu (JSON and CSV files here; a database and an API in production)."""

  def __init__(
    self, orders: Path = PROJECT_DIR / "data/support/orders.json", menu: Path = PROJECT_DIR / "data/nimbus/menu.csv"
  ):
    data = json.loads(orders.read_text())
    self.today = date.fromisoformat(data["today"])  # the dataset's own clock, so policy answers are reproducible
    self.orders = {o["order_id"]: o for o in data["orders"]}
    self.customers: dict[str, str] = data.get("customers", {})
    self.subscriptions = {sub["customer_id"]: dict(sub) for sub in data.get("subscriptions", [])}  # mutable copies
    with menu.open(newline="") as f:
      self.menu = {row["item"].casefold(): row for row in csv.DictReader(f)}

  def order(self, order_id: str) -> dict | None:
    return self.orders.get(order_id.strip().upper())

  def orders_of(self, customer_id: str) -> list[dict]:
    return sorted((o for o in self.orders.values() if o["customer_id"] == customer_id), key=lambda o: o["ordered_on"])


class OrderId(BaseModel):
  order_id: str = Field(pattern=r"^NC-\d{4,6}$", description="The order number exactly as written, like NC-12093")


class RefundCheck(BaseModel):
  order_id: str = Field(pattern=r"^NC-\d{4,6}$", description="The order number, like NC-12093")
  opened: bool = Field(description="Whether the customer says the bag was opened; ask them if you don't know")


class Escalation(BaseModel):
  reason: str = Field(description="One sentence for the support team: what happened and what the customer wants")
  urgency: Literal["normal", "high", "urgent"] = Field(description="urgent for health, safety or legal threats")


def explain_bad_arguments(error: Exception) -> str:
  """What the model sees when its arguments break the schema: the field and the rule, so it can fix the call."""
  details = getattr(error, "errors", lambda: [])()
  if not details:
    return f"Invalid arguments: {error}"
  return "Invalid arguments: " + "; ".join(f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in details)


def build_tools(store: SupportStore) -> list[BaseTool]:
  @tool(args_schema=OrderId, response_format="content_and_artifact")
  def get_order(order_id: str) -> tuple[str, dict]:
    """Look up a Nimbus order: status, items, total, and order and delivery dates."""
    order = store.order(order_id)
    if order is None:
      raise ToolException(f"No order {order_id}. Ask the customer to check the number (it looks like NC-12093).")
    items = ", ".join(f"{i['qty']} × {i['name']}" for i in order["items"])
    delivered = f", delivered {order['delivered_on']}" if order["delivered_on"] else ""
    summary = (
      f"{order['order_id']}: {order['status']}, ordered {order['ordered_on']}{delivered}; {items}; ₹{order['total']}"
    )
    return summary, order  # the model reads the summary; the app keeps the full record

  @tool(args_schema=RefundCheck)
  def check_refund(order_id: str, opened: bool) -> str:
    """Check whether an order can be refunded under Nimbus policy. Call this before promising any refund."""
    order = store.order(order_id)
    if order is None:
      raise ToolException(f"No order {order_id}.")
    if order["status"] in ("processing", "packed"):
      return f"{order_id} hasn't shipped yet: it can be cancelled instead (free until it is packed)."
    if order["status"] != "delivered":
      return f"{order_id} is {order['status']}: refunds apply to delivered orders."
    if opened:
      return f"{order_id}: opened bags can't be returned. Offer to escalate if the beans were faulty."
    days = (store.today - date.fromisoformat(order["delivered_on"])).days
    if days > REFUND_WINDOW_DAYS:
      return f"{order_id}: delivered {days} days ago, outside the {REFUND_WINDOW_DAYS}-day window. Not refundable."
    return (
      f"{order_id}: eligible. Unopened, delivered {days} days ago (window {REFUND_WINDOW_DAYS} days). "
      f"₹{order['total']} goes back to the original payment method in 5-7 business days."
    )

  @tool
  def cafe_menu_item(name: str) -> dict:
    """Price in rupees and whether a café drink or snack is vegan. name: the item, e.g. 'Oat Latte'."""
    row = store.menu.get(name.strip().casefold())
    if row is None:
      raise ToolException(
        f"'{name}' isn't on the café menu. Items: {', '.join(r['item'] for r in store.menu.values())}."
      )
    return {"item": row["item"], "price_rupees": int(row["price_rupees"]), "vegan": row["vegan"] == "yes"}  # → JSON

  @tool(
    "loyalty_points",
    description="Nimbus Beans points earned (1 point per ₹10). For an order, pass order_id and the total is looked "
    "up for you; for any other amount, pass amount_rupees.",
  )
  def points_for(order_id: str | None = None, amount_rupees: int | None = None) -> str:
    # Taking the order id (instead of an amount the model must first look up) removes a dependent step: in the
    # scorecard the model called this in parallel with get_order and guessed the amount; descriptions didn't fix it.
    if order_id:
      order = store.order(order_id)
      if order is None:
        raise ToolException(f"No order {order_id}.")
      return f"{order['order_id']} (₹{order['total']}) earns {order['total'] // 10} points."
    if amount_rupees is None:
      raise ToolException("Pass order_id or amount_rupees.")
    return f"₹{amount_rupees} earns {amount_rupees // 10} points."

  @tool(args_schema=Escalation, return_direct=True)
  def escalate_to_human(reason: str, urgency: str) -> str:
    """Hand the conversation to a person. Use for safety issues, legal threats, refunds of opened faulty bags, or
    anything you can't resolve with the other tools."""
    hours = {"urgent": "1 hour", "high": "4 hours", "normal": "1 business day"}[urgency]
    return f"I've passed this to our support team ({urgency}). Someone will contact you within {hours}."

  for t in (get_order, check_refund, cafe_menu_item, points_for):
    t.handle_tool_error = True  # a ToolException becomes a message for the model, not a crash
  for t in (get_order, check_refund, escalate_to_human):
    t.handle_validation_error = explain_bad_arguments  # so do arguments that break the schema ("12093")
  return [get_order, check_refund, cafe_menu_item, points_for, escalate_to_human]
