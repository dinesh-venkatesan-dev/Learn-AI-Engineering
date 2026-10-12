"""Orders: one order by number, with only the fields a client may see."""

from typing import Annotated

from fastapi import APIRouter, Path

from nimbus.api.deps import StoreDep
from nimbus.api.errors import NotFound
from nimbus.support.records import Order

router = APIRouter(prefix="/orders", tags=["orders"])
OrderId = Annotated[str, Path(pattern=r"^NC-\d{4,6}$", description="Like NC-12093", examples=["NC-12093"])]


@router.get("/{order_id}")
def get_order(order_id: OrderId, store: StoreDep) -> Order:
  """One order: status, items, total and dates."""
  order = store.order(order_id)
  if order is None:
    raise NotFound(f"No order {order_id}")
  return Order.model_validate(order)  # the return type filters: customer ids never leave the service
