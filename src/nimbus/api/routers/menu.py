"""The café menu, filtered by validated query parameters."""

from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from nimbus.api.deps import StoreDep
from nimbus.support.records import MenuItem
from nimbus.support.tools import SupportStore

router = APIRouter(prefix="/menu", tags=["menu"])


class Category(StrEnum):
  hot = "hot"
  cold = "cold"
  food = "food"


class MenuFilter(BaseModel):
  """Query parameters for /menu, as one reusable, validated model."""

  model_config = {"extra": "forbid"}  # a misspelt parameter (?vegen=true) is an error, not silently ignored

  vegan: bool | None = Field(None, description="Only vegan items (true) or only non-vegan (false)")
  max_price: int | None = Field(None, ge=1, description="Rupees")
  limit: int = Field(10, ge=1, le=20)


@router.get("/{category}")
def menu_category(category: Category, filters: Annotated[MenuFilter, Query()], store: StoreDep) -> list[MenuItem]:
  """Café items of one category (hot, cold or food)."""
  return menu_items(store, category, filters)


@router.get("")
def menu(filters: Annotated[MenuFilter, Query()], store: StoreDep) -> list[MenuItem]:
  """Every café item, filtered."""
  return menu_items(store, None, filters)


def menu_items(store: SupportStore, category: Category | None, filters: MenuFilter) -> list[MenuItem]:
  items = [MenuItem.from_row(row) for row in store.menu.values()]
  items = [
    i
    for i in items
    if (category is None or i.category == category)
    and (filters.vegan is None or i.vegan == filters.vegan)
    and (filters.max_price is None or i.price_rupees <= filters.max_price)
  ]
  return items[: filters.limit]
