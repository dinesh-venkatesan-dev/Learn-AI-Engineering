"""Records Nimbus shares with the outside world (the MCP server, the HTTP API): only the fields a client may see."""

from pydantic import BaseModel, Field


class OrderLine(BaseModel):
  name: str
  qty: int
  price: int


class Order(BaseModel):
  order_id: str
  status: str
  ordered_on: str
  delivered_on: str | None
  items: list[OrderLine]
  total: int = Field(description="Rupees")


class MenuItem(BaseModel):
  item: str
  category: str
  price_rupees: int
  vegan: bool

  @classmethod
  def from_row(cls, row: dict) -> "MenuItem":
    return cls(
      item=row["item"], category=row["category"], price_rupees=int(row["price_rupees"]), vegan=row["vegan"] == "yes"
    )
