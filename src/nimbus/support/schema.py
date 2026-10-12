"""What triage produces: a validated object, not text. Field descriptions are sent to the model as the schema."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
  "order_problem",  # damaged, late, missing or wrong item
  "refund_return",
  "cancel_order",
  "subscription",
  "product_question",
  "cafe_experience",
  "allergy_safety",
  "other",
]
Urgency = Literal["low", "normal", "high", "urgent"]


class ProductMention(BaseModel):
  name: str = Field(description="Product as the customer wrote it, e.g. 'Nilgiri Sunrise' or 'cold brew'")
  quantity: int | None = Field(None, ge=1, description="How many, if the customer says")


class TicketTriage(BaseModel):
  """How to route one customer support message."""

  category: Category
  urgency: Urgency = Field(
    description="urgent: health or safety risk, or a legal threat · high: money taken but nothing received, damaged "
    "or wrong delivery · normal: a request that needs action · low: a question or feedback"
  )
  order_id: str | None = Field(
    None, pattern=r"^NC-\d{4,6}$", description="Order number exactly as written, like NC-10432; null if none"
  )
  language: str = Field(description="ISO 639-1 code of the customer's language, e.g. en, hi, ta")
  sentiment: Literal["positive", "neutral", "negative"]
  summary: str = Field(max_length=200, description="One English sentence for the support agent")
  needs_human: bool = Field(
    description="true for refunds and returns, allergy or safety issues, legal threats, and urgent tickets"
  )
  products: list[ProductMention] = Field(default_factory=list, description="Every product the customer mentions")


class Ticket(BaseModel):
  id: str
  text: str
  attachment: Path | None = None  # a photo of a receipt, a damaged bag...
