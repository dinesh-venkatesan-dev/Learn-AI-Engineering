"""Prompts are versioned like code: a change gets a new version, and the scorecard decides which one ships.

v1  zero-shot: role, categories and the schema
v2  v1 + routing rules + three worked examples (few-shot)
"""

import json

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

from nimbus.support.schema import Category

CATEGORIES = ", ".join(Category.__args__)

SYSTEM_V1 = (
  "You triage customer messages for Nimbus Coffee Co., an Indian coffee roaster with cafés and bean subscriptions. "
  "Classify each message into one of these categories: {categories}."
)

RULES = """Rules:
- Damaged, late, missing or wrong deliveries are order_problem, even if the customer also wants money back.
- Asking for money back on an undamaged order, or returning a bag, is refund_return.
- Stopping an order before it ships is cancel_order; pausing, skipping or switching a subscription is subscription.
- Anything about allergens, contamination or feeling unwell is allergy_safety and urgent.
- A threat of legal action, a consumer court or a chargeback is urgent.
- Only copy an order id that is literally in the message (format NC- followed by digits)."""

EXAMPLES = [
  {
    "ticket": "My Nilgiri Sunrise bag (order NC-20417) arrived torn and half the beans spilled. Photo attached.",
    "triage": {
      "category": "order_problem",
      "urgency": "high",
      "order_id": "NC-20417",
      "language": "en",
      "sentiment": "negative",
      "summary": "Bag of Nilgiri Sunrise arrived torn with beans spilled.",
      "needs_human": False,
      "products": [{"name": "Nilgiri Sunrise", "quantity": 1}],
    },
  },
  {
    "ticket": "Kya main apna subscription do hafte ke liye pause kar sakti hoon? Travel kar rahi hoon.",
    "triage": {
      "category": "subscription",
      "urgency": "normal",
      "order_id": None,
      "language": "hi",
      "sentiment": "neutral",
      "summary": "Customer wants to pause her subscription for two weeks while travelling.",
      "needs_human": False,
      "products": [],
    },
  },
  {
    "ticket": "The hazelnut syrup in my latte today wasn't mentioned anywhere and I have a nut allergy. "
    "My lips are swelling.",
    "triage": {
      "category": "allergy_safety",
      "urgency": "urgent",
      "order_id": None,
      "language": "en",
      "sentiment": "negative",
      "summary": "Customer with a nut allergy reacted to unlisted hazelnut syrup in a café latte.",
      "needs_human": True,
      "products": [{"name": "latte", "quantity": 1}],
    },
  },
]

FEW_SHOT = FewShotChatMessagePromptTemplate(
  examples=[{"ticket": e["ticket"], "triage": json.dumps(e["triage"], ensure_ascii=False)} for e in EXAMPLES],
  example_prompt=ChatPromptTemplate.from_messages([("human", "{ticket}"), ("ai", "{triage}")]),
)

TRIAGE_PROMPTS: dict[str, ChatPromptTemplate] = {
  "v1": ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{ticket}")]).partial(categories=CATEGORIES),
  "v2": ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V1 + "\n\n" + RULES), FEW_SHOT, ("human", "{ticket}")]
  ).partial(categories=CATEGORIES),
}

REPLY_PROMPT = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "You are Nimbus Coffee Co.'s support assistant. Write a short, warm reply (at most 80 words) in the customer's "
      "language ({language}). Next step for this kind of request: {next_step} Never promise refunds or "
      "compensation; a teammate confirms those.",
    ),
    MessagesPlaceholder("history"),  # earlier messages in this conversation, if any
    ("human", "{ticket}"),
  ]
)

NEXT_STEPS = {
  "order_problem": "ask for a photo within 3 days of delivery; a replacement is sent once the photo is checked.",
  "refund_return": "unopened bags can be returned within 14 days; a teammate will confirm the refund.",
  "cancel_order": "orders can be cancelled free until they are packed; packed orders must be returned instead.",
  "subscription": "subscriptions can be paused, skipped or switched anytime from the account page.",
  "product_question": "answer briefly and point to the shop page for details.",
  "cafe_experience": "thank them; café drinks are not refundable but the café will happily remake a drink.",
  "allergy_safety": "tell them a teammate will call them right away and to seek medical help if they feel unwell.",
  "other": "thank them and say a teammate will follow up.",
}
