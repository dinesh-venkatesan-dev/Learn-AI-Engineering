from nimbus.support.graph import order_note, reply_problems
from nimbus.support.schema import TicketTriage

ORDER = {"order_id": "NC-31877", "status": "delivered", "ordered_on": "2026-03-02", "delivered_on": "2026-03-05"}


def triage(language="en", category="order_problem"):
  return TicketTriage(
    category=category,
    urgency="high",
    order_id=None,
    language=language,
    sentiment="negative",
    needs_human=False,
    summary="x",
  )


def test_a_reply_must_state_the_real_order_status():
  assert reply_problems("Your order NC-31877 was delivered on 5 March.", triage(), [{"record": ORDER}]) == []
  problems = reply_problems("Sorry for the trouble, we'll look into it.", triage(), [{"record": ORDER}])
  assert problems == ["say that order NC-31877 is delivered"]


def test_other_languages_must_at_least_name_the_order():
  assert reply_problems("आपका ऑर्डर NC-31877 पहुँच गया है।", triage("hi"), [{"record": ORDER}]) == []
  assert reply_problems("आपका ऑर्डर पहुँच गया है।", triage("hi"), [{"record": ORDER}]) == ["mention order NC-31877"]


def test_claimed_actions_are_caught_with_curly_apostrophes():
  problems = reply_problems("I’ve gone ahead and cancelled both orders!", triage(category="cancel_order"), [])
  assert len(problems) == 1 and "don't say you cancelled" in problems[0]
  assert reply_problems("Your refund will be processed within 5 days.", triage(category="refund_return"), []) == []


def test_order_notes():
  assert order_note(None, None) == ""
  assert "No order NC-1" in order_note("NC-1", None)
  assert order_note("NC-31877", ORDER).startswith("Order NC-31877 on file: status delivered")


def test_invented_links_are_caught():
  assert reply_problems("See our [shop page](link) for details.", triage(category="product_question"), []) == [
    "don't add links or URLs"
  ]
  assert reply_problems("Visit https://nimbuscoffee.co/shop", triage(category="product_question"), []) == [
    "don't add links or URLs"
  ]
  assert reply_problems("Check the shop page for details.", triage(category="product_question"), []) == []


def test_refund_decisions_are_made_in_code():
  from datetime import date

  from nimbus.support.graph import refund_decision

  class Store:
    today = date(2026, 3, 20)

  base = {"order_id": "NC-1", "status": "delivered", "delivered_on": "2026-03-11", "opened": False, "total": 720}
  assert "is eligible" in refund_decision(Store(), base, "NC-1")
  assert "opened bags can't be returned" in refund_decision(Store(), {**base, "opened": True}, "NC-1")
  assert "outside the 14-day window" in refund_decision(Store(), {**base, "delivered_on": "2026-02-01"}, "NC-1")
  assert "refunds apply to delivered orders" in refund_decision(Store(), {**base, "status": "processing"}, "NC-1")
  assert "No order NC-9" in refund_decision(Store(), None, "NC-9")
