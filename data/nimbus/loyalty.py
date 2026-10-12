"""Nimbus Beans loyalty programme: points calculation."""

from datetime import date, timedelta

POINTS_PER_RUPEE = 0.1
REDEEM_THRESHOLD = 500
EXPIRY_DAYS = 365


class LoyaltyAccount:
  """A customer's points balance."""

  def __init__(self, customer_id: str):
    self.customer_id = customer_id
    self.entries: list[tuple[date, int]] = []

  def earn(self, amount_rupees: int, on: date) -> int:
    """Add points for a purchase: 1 point per 10 rupees."""
    points = int(amount_rupees * POINTS_PER_RUPEE)
    self.entries.append((on, points))
    return points

  def balance(self, today: date) -> int:
    """Points that have not expired yet."""
    cutoff = today - timedelta(days=EXPIRY_DAYS)
    return sum(points for earned, points in self.entries if earned > cutoff)


def can_redeem(account: LoyaltyAccount, today: date) -> bool:
  """A free bag needs at least REDEEM_THRESHOLD points."""
  return account.balance(today) >= REDEEM_THRESHOLD
