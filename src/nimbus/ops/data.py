"""Nimbus ops data: the numbers an ops review needs, computed in code from the warehouse (never by a model).

  cafe_names()                          the cafés
  cafe_week(cafe, start, end)           revenue, orders, refunds, average rating, best and weakest sellers, and the
                                        change against the week before: everything a review quotes
  low_reviews(cafe, start, end)         reviews rated 2 or lower, with their ids, for evidence
  ground_truth(start, end)              every café's week: what the evaluation checks a review against

Dates are ISO days, inclusive. The warehouse covers 2026-07-01 to 2026-09-30.
"""

from datetime import date, timedelta

from nimbus.data.sql import query

WEEK = ("2026-09-24", "2026-09-30")  # the last full week in the warehouse: the review's default
COVERED = ("2026-07-01", "2026-09-30")  # the warehouse's first and last day


def cafe_names() -> list[str]:
  return [row[0] for row in query("select name from cafes order by cafe_id")[1]]


def _resolve(cafe: str) -> str:
  names = cafe_names()
  matches = [n for n in names if cafe.casefold().removeprefix("nimbus ").strip() in n.casefold()]
  if len(matches) != 1:
    raise ValueError(f"Unknown café {cafe!r}. Cafés: {', '.join(names)}.")
  return matches[0]


def _revenue(name: str, start: str, end: str) -> int:
  sql = (
    "select coalesce(sum(i.quantity * i.unit_price), 0) from orders o join cafes c using(cafe_id) "
    "join order_items i using(order_id) where c.name = :n and o.status = 'completed' "
    "and date(o.ordered_at) between :s and :e"
  )
  return _scalar(sql, n=name, s=start, e=end)


def _scalar(sql: str, **params):
  from sqlalchemy import text

  from nimbus.data.sql import engine

  with engine().connect() as connection:
    return connection.execute(text(sql), params).scalar()


def _rows(sql: str, **params) -> list[tuple]:
  from sqlalchemy import text

  from nimbus.data.sql import engine

  with engine().connect() as connection:
    return [tuple(r) for r in connection.execute(text(sql), params)]


def cafe_week(cafe: str, start: str = WEEK[0], end: str = WEEK[1]) -> dict:
  name = _resolve(cafe)
  days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
  before = (
    (date.fromisoformat(start) - timedelta(days=days)).isoformat(),
    (date.fromisoformat(start) - timedelta(days=1)).isoformat(),
  )
  revenue, previous = _revenue(name, start, end), _revenue(name, *before)
  window = "c.name = :n and date(o.ordered_at) between :s and :e"
  counts = dict(
    _rows(
      f"select o.status, count(*) from orders o join cafes c using(cafe_id) where {window} group by 1",
      n=name,
      s=start,
      e=end,
    )
  )
  rating = _scalar(
    "select round(avg(r.rating), 2) from reviews r join orders o using(order_id) join cafes c using(cafe_id) "
    f"where {window}",
    n=name,
    s=start,
    e=end,
  )
  items = _rows(
    "select p.name, sum(i.quantity * i.unit_price) as revenue from orders o join cafes c using(cafe_id) "
    "join order_items i using(order_id) join products p using(product_id) "
    f"where {window} and o.status = 'completed' group by 1 order by 2 desc, 1",
    n=name,
    s=start,
    e=end,
  )
  return {
    "cafe": name,
    "start": start,
    "end": end,
    "revenue_rupees": int(revenue),
    "previous_week_rupees": int(previous),
    "change_percent": round((revenue - previous) / previous * 100, 1) if previous else None,
    "completed_orders": counts.get("completed", 0),
    "refunded_orders": counts.get("refunded", 0),
    "average_rating": rating,
    "best_seller": items[0][0] if items else None,
    "weakest_seller": items[-1][0] if items else None,
  }


def low_reviews(cafe: str, start: str = WEEK[0], end: str = WEEK[1], max_rating: int = 2) -> list[dict]:
  name = _resolve(cafe)
  rows = _rows(
    "select r.review_id, r.rating, r.comment from reviews r join orders o using(order_id) join cafes c using(cafe_id) "
    "where c.name = :n and date(o.ordered_at) between :s and :e and r.rating <= :m order by r.review_id",
    n=name,
    s=start,
    e=end,
    m=max_rating,
  )
  return [{"review_id": i, "rating": r, "comment": c} for i, r, c in rows]


def ground_truth(start: str = WEEK[0], end: str = WEEK[1]) -> dict[str, dict]:
  return {name: cafe_week(name, start, end) for name in cafe_names()}
