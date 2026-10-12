"""Nimbus Coffee's data warehouse: a SQLite database the course's agents query.

build_warehouse() creates ai-course/.cache/data/nimbus.db the first time (deterministic: the same data everywhere).
Tables
  cafes(cafe_id, name, city, opened_on)
  products(product_id, name, category, price)                       price in rupees
  customers(customer_id, name, city, signup_date, loyalty_tier)       tier: none / silver / gold
  orders(order_id, customer_id, cafe_id, ordered_at, channel, status) channel: in_store / app / delivery
                                                                      status: completed / refunded / cancelled
  order_items(order_id, product_id, quantity, unit_price)
  reviews(review_id, order_id, rating, comment)                       rating 1–5
"""

import random
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

from nimbus.config import PROJECT_DIR

DB_PATH = PROJECT_DIR / ".cache" / "data" / "nimbus.db"

CAFES = [
  (1, "Nimbus Bandra", "Mumbai", "2023-02-01"),
  (2, "Nimbus Colaba", "Mumbai", "2024-06-15"),
  (3, "Nimbus Indiranagar", "Bengaluru", "2023-05-10"),
  (4, "Nimbus Koramangala", "Bengaluru", "2025-01-20"),
  (5, "Nimbus Hauz Khas", "Delhi", "2024-03-01"),
  (6, "Nimbus Besant Nagar", "Chennai", "2026-08-01"),
]
PRODUCTS = [
  (1, "Filter Coffee", "drink", 120),
  (2, "Cappuccino", "drink", 180),
  (3, "Oat Latte", "drink", 220),
  (4, "Cold Brew", "drink", 200),
  (5, "Masala Chai", "drink", 100),
  (6, "Mocha", "drink", 230),
  (7, "Iced Americano", "drink", 170),
  (8, "Banana Bread", "food", 150),
  (9, "Vegan Brownie", "food", 170),
  (10, "Croissant", "food", 140),
  (11, "Paneer Sandwich", "food", 210),
  (12, "Granola Bowl", "food", 260),
]
FIRST = [
  "Aarav",
  "Diya",
  "Ishaan",
  "Kavya",
  "Rohan",
  "Ananya",
  "Vikram",
  "Meera",
  "Arjun",
  "Priya",
  "Kabir",
  "Sara",
  "Neel",
  "Tara",
  "Dev",
  "Isha",
  "Yash",
  "Riya",
  "Om",
  "Zoya",
]
LAST = ["Sharma", "Iyer", "Khan", "Reddy", "Patel", "Nair", "Gupta", "Das", "Menon", "Singh"]
POSITIVE = [
  "Great coffee, friendly staff.",
  "Loved the cold brew.",
  "Quick service and cosy seating.",
  "Best banana bread in town.",
  "Perfect oat latte.",
]
NEGATIVE = [
  "Order took too long.",
  "Coffee was lukewarm.",
  "Wrong item delivered.",
  "Too expensive for the size.",
  "Delivery arrived spilled.",
]

START, END = date(2026, 7, 1), date(2026, 9, 30)  # Q3 2026


def _generate(connection: sqlite3.Connection) -> None:
  rng = random.Random(40)
  connection.executescript("""
    CREATE TABLE cafes (cafe_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL, opened_on DATE NOT NULL);
    CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
                           price INTEGER NOT NULL);
    CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL,
                            signup_date DATE NOT NULL, loyalty_tier TEXT NOT NULL);
    CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers,
                         cafe_id INTEGER NOT NULL REFERENCES cafes, ordered_at TIMESTAMP NOT NULL,
                         channel TEXT NOT NULL, status TEXT NOT NULL);
    CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders,
                              product_id INTEGER NOT NULL REFERENCES products, quantity INTEGER NOT NULL,
                              unit_price INTEGER NOT NULL);
    CREATE TABLE reviews (review_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders,
                          rating INTEGER NOT NULL, comment TEXT);
  """)
  connection.executemany("INSERT INTO cafes VALUES (?, ?, ?, ?)", CAFES)
  connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)

  cities = [c[2] for c in CAFES]
  customers = []
  for customer_id in range(1, 801):
    tier = rng.choices(["none", "silver", "gold"], weights=[70, 22, 8])[0]
    signup = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 1300))
    customers.append(
      (customer_id, f"{rng.choice(FIRST)} {rng.choice(LAST)}", rng.choice(cities), signup.isoformat(), tier)
    )
  connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

  cafe_weights = [30, 18, 26, 14, 22, 8]  # Bandra busiest; Besant Nagar newest (opened Aug 1)
  product_weights = [14, 16, 10, 12, 15, 6, 7, 6, 5, 4, 3, 2]
  orders, items, reviews = [], [], []
  order_id = 0
  day = START
  while day <= END:
    daily = 55 + (10 if day.weekday() >= 5 else 0) + (day - START).days // 9  # weekends busier, slow growth
    for _ in range(daily):
      cafe = rng.choices(CAFES, weights=cafe_weights)[0]
      if cafe[0] == 6 and day < date(2026, 8, 1):
        continue  # not open yet
      order_id += 1
      customer = rng.choice(customers)
      at = datetime(
        day.year, day.month, day.day, rng.choice([7, 8, 8, 9, 10, 12, 13, 16, 17, 18, 19, 20]), rng.randrange(60)
      )
      channel = rng.choices(["in_store", "app", "delivery"], weights=[60, 25, 15])[0]
      status = rng.choices(["completed", "refunded", "cancelled"], weights=[93, 4 if channel != "delivery" else 9, 3])[
        0
      ]
      orders.append((order_id, customer[0], cafe[0], at.isoformat(sep=" "), channel, status))
      for product in rng.sample(rng.choices(PRODUCTS, weights=product_weights, k=3), k=rng.choice([1, 1, 2, 2, 3])):
        items.append((order_id, product[0], rng.choice([1, 1, 1, 2]), product[3]))
      if status == "completed" and rng.random() < 0.12:
        good = rng.random() < 0.8
        reviews.append(
          (
            len(reviews) + 1,
            order_id,
            rng.choice([4, 5]) if good else rng.choice([1, 2, 3]),
            rng.choice(POSITIVE if good else NEGATIVE),
          )
        )
    day += timedelta(days=1)
  connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
  connection.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", items)
  connection.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?)", reviews)
  connection.commit()


def build_warehouse(path: Path = DB_PATH) -> Path:
  """Create the database if it doesn't exist yet; return its path."""
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    with closing(sqlite3.connect(temporary)) as connection:  # sqlite3's own `with` commits but doesn't close
      _generate(connection)
    temporary.rename(path)  # only after the connection is closed
  return path
