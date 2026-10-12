"""Business context for model-written SQL (Data Module 1): what the schema can't say.

The schema says which columns exist; the GLOSSARY says what the business means by its words (revenue counts completed
orders only) and lists the allowed values of categorical columns; EXAMPLES show the house style on the tricky rules.
"""

GLOSSARY = """Business definitions (use these exactly):
- revenue: SUM(order_items.quantity * order_items.unit_price) over orders with status = 'completed' only.
- order count: number of orders, any status; "completed orders" means status = 'completed'.
- average order value (AOV): revenue / number of completed orders.
- refund rate: refunded orders / all orders (status = 'refunded').
- dates: orders.ordered_at is 'YYYY-MM-DD HH:MM:SS' text; use strftime / date() to group. The data covers Q3 2026
  (July to September). "Last month" = September 2026. strftime('%w', ...) numbers weekdays 0 = Sunday ... 6 = Saturday.
- prices are in rupees. Cafés are identified by cafes.name, e.g. 'Nimbus Bandra'.
Allowed values of categorical columns (use exactly these):
- orders.channel: 'in_store', 'app', 'delivery' (every order has exactly one channel)
- orders.status: 'completed', 'refunded', 'cancelled'
- products.category: 'drink', 'food'
- customers.loyalty_tier: 'none', 'silver', 'gold'
- cafes.name: 'Nimbus Bandra', 'Nimbus Colaba', 'Nimbus Indiranagar', 'Nimbus Koramangala', 'Nimbus Hauz Khas',
  'Nimbus Besant Nagar'
"""

EXAMPLES = """Examples of good queries:
Q: Revenue per city in July 2026?
SQL: SELECT c.city, SUM(oi.quantity * oi.unit_price) AS revenue FROM orders o JOIN order_items oi ON oi.order_id =
     o.order_id JOIN cafes c ON c.cafe_id = o.cafe_id WHERE o.status = 'completed' AND strftime('%Y-%m', o.ordered_at)
     = '2026-07' GROUP BY c.city ORDER BY revenue DESC;
Q: Average order value for app orders?
SQL: SELECT ROUND(1.0 * SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.order_id), 2) AS aov FROM orders o
     JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status = 'completed' AND o.channel = 'app';
"""


def sql_instructions(schema: str, glossary: str = "", examples: str = "") -> str:
  """The system prompt for writing SQL: the schema, then whatever business context there is."""
  context = "\n\n".join(part for part in (glossary, examples) if part)
  return (
    f"You write SQLite queries for Nimbus Coffee's database.\n\nSchema:\n{schema}\n\n"
    + (f"{context}\n\n" if context else "")
    + "Write ONE query that answers the question"
    + (", following the definitions exactly." if glossary else ". Return only the columns needed.")
  )
