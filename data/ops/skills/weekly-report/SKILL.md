---
name: weekly-report
description: How to write the weekly ops review and which managers to email afterwards.
---
# Weekly ops review

1. Get every café's numbers from sales-analyst and its low reviews from voice-of-customer.
2. A café NEEDS ATTENTION when its revenue fell more than 10% on the week before, or its average rating is below 4.0.
3. Write /review.md in this order:
   - `## Summary`: a table with one row per café: café, revenue (₹), change (%), rating.
   - `## Needs attention`: each café that needs attention, with the reason (the rule it broke, with the number).
   - `## Cafés`: one short section per café, with its best seller and any low reviews (ids and comments).
     List only the reviews voice-of-customer reported, with their ids; when it reported none, write "No low reviews".
4. Email the manager of every café that needs attention (and no one else), following the house rules.
