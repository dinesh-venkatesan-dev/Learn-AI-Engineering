"""A load generator in ~20 lines: N virtual users, each sending requests one after another for a fixed time.
Real tools (locust, k6, vegeta) do the same with more reports.

  uv run python -m nimbus.api.loadtest http://127.0.0.1:8000 --users 8 --seconds 10
"""

import argparse
import asyncio
import statistics
import time

import httpx


async def load(base_url: str, users: int, seconds: float, path: str = "/agent/chat", customer: str = "c-1042") -> dict:
  latencies, errors = [], 0

  async def user(client: httpx.AsyncClient) -> None:
    nonlocal errors
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
      started = time.perf_counter()
      try:
        r = await client.post(path, json={"message": "Where is my order?"}, headers={"X-Customer-ID": customer})
        errors += r.status_code >= 400
      except httpx.HTTPError:
        errors += 1
      latencies.append(time.perf_counter() - started)

  async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
    started = time.perf_counter()
    await asyncio.gather(*(user(client) for _ in range(users)))
    elapsed = time.perf_counter() - started
  ordered = sorted(latencies)
  return {
    "users": users,
    "requests": len(ordered),
    "throughput_rps": round(len(ordered) / elapsed, 2),
    "p50_s": round(statistics.median(ordered), 2),
    "p95_s": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
    "error_rate": round(errors / max(len(ordered), 1), 3),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("base_url")
  parser.add_argument("--users", type=int, default=8)
  parser.add_argument("--seconds", type=float, default=10)
  args = parser.parse_args()
  print(asyncio.run(load(args.base_url, args.users, args.seconds)))
