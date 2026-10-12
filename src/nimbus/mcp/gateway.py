"""The Nimbus MCP gateway: one server in front of the support and ops servers.

  build_gateway(public=False)   staff: support_* and ops_* tools
  build_gateway(public=True)    customers: tools tagged "internal" are hidden from the list AND refused if called

Middleware, outermost first: an audit log of every call (with timing), rate limiting, and response caching for
repeated reads.
"""

import logging
import time

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp_tasks import TasksExtension

from nimbus.mcp import ops, server

log = logging.getLogger(__name__)


class Audit(Middleware):
  """Every tool call with its arguments, outcome and time: the first thing to read when a host misbehaves."""

  def __init__(self) -> None:
    self.entries: list[dict] = []

  async def on_call_tool(self, context: MiddlewareContext, call_next):
    started = time.perf_counter()
    entry = {"tool": context.message.name, "arguments": context.message.arguments, "ok": True}
    try:
      return await call_next(context)
    except Exception:
      entry["ok"] = False
      raise
    finally:
      entry["ms"] = round((time.perf_counter() - started) * 1000, 1)
      self.entries.append(entry)
      log.info("mcp call %s", entry)


def build_gateway(public: bool = False, rate_per_second: float = 20.0) -> tuple[FastMCP, Audit]:
  gateway = FastMCP("Nimbus Gateway", instructions="Nimbus Coffee: orders, the café menu, policies; ops for staff.")
  gateway.mount(server.mcp, namespace="support")  # a live link: tools added to a child show up here
  gateway.mount(ops.mcp, namespace="ops")
  gateway.add_extension(TasksExtension())  # ops_weekly_report runs as a background task here too
  if public:
    gateway.disable(tags={"internal"})  # hidden from tools/list and refused when called
  audit = Audit()
  for middleware in (
    audit,
    RateLimitingMiddleware(max_requests_per_second=rate_per_second, burst_capacity=50),
    ResponseCachingMiddleware(),
  ):
    gateway.add_middleware(middleware)
  return gateway, audit
