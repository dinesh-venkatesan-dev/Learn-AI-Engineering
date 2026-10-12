"""The Nimbus web app as deployed: the REST API and the signed-in account MCP server in one ASGI app (MCP Module 4).

  /health, /orders/..., /menu   the REST API (nimbus.api.app)
  /mcp/                         the account MCP server: JWTs, stateless, Host and Origin checked

Run it like any web app, as many replicas as you need (the public key comes from the environment):
  NIMBUS_JWT_PUBLIC_KEY="$(cat key.pub)" uv run uvicorn nimbus.mcp.web:create_web_app --factory --workers 2
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from nimbus.api.app import create_app
from nimbus.mcp.account import build_account_server
from nimbus.mcp.auth import verifier

ALLOWED_HOSTS = ["127.0.0.1:*", "localhost:*", "api.nimbus.example"]
ALLOWED_ORIGINS = ["https://nimbus.example"]  # our own web front end; browsers elsewhere are refused


def create_web_app(public_key: str | None = None) -> FastAPI:
  mcp_app = build_account_server(verifier(public_key)).http_app(
    path="/",  # mounted at /mcp below, so the endpoint is /mcp/
    stateless_http=True,  # no session kept in a worker: any replica answers any request, legacy clients included
    host_origin_protection=True,  # refuse unexpected Host headers (DNS rebinding) and browser Origins
    allowed_hosts=ALLOWED_HOSTS,
    allowed_origins=ALLOWED_ORIGINS,
  )
  app = create_app()
  api_lifespan = app.router.lifespan_context

  @asynccontextmanager
  async def both(app_: FastAPI):  # the MCP app's lifespan MUST run too, or every MCP request fails
    async with api_lifespan(app_), mcp_app.lifespan(app_):
      yield

  app.router.lifespan_context = both
  app.mount("/mcp", mcp_app)
  return app
