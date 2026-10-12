"""The Nimbus HTTP API: orders, the café menu, support replies, and Nimbus Notes (accounts, notes, chat).

  create_app(store=None, settings=None, chat_model=None)   a factory, so tests and labs build apps with their own parts
  lifespan        at start-up, ONCE: the tables, the chat model client, the support agent (with its checkpointer),
                  the job queue; at shutdown: everything closed
  routers         orders, menu, support, agent, reports (API key + audit); auth, notes, chat (signed-in users)
  errors          one shape, {"error": {"code", "message", ...}}, for every failure
  middleware      request id + timing, GZip for large responses, CORS for our own browser apps (exact origins)

Run it:  uv run uvicorn nimbus.api.app:app --reload        then open http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from langchain_core.language_models import BaseChatModel

from nimbus.api.db import create_tables, make_engine
from nimbus.api.errors import install_error_handlers
from nimbus.api.middleware import install_request_context
from nimbus.api.routers import agent, auth, chat, feedback, menu, notes, orders, reports, support
from nimbus.config import Settings, get_settings
from nimbus.observability.metrics import RequestMetrics, metrics_endpoint, setup_json_logs
from nimbus.observability.tracing import guard_uploads
from nimbus.support.tools import SupportStore

# Kept for the labs and older imports: the menu's query model and categories live with the menu router now.
Category, MenuFilter = menu.Category, menu.MenuFilter


def connect_queue(settings: Settings):
  """The RQ queue, if Redis answers; without it the API runs and /reports answers 503."""
  import redis
  from rq import Queue

  client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5)
  try:
    client.ping()
  except redis.RedisError:
    client.close()
    return None
  return Queue("reports", connection=client)


def create_app(
  store: SupportStore | None = None, *, settings: Settings | None = None, chat_model: BaseChatModel | None = None
) -> FastAPI:
  settings = settings or get_settings()
  guard_uploads(settings)  # traces go to LangSmith only with a key configured for Nimbus
  if settings.log_format == "json":
    setup_json_logs(settings.log_level)

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    from nimbus.api.agent_service import AgentService
    from nimbus.llm import get_chat_model
    from nimbus.llm.models import aclose_clients

    create_tables(app.state.engine)
    if chat_model is None and settings.fake_model_delay is not None:  # load tests: measure the service, not the model
      from nimbus.llm.fakes import slow_fake

      app.state.chat_model = slow_fake(settings.fake_model_delay)
    else:
      app.state.chat_model = chat_model or get_chat_model(settings)  # ONE client for every request
    app.state.agent_service = await AgentService.start(settings, app.state.chat_model, app.state.store)  # ONCE
    app.state.queue = connect_queue(settings)
    app.state.redis = app.state.queue.connection if app.state.queue is not None else None  # limits and budgets
    yield
    await app.state.agent_service.close()
    if app.state.queue is not None:
      app.state.queue.connection.close()
    if chat_model is None:
      await aclose_clients()  # close the HTTP connections we opened
    app.state.engine.dispose()

  app = FastAPI(
    title="Nimbus API",
    version="0.2.0",
    description="Orders, the café menu and support replies of Nimbus Coffee.",
    lifespan=lifespan,
  )
  app.state.settings = settings
  app.state.store = store or SupportStore()
  app.state.audit = []
  app.state.shadow = []  # shadow answers, for comparison (a database or the traces in production)
  app.state.engine = make_engine(settings.api_database_url)
  app.state.chat_model = chat_model  # set again by the lifespan; here so apps without a lifespan run still work

  app.add_route("/metrics", metrics_endpoint, include_in_schema=False)  # for Prometheus

  @app.get("/health", tags=["ops"])
  def health() -> dict[str, str]:
    return {"status": "ok"}

  for router in (
    orders.router,
    menu.router,
    support.router,
    auth.router,
    notes.router,
    chat.router,
    feedback.router,
    agent.router,
    reports.router,
  ):
    app.include_router(router)
  install_error_handlers(app)
  # Middleware added LAST runs FIRST. GZip innermost, where responses still have their size (small ones stay as
  # they are); the request id and timing around it; CORS outermost, answering browser preflights first.
  app.add_middleware(GZipMiddleware, minimum_size=1000)
  app.add_middleware(RequestMetrics)  # inside the request id: its metrics and log lines carry it
  install_request_context(app)
  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # exact origins, never "*" with credentials
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Server-Timing"],
  )
  return app


app = create_app()
