"""One error shape for the whole API, for people AND programs:  {"error": {"code", "message", ...}}

raise ModelUnavailable()      anywhere in the code → 503 with Retry-After
raise NotFound("No order ...")                     → 404
validation errors                                  → 422 with every field that's wrong
anything unexpected                                → 500, details only in OUR logs (with the request id)
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

log = logging.getLogger("nimbus.api")


class ApiError(Exception):
  status, code = 400, "bad_request"

  def __init__(self, message: str, headers: dict[str, str] | None = None, **extra):
    super().__init__(message)
    self.message, self.headers, self.extra = message, headers, extra


class NotFound(ApiError):
  status, code = 404, "not_found"


class Unauthorized(ApiError):
  status, code = 401, "unauthorized"


class ModelUnavailable(ApiError):
  status, code = 503, "model_unavailable"

  def __init__(self, message: str = "The assistant is busy. Please retry in a few seconds.", retry_after: int = 10):
    super().__init__(message, headers={"Retry-After": str(retry_after)})


class ModelError(ApiError):
  status, code = 502, "model_error"


def error_response(status: int, code: str, message: str, headers=None, **extra) -> JSONResponse:
  return JSONResponse({"error": {"code": code, "message": message, **extra}}, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
  @app.exception_handler(ApiError)
  async def api_error(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(exc.status, exc.code, exc.message, exc.headers, **exc.extra)

  @app.exception_handler(HTTPException)
  async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    codes = {404: "not_found", 405: "method_not_allowed", 401: "unauthorized", 403: "forbidden", 415: "unsupported"}
    return error_response(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail), exc.headers)

  @app.exception_handler(RequestValidationError)
  async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = [
      {"field": ".".join(str(p) for p in e["loc"][1:]) or e["loc"][0], "problem": e["msg"]} for e in exc.errors()
    ]
    return error_response(422, "invalid_request", "Some fields are invalid.", fields=fields)  # messages only: no input

  @app.exception_handler(Exception)
  async def unexpected(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    log.error("unhandled error on %s (request %s): %r", request.url.path, request_id, exc, exc_info=exc)
    return error_response(500, "internal_error", "Something went wrong on our side.", request_id=request_id)


async def ask_model(model, messages, failed: str = "The assistant couldn't answer this request.", config=None):
  """One model call for an endpoint: a provider that's down or busy becomes 503 (retry), anything else 502."""
  from nimbus.llm.errors import classify

  try:
    return await model.ainvoke(messages, config)
  except Exception as error:  # noqa: BLE001 - every provider failure becomes a meaningful status
    if classify(error).action == "retry":
      raise ModelUnavailable() from error
    raise ModelError(failed) from error
