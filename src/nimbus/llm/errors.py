"""Model errors, sorted by what to do about them.

The client already retries temporary failures (max_retries). What reaches your code has either exhausted its retries
or can never succeed as sent. Callers need one decision: try again later, or fix the request.
"""

from dataclasses import dataclass
from typing import Literal

import httpx

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
FIX_STATUSES = {
  400: "bad request",
  401: "invalid API key",
  403: "not allowed",
  404: "unknown model or endpoint",
  413: "request too large",
  422: "invalid request body",
}


@dataclass(frozen=True)
class Failure:
  action: Literal["retry", "fix"]  # retry: temporary, try again later · fix: the request or configuration is wrong
  reason: str
  status: int | None = None
  detail: str = ""  # the provider's own message, for logs and alerts


def detail_of(error: BaseException) -> str:
  """The provider's error message (e.g. 'Invalid model: ...'), whichever exception carries the response."""
  for candidate in (error, error.__cause__, error.__context__):
    response = getattr(candidate, "response", None)
    if response is not None:
      try:
        body = response.json()
        return str(body.get("message") or body.get("detail") or body.get("error", {}).get("message") or body)[:200]
      except Exception:  # noqa: BLE001 — not JSON, or the body was already consumed
        return str(getattr(response, "text", ""))[:200]
  return ""


def status_of(error: BaseException) -> int | None:
  """The HTTP status, whichever provider SDK raised the error."""
  for candidate in (error, error.__cause__, error.__context__):
    if candidate is None:
      continue
    if isinstance(candidate, httpx.HTTPStatusError):
      return candidate.response.status_code
    status = getattr(candidate, "status_code", None) or getattr(
      getattr(candidate, "response", None), "status_code", None
    )
    if isinstance(status, int):
      return status
  return None


def classify(error: BaseException) -> Failure:
  if isinstance(error, (httpx.TimeoutException, TimeoutError)) or "timed out" in str(error).lower():
    return Failure("retry", "timed out")
  if isinstance(error, (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError)):
    return Failure("retry", "network error")
  status = status_of(error)
  if status in RETRY_STATUSES:
    return Failure("retry", "rate limited" if status == 429 else "provider error", status, detail_of(error))
  detail = detail_of(error)
  if status == 400 and "Parsing failed" in detail:  # Groq: the model's own output didn't parse; a retry may work
    return Failure("retry", "unparsable model output", status, detail)
  if status in FIX_STATUSES:
    return Failure("fix", FIX_STATUSES[status], status, detail)
  if status and status >= 500:
    return Failure("retry", "provider error", status, detail_of(error))
  return Failure("fix", f"{type(error).__name__}: {error}"[:200], status)
