"""Dependencies: what endpoints need, read from the request and from what the lifespan created (app.state).

SettingsDep   the app's settings (create_app(settings=...) in tests), never the global get_settings()
ChatModelDep  the ONE chat client the lifespan created
StoreDep      the order system
ApiKey        the caller's key, checked (sub-dependency of the settings)
"""

import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, Header, Request
from langchain_core.language_models import BaseChatModel

from nimbus.api.errors import Unauthorized
from nimbus.config import Settings
from nimbus.support.tools import SupportStore


def app_settings(request: Request) -> Settings:
  return request.app.state.settings


def chat_model(request: Request) -> BaseChatModel:
  return request.app.state.chat_model


def store(request: Request) -> SupportStore:
  return request.app.state.store


SettingsDep = Annotated[Settings, Depends(app_settings)]
ChatModelDep = Annotated[BaseChatModel, Depends(chat_model)]
StoreDep = Annotated[SupportStore, Depends(store)]


def api_key(settings: SettingsDep, x_api_key: Annotated[str | None, Header()] = None) -> str:
  """The caller's id (a prefix of the key's hash, never the key). Keys are matched by hash (api_key_hashes) or, for
  local setups, as plain values in constant time (api_keys). With no keys configured the API is open (development)."""
  if not settings.api_keys and not settings.api_key_hashes:
    return "dev"
  if x_api_key:
    digest = hashlib.sha256(x_api_key.encode()).hexdigest()
    if digest in settings.api_key_hashes or any(
      hmac.compare_digest(x_api_key, k.get_secret_value()) for k in settings.api_keys
    ):
      return digest[:12]
  raise Unauthorized("Missing or unknown X-API-Key.", headers={"WWW-Authenticate": "ApiKey"})


ApiKey = Annotated[str, Depends(api_key)]


def audit(request: Request, key: ApiKey) -> None:
  """Router-wide: every call to a protected router is recorded with the caller (a dependency that returns nothing)."""
  request.app.state.audit.append({"path": request.url.path, "method": request.method, "caller": key})
