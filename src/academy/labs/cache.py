"""A disk cache for the model calls labs make, so re-running a lab costs no requests.

Labs say "cached after the first run" in their call counts. Production caching (exact and semantic, with
expiry and invalidation) is its own module in the Production track; this is only a convenience for the labs.
"""

import hashlib
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from nimbus.config import PROJECT_DIR

CACHE = PROJECT_DIR / ".cache" / "labs" / "calls"


def _path(kind: str, model: BaseChatModel, prompt: Any) -> Any:
  name = getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__
  key = json.dumps([kind, str(name), prompt], default=str, ensure_ascii=False, sort_keys=True)
  return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def text(model: BaseChatModel, prompt: Any) -> str:
  """model.invoke(prompt).text, cached on disk by model and prompt."""
  path = _path("text", model, prompt)
  if path.exists():
    return json.loads(path.read_text())["text"]
  reply = model.invoke(prompt).text.strip()
  CACHE.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({"text": reply}, ensure_ascii=False))
  return reply


def structured[T: BaseModel](model: BaseChatModel, schema: type[T], prompt: Any) -> T:
  """model.with_structured_output(schema).invoke(prompt), cached on disk."""
  path = _path(f"structured:{schema.__name__}:{json.dumps(schema.model_json_schema(), sort_keys=True)}", model, prompt)
  if path.exists():
    return schema.model_validate_json(path.read_text())
  result = model.with_structured_output(schema, method="json_schema").invoke(prompt)
  CACHE.mkdir(parents=True, exist_ok=True)
  path.write_text(result.model_dump_json())
  return result
