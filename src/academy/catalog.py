"""The course catalogue: tracks and modules (content/curriculum.json), module pages (content/<track>/NN_*.json) and the
syllabus of every learning point from the old courses (content/syllabus.json), with coverage computed from them.

Coverage rule: each syllabus item has exactly one home module (via the module's `from` patterns, minus its `except`
patterns, so a lesson can move to the module where it fits better). A *built* module must cover every item of its home
with its labs and production path (`covers` patterns), and its `capstones` with the production path itself, because in
this course the Nimbus product is the capstone. Tests enforce it.
"""

import importlib
import inspect
import json
from functools import lru_cache
from pathlib import Path

from nimbus.config import PROJECT_DIR

CONTENT = PROJECT_DIR / "content"


@lru_cache
def curriculum() -> dict:
  return json.loads((CONTENT / "curriculum.json").read_text())


@lru_cache
def syllabus() -> list[dict]:
  return json.loads((CONTENT / "syllabus.json").read_text())


@lru_cache
def importance() -> dict:
  """How much each module and lesson matters for a production LLM app: 3 essential, 2 important, 1 good to know."""
  return json.loads((CONTENT / "importance.json").read_text())


def lesson_stars(lesson: str) -> int:
  rated = importance()
  return rated["lessons"].get(lesson, rated["default_lesson_stars"])


def module_importance(track: str, module_id: str) -> dict:
  return importance()["modules"][f"{track}/{module_id}"]


def matches(item: dict, pattern: str) -> bool:
  """A pattern names a part (rag.p2), a lesson (rag.l06) or one item (rag.l06.2)."""
  return pattern in (item["part"], item["lesson"], item["id"])


def owns(module: dict, item: dict) -> bool:
  return any(matches(item, p) for p in module["from"]) and not any(matches(item, p) for p in module.get("except", []))


def home_items(module: dict) -> list[dict]:
  return [s for s in syllabus() if owns(module, s)]


def module_file(track: str, module_id: str) -> Path | None:
  return next(iter(sorted((CONTENT / track).glob(f"{module_id}_*.json"))), None) if (CONTENT / track).exists() else None


def load_module(track: str, module_id: str) -> dict | None:
  path = module_file(track, module_id)
  return json.loads(path.read_text()) if path else None


def covers_of(page: dict) -> list[str]:
  """Every covers pattern a module page declares (labs + production path)."""
  patterns = [p for item in page.get("labs", []) for p in item.get("covers", [])]
  return patterns + page.get("path", {}).get("covers", [])


def coverage() -> dict:
  """Per track and module: how many home items exist and how many a built module covers."""
  tracks = []
  for track in curriculum()["tracks"]:
    modules = []
    for module in track["modules"]:
      items = home_items(module)
      page = load_module(track["id"], module["id"]) if module["status"] != "planned" else None
      patterns = covers_of(page) if page else []
      covered = [s for s in items if any(matches(s, p) for p in patterns)]
      rated = module_importance(track["id"], module["id"])
      modules.append(
        {**module, "items": len(items), "covered": len(covered), "stars": rated["stars"], "why": rated["why"]}
      )
    tracks.append(
      {
        **{k: v for k, v in track.items() if k != "modules"},
        "modules": modules,
        "items": sum(m["items"] for m in modules),
        "covered": sum(m["covered"] for m in modules),
      }
    )
  return {
    "tracks": tracks,
    "items": sum(t["items"] for t in tracks),
    "covered": sum(t["covered"] for t in tracks),
    "scale": importance()["scale"],
  }


HOOKS = ("wrap_model_call", "wrap_tool_call", "before_model", "after_model", "before_agent", "after_agent")


def _written_function(obj):
  """Middleware made by a decorator (@dynamic_prompt, @before_model, ...) is an object; the function you wrote is
  kept in a closure of one of its hooks. Return that function, or the object unchanged."""
  if inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismodule(obj) or inspect.ismethod(obj):
    return obj
  for hook in HOOKS:
    method = getattr(obj, hook, None)
    for cell in getattr(getattr(method, "__func__", method), "__closure__", None) or ():
      value = cell.cell_contents
      if inspect.isfunction(value) and value.__name__ == type(obj).__name__:
        return value
  return obj


def source_of(ref: dict) -> dict:
  """The real source of a function or class, with its file, line numbers, and notes pinned to matching lines."""
  obj = importlib.import_module(ref["module"])
  for name in filter(None, ref["symbol"].split(".")):  # "VectorIndex.search" → the method; "" → the whole module
    obj = getattr(obj, name)
  obj = inspect.unwrap(obj)  # decorated functions (lru_cache, …): show the code that was written
  obj = _written_function(obj)
  lines, start = inspect.getsourcelines(obj)
  path = Path(inspect.getsourcefile(obj)).resolve()
  notes = []
  for note in ref.get("notes", []):
    at = next((i for i, line in enumerate(lines) if note["match"] in line), None)
    if at is not None:
      notes.append({"line": start + at, "text": note["text"]})
  return {
    "module": ref["module"],
    "symbol": ref["symbol"] or ref["module"],
    "why": ref.get("why", ""),
    "file": path.relative_to(PROJECT_DIR).as_posix(),
    "abs_path": str(path),
    "start": start,
    "source": "".join(lines),
    "notes": notes,
    "missing_notes": len(ref.get("notes", [])) - len(notes),
  }
