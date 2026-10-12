"""The completeness guarantee: every learning point from the old courses has exactly one home module, and a module
marked "built" covers every point it owns. Nothing can be dropped silently."""

import pytest

import academy.labs.agents.m01_graphs  # noqa: F401
import academy.labs.agents.m02_control  # noqa: F401
import academy.labs.agents.m03_workflows  # noqa: F401
import academy.labs.agents.m04_agents  # noqa: F401
import academy.labs.agents.m05_memory  # noqa: F401
import academy.labs.agents.m06_hitl  # noqa: F401
import academy.labs.agents.m07_multi  # noqa: F401
import academy.labs.agents.m08_production  # noqa: F401
import academy.labs.agents.m09_deep  # noqa: F401
import academy.labs.agents.m10_deep_control  # noqa: F401
import academy.labs.data.m01_text_to_sql  # noqa: F401
import academy.labs.data.m02_sql_agents  # noqa: F401
import academy.labs.data.m03_analytics  # noqa: F401
import academy.labs.fastapi.m01_apis  # noqa: F401
import academy.labs.fastapi.m02_data_streaming  # noqa: F401
import academy.labs.fastapi.m03_security_testing  # noqa: F401
import academy.labs.fastapi.m04_agent_service  # noqa: F401
import academy.labs.langchain.m01_models  # noqa: F401  (registers the labs)
import academy.labs.langchain.m02_lcel  # noqa: F401
import academy.labs.langchain.m03_tools  # noqa: F401
import academy.labs.langchain.m04_agents  # noqa: F401
import academy.labs.langchain.m05_middleware  # noqa: F401
import academy.labs.mcp.m01_protocol  # noqa: F401
import academy.labs.mcp.m02_servers  # noqa: F401
import academy.labs.mcp.m03_clients  # noqa: F401
import academy.labs.mcp.m04_production  # noqa: F401
import academy.labs.production.m01_reliability  # noqa: F401
import academy.labs.production.m02_caching  # noqa: F401
import academy.labs.production.m03_tracing  # noqa: F401
import academy.labs.production.m04_evaluation  # noqa: F401
import academy.labs.production.m05_shipping  # noqa: F401
import academy.labs.rag.m01_foundations  # noqa: F401  (registers the labs)
import academy.labs.rag.m02_loading  # noqa: F401
import academy.labs.rag.m03_ocr  # noqa: F401
import academy.labs.rag.m04_chunking  # noqa: F401
import academy.labs.rag.m05_storage  # noqa: F401
import academy.labs.rag.m06_retrieval  # noqa: F401
import academy.labs.rag.m06_transform  # noqa: F401
import academy.labs.rag.m07_generation  # noqa: F401
import academy.labs.rag.m08_advanced  # noqa: F401
import academy.labs.rag.m09_production  # noqa: F401
from academy import catalog
from academy.labs.runtime import REGISTRY

ALL_MODULES = [(t["id"], m) for t in catalog.curriculum()["tracks"] for m in t["modules"]]
BUILT = [(track, m) for track, m in ALL_MODULES if m["status"] == "built"]
OPEN = [(track, m) for track, m in ALL_MODULES if m["status"] != "planned"]  # built, or a preview with some labs


def test_every_learning_point_has_exactly_one_home():
  homes = {s["id"]: [f"{t}/{m['id']}" for t, m in ALL_MODULES if catalog.owns(m, s)] for s in catalog.syllabus()}
  assert not [i for i, h in homes.items() if not h], "learning points with no module"
  assert not {i: h for i, h in homes.items() if len(h) > 1}, "learning points claimed by two modules"


def test_every_module_and_rated_lesson_is_real():
  rated = catalog.importance()
  modules = {f"{t}/{m['id']}" for t, m in ALL_MODULES}
  assert set(rated["modules"]) == modules, "every module needs exactly one rating"
  assert all(r["stars"] in (1, 2, 3) and r["why"] for r in rated["modules"].values())
  lessons = {s["lesson"] for s in catalog.syllabus()}
  assert set(rated["lessons"]) <= lessons, f"unknown lessons: {set(rated['lessons']) - lessons}"
  assert set(rated["lessons"].values()) <= {1, 3}, "2 is the default: list only 1s and 3s"


@pytest.mark.parametrize(("track", "module"), BUILT, ids=[f"{t}/{m['id']}" for t, m in BUILT])
def test_built_module_covers_everything_it_owns(track, module):
  page = catalog.load_module(track, module["id"])
  assert page is not None, "a built module needs its page"
  patterns = catalog.covers_of(page)
  missing = [s["id"] for s in catalog.home_items(module) if not any(catalog.matches(s, p) for p in patterns)]
  assert not missing, f"not covered yet: {missing}"
  for p in patterns:  # a typo in a pattern would silently cover nothing
    assert any(catalog.matches(s, p) for s in catalog.syllabus()), f"pattern {p} matches no learning point"


@pytest.mark.parametrize(("track", "module"), BUILT, ids=[f"{t}/{m['id']}" for t, m in BUILT])
def test_built_module_path_covers_its_capstones(track, module):
  path = catalog.load_module(track, module["id"])["path"].get("covers", [])
  capstone = [s for s in catalog.home_items(module) if any(catalog.matches(s, p) for p in module.get("capstones", []))]
  missing = [s["id"] for s in capstone if not any(catalog.matches(s, p) for p in path)]
  assert not missing, f"capstone points the production path doesn't cover: {missing}"


def test_merge_patterns_are_real():
  for _, m in ALL_MODULES:
    for p in m.get("except", []) + m.get("capstones", []):
      assert any(catalog.matches(s, p) for s in catalog.home_items(m) + catalog.syllabus()), f"{m['id']}: {p}"
    for p in m.get("capstones", []):
      assert any(catalog.matches(s, p) for s in catalog.home_items(m)), f"{m['id']}: capstone {p} isn't in its home"


@pytest.mark.parametrize(("track", "module"), OPEN, ids=[f"{t}/{m['id']}" for t, m in OPEN])
def test_built_module_labs_and_code_resolve(track, module):
  page = catalog.load_module(track, module["id"])
  for entry in page["labs"] + [s for s in page["path"]["steps"] if s.get("lab")]:
    assert entry["lab"] in REGISTRY, f"unknown lab {entry['lab']}"
  refs = [r for entry in page["labs"] for r in entry["code"]] + [r for s in page["path"]["steps"] for r in s["code"]]
  for ref in refs:
    resolved = catalog.source_of(ref)
    assert resolved["missing_notes"] == 0, f"{ref['symbol']}: a note's `match` text isn't in the code any more"
