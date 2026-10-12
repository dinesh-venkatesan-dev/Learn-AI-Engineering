"""A prompt registry: the same model as LangSmith's Prompt Hub, kept in files next to the code.

  commit = registry.push("support-reply", messages, model={"temperature": 0})   every push is a commit (a hash)
  registry.tag("support-reply", "production", commit)                          a moving pointer (staging, production)
  prompt, version = registry.pull("support-reply:production")                  what the app runs: cached briefly
  registry.pull("support-reply:<commit>")                                       one exact version

With a LangSmith key the same operations map to client.push_prompt / pull_prompt and moving a commit tag; the file
registry is what Nimbus uses without one, and in tests.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from nimbus.config import PROJECT_DIR

PROMPTS_DIR = PROJECT_DIR / "data" / "prompts"


@dataclass(frozen=True)
class PromptVersion:
  name: str
  commit: str
  messages: tuple
  model: dict

  def template(self) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([tuple(m) for m in self.messages])


class PromptRegistry:
  def __init__(self, folder: Path = PROMPTS_DIR, cache_s: float = 60) -> None:
    self.folder, self.cache_s = folder, cache_s
    self._cache: dict[str, tuple[float, PromptVersion]] = {}
    folder.mkdir(parents=True, exist_ok=True)

  def _path(self, name: str) -> Path:
    return self.folder / f"{name}.json"

  def _load(self, name: str) -> dict:
    path = self._path(name)
    return json.loads(path.read_text()) if path.exists() else {"commits": {}, "order": [], "tags": {}}

  def _save(self, name: str, data: dict) -> None:
    self._path(name).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")

  def push(self, name: str, messages: list[tuple[str, str]], model: dict | None = None, note: str = "") -> str:
    """A new commit (or the existing one, if nothing changed): the hash of the messages and model settings."""
    body = {"messages": [list(m) for m in messages], "model": model or {}}
    commit = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    data = self._load(name)
    if commit not in data["commits"]:
      data["commits"][commit] = {**body, "note": note, "pushed_at": time.time()}
      data["order"].append(commit)
      self._save(name, data)
    return commit

  def tag(self, name: str, tag: str, commit: str) -> str | None:
    """Point a tag at a commit; returns the commit it pointed at before (what a rollback goes back to)."""
    data = self._load(name)
    if commit not in data["commits"]:
      raise KeyError(f"{name} has no commit {commit}")
    previous = data["tags"].get(tag)
    data["tags"][tag] = commit
    self._save(name, data)
    self._cache.clear()
    return previous

  def pull(self, ref: str, skip_cache: bool = False) -> PromptVersion:
    """'name' (latest), 'name:<tag>' or 'name:<commit>'. Cached for cache_s seconds, like the Hub's client."""
    cached = self._cache.get(ref)
    if cached and not skip_cache and time.time() - cached[0] < self.cache_s:
      return cached[1]
    name, _, pointer = ref.partition(":")
    data = self._load(name)
    commit = data["tags"].get(pointer, pointer) if pointer else data["order"][-1]
    entry = data["commits"][commit]
    version = PromptVersion(name, commit, tuple(tuple(m) for m in entry["messages"]), entry["model"])
    self._cache[ref] = (time.time(), version)
    return version

  def history(self, name: str) -> list[dict]:
    data = self._load(name)
    tags = {c: t for t, c in data["tags"].items()}
    return [{"commit": c, "note": data["commits"][c]["note"], "tag": tags.get(c, "")} for c in data["order"]]
