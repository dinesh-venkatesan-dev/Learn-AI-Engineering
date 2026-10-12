"""Tool pinning: load an MCP server's tools only if they are exactly the ones a person reviewed (MCP Module 4).

A tool's name, description and argument schema are model input. A server can change them at any time (a "rug pull"),
and a description can hide instructions for the model (tool poisoning). So after reviewing a server's tools we store
a fingerprint of each one, in the repo next to the config, and an agent loads only tools whose fingerprint matches.

  approved = fingerprints(tools)                  after a review; save with save_pins()
  safe, blocked = pinned(tools, load_pins())      at start-up: blocked tools need a new review
"""

import hashlib
import json
from pathlib import Path

from nimbus.config import PROJECT_DIR

PINS = PROJECT_DIR / "data" / "mcp" / "tool_pins.json"


def fingerprint(tool) -> str:
  """A hash of everything the model reads about a LangChain tool from MCPAdapter: name, description, arguments."""
  contract = {"name": tool.name, "description": tool.description, "args": tool.args_schema}
  return hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fingerprints(tools) -> dict[str, str]:
  return {t.name: fingerprint(t) for t in tools}


def pinned(tools, approved: dict[str, str]) -> tuple[list, list[str]]:
  """The tools that match their reviewed fingerprint, and the names of the ones that don't (new or changed)."""
  safe, blocked = [], []
  for tool in tools:
    (safe if approved.get(tool.name) == fingerprint(tool) else blocked).append(tool)
  return safe, [t.name for t in blocked]


def save_pins(approved: dict[str, str], path: Path = PINS) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(approved, indent=1, sort_keys=True) + "\n")


def load_pins(path: Path = PINS) -> dict[str, str]:
  return json.loads(path.read_text()) if path.exists() else {}
