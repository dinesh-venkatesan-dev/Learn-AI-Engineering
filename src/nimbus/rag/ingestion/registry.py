"""The document registry: facts about documents that aren't written inside them.

Which version of a policy is in force, when a notice expires, who may read a file: a content system knows these,
the file doesn't. data/nimbus/.documents.json plays that role here. Ingestion copies each document's entry into
the metadata of all its chunks, so the retriever can filter on it (for example: only what is valid today).
"""

import json
from pathlib import Path

from langchain_core.documents import Document

BEGINNING, FOREVER = "0001-01-01", "9999-12-31"  # ISO dates compare correctly as text


def load_registry(data_dir: Path) -> dict[str, dict]:
  path = data_dir / ".documents.json"
  if not path.exists():
    return {}
  return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def apply_registry(documents: list[Document], registry: dict[str, dict]) -> list[Document]:
  """Every Document gets valid_from / valid_to (defaults: always valid) plus anything else its entry says."""
  for doc in documents:
    entry = registry.get(doc.metadata["source"], {})
    doc.metadata["valid_from"] = entry.get("valid_from", BEGINNING)
    doc.metadata["valid_to"] = entry.get("valid_to", FOREVER)
    doc.metadata["access"] = entry.get("access", "staff")  # deny by default: only documents marked public are public
  return documents


AUDIENCES = {"public": ["public"], "staff": ["public", "staff"]}  # who may read which access levels


def valid_on(day: str, audience: str = "staff") -> dict:
  """A vector-store filter: documents in force on `day` (ISO date) that `audience` may read."""
  return {
    "$and": [
      {"valid_from": {"$lte": day}},
      {"valid_to": {"$gte": day}},
      {"access": {"$in": AUDIENCES[audience]}},
    ]
  }
