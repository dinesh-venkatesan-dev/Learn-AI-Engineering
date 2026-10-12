"""Tracing Nimbus with LangSmith, safely.

guard_uploads(settings)   at start-up: without a key configured for Nimbus, NO trace can leave the machine
                          (the SDK's endpoint is pinned to a local address), whatever else the environment says
tracing_client(settings)  the Client every trace goes through: PII and secrets anonymized, sampling applied
pii_anonymizer()          regex rules for emails, phone numbers and card numbers, plus the SDK's secret rules
run_config(...)           config for one invocation: a run id chosen up front (for feedback), the thread id
                          (conversations), tags and metadata copied to every run in the trace
"""

import os
import uuid

from langsmith import Client
from langsmith.anonymizer import create_anonymizer

from nimbus.config import Settings

LOCAL_ONLY = "http://127.0.0.1:9"  # nothing listens here: an upload attempt fails on this machine
PII_RULES = [
  {"pattern": r"[\w.+-]+@[\w-]+\.[\w.]+", "replace": "<email>"},
  {"pattern": r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b", "replace": "<phone>"},
  {"pattern": r"\b(?:\d[ -]?){13,16}\b", "replace": "<card>"},
  {"pattern": r"\bsk-[A-Za-z0-9_-]{8,}\b", "replace": "<secret>"},
  {"pattern": r"\b(?:gsk|lsv2_pt|lsv2_sk)_[A-Za-z0-9_]{8,}\b", "replace": "<secret>"},
]


def langsmith_ready(settings: Settings) -> bool:
  return settings.langsmith_tracing and settings.langsmith_api_key is not None


def guard_uploads(settings: Settings) -> bool:
  """Call before any LangSmith client exists. Returns True when traces may go to LangSmith."""
  if langsmith_ready(settings):
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"
    return True
  os.environ["LANGSMITH_ENDPOINT"] = LOCAL_ONLY
  os.environ["LANGSMITH_TRACING"] = "false"
  return False


def pii_anonymizer():
  return create_anonymizer(PII_RULES)  # applied to inputs, outputs and messages before anything is sent


def tracing_client(settings: Settings, api_url: str | None = None, **overrides) -> Client:
  """api_url: where traces go; a TraceCapture's url in tests and labs."""
  key = settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else "local-only"
  return Client(
    api_url=api_url or (settings.langsmith_endpoint if langsmith_ready(settings) else LOCAL_ONLY),
    api_key=key,
    anonymizer=overrides.pop("anonymizer", pii_anonymizer()),
    tracing_sampling_rate=overrides.pop("tracing_sampling_rate", settings.langsmith_sampling_rate),
    **overrides,
  )


def run_config(
  run_name: str, *, thread_id: str | None = None, tags: list[str] | None = None, **metadata
) -> tuple[str, dict]:
  """(run_id, config). Keep the run id: a user's rating of this answer points at it."""
  run_id = uuid.uuid4()
  meta = {**metadata, **({"thread_id": thread_id} if thread_id else {})}
  return str(run_id), {"run_id": run_id, "run_name": run_name, "tags": tags or [], "metadata": meta}
