"""Evaluating the support reply prompt (Production Module 4): a dataset with splits, code checks, an LLM judge, and
experiments compared example by example.

  examples = load_examples(split="smoke")
  outputs = answer_all(prompt_ref, examples)                    model calls happen HERE, outside evaluate()
  results = run_experiment(outputs, examples, [facts, no_promises])   langsmith.evaluate, nothing uploaded
  regressions(baseline, candidate)                              examples the baseline got right and the candidate not
"""

import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import evaluate
from langsmith.schemas import Example

from nimbus.config import PROJECT_DIR

DATASET = PROJECT_DIR / "data" / "eval" / "support_reply_dataset.json"


def load_examples(split: str | None = None) -> list[dict]:
  examples = json.loads(DATASET.read_text())["examples"]
  return [e for e in examples if split is None or e["split"] == split]


def as_example(e: dict) -> Example:
  import uuid
  from datetime import UTC, datetime

  return Example(
    id=uuid.uuid5(uuid.NAMESPACE_URL, e["id"]),
    inputs=e["inputs"],
    outputs=e["reference_outputs"],
    metadata={**e["metadata"], "split": e["split"], "example_id": e["id"]},
    dataset_id=uuid.uuid5(uuid.NAMESPACE_URL, "support-reply"),
    created_at=datetime.now(UTC),
  )


def answer_all(prompt_ref: str, examples: list[dict], model=None) -> dict[str, dict]:
  """The app under test: the support reply with a given prompt version. Returns outputs by example id."""
  from nimbus.api.routers.support import system_prompt
  from nimbus.api.schemas import ReplyRequest
  from nimbus.llm import get_chat_model

  model = model or get_chat_model().with_retry(stop_after_attempt=4, wait_exponential_jitter=True)
  outputs = {}
  for e in examples:
    body = ReplyRequest(message=e["inputs"]["message"], tone=e["inputs"].get("tone", "warm"))
    started = time.perf_counter()
    reply = model.invoke([SystemMessage(system_prompt(body, prompt_ref)), HumanMessage(body.message)])
    usage = reply.usage_metadata or {}
    outputs[e["id"]] = {
      "answer": reply.text,
      "ms": (time.perf_counter() - started) * 1000,
      "tokens": usage.get("total_tokens", 0),
    }
  return outputs


# ── code evaluators: plain functions, (outputs, reference_outputs) → score ──
def facts(outputs: dict, reference_outputs: dict) -> dict:
  found = all(re.search(p, outputs["answer"]) for p in reference_outputs["must_contain"])
  return {"key": "facts", "score": int(found)}


def no_promises(outputs: dict, reference_outputs: dict) -> dict:
  promised = re.search(reference_outputs["must_not"], outputs["answer"]) if reference_outputs.get("must_not") else None
  return {"key": "no_promises", "score": int(promised is None), "comment": promised.group() if promised else ""}


def length(outputs: dict, reference_outputs: dict) -> list[dict]:
  words = len(outputs["answer"].split())
  return [{"key": "words", "score": words}, {"key": "short_enough", "score": int(words <= 75)}]  # several scores


def tone_label(outputs: dict) -> dict:
  return {
    "key": "opens_with",
    "value": "apology" if re.match(r"(?i)\W*(sorry|we're sorry|apolog)", outputs["answer"]) else "other",
  }


CODE_EVALUATORS = [facts, no_promises, length]


def run_experiment(outputs: dict[str, dict], examples: list[dict], evaluators, prefix: str = "support-reply"):
  """langsmith.evaluate on outputs computed beforehand (a pure-Python target): nothing is uploaded."""

  def target(inputs: dict) -> dict:
    return next(outputs[e["id"]] for e in examples if e["inputs"] == inputs)

  return evaluate(
    target,
    data=[as_example(e) for e in examples],
    evaluators=evaluators,
    upload_results=False,
    experiment_prefix=prefix,
    max_concurrency=4,
  )


def scores(results) -> dict[str, dict[str, float]]:
  """{example id: {evaluator key: score}} from an experiment's results."""
  out = {}
  for row in results:
    example_id = row["example"].metadata["example_id"]
    out[example_id] = {
      r.key: (r.score if r.score is not None else r.value) for r in row["evaluation_results"]["results"]
    }
  return out


def averages(per_example: dict[str, dict]) -> dict[str, float]:
  keys = {k for s in per_example.values() for k, v in s.items() if isinstance(v, int | float)}
  return {k: round(sum(s.get(k, 0) for s in per_example.values()) / max(len(per_example), 1), 3) for k in sorted(keys)}


def regressions(baseline: dict, candidate: dict, keys=("facts", "no_promises")) -> list[tuple[str, str]]:
  """(example, check) pairs the baseline passed and the candidate failed: what a release must not break."""
  return [(e, k) for e in baseline for k in keys if baseline[e].get(k) == 1 and candidate.get(e, {}).get(k) == 0]


def gate(baseline: dict, candidate: dict, keys=("facts", "no_promises")) -> tuple[bool, list]:
  """Promote only if nothing got worse."""
  broken = regressions(baseline, candidate, keys)
  return not broken, broken
