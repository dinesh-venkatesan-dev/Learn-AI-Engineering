"""Assistant evaluation: questions for one specialist, for two at once, and small talk, through each architecture.

correct      the reply contains what the question needs (phrases, all-of phrases, whole numbers, a pattern)
specialists  how many specialists (or tools, for the single agent) were used
calls        model calls, counted by a callback on the whole run (specialists included)
"""

import json
import re
import time
from pathlib import Path

from nimbus.llm.usage import UsageRecorder


def load_questions(path: Path) -> list[dict]:
  return json.loads(path.read_text())


def correct(question: dict, reply: str, used: list[str]) -> bool:
  text = reply.casefold().replace("’", "'")
  if question["kind"] == "small talk":
    return bool(text.strip()) and not used  # nothing to look up: answer without asking anyone
  numbers = {int(n) for n in re.findall(r"\d+", text.replace(",", ""))}
  return (
    (not question.get("answer_any") or any(a in text for a in question["answer_any"]))
    and all(a in text for a in question.get("answer_all", []))
    and all(n in numbers for n in question.get("answer_numbers", []))
    and (not question.get("answer_regex") or re.search(question["answer_regex"], text) is not None)
  )


def run_question(ask, question: dict) -> dict:
  recorder = UsageRecorder()
  started = time.perf_counter()
  try:
    result = ask(question["customer_id"], question["question"], {"callbacks": [recorder]})
    reply, used, error = result.reply, result.specialists, None
  except Exception as exc:  # noqa: BLE001 — a failed question is a result
    reply, used, error = "", [], f"{type(exc).__name__}: {exc}"[:200]
  usage = recorder.usage()
  return {
    "reply": reply,
    "used": used,
    "error": error,
    "ok": error is None and correct(question, reply, used),
    "calls": len(recorder.calls),
    "tokens": usage.input_tokens + usage.output_tokens,
    "ms": (time.perf_counter() - started) * 1000,
  }
