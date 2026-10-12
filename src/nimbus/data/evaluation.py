"""Execution accuracy for text to SQL (Data track): run the system's SQL and a trusted query, compare RESULTS.

Many different queries are right, so the SQL text is never compared. A result is right when it contains every value
of the trusted query's first row; `strict` also refuses the harmless differences (25.44 vs 0.2544, '0' vs 0), so the
two scores can be read side by side.

  rows = evaluate(lambda question, config: system.ask(question, config=config), load_questions())
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from nimbus.config import PROJECT_DIR
from nimbus.data.sql import matches, query
from nimbus.llm.usage import UsageRecorder

QUESTIONS = PROJECT_DIR / "data" / "eval" / "sql_questions.json"


def load_questions(kind: str = "questions") -> list[dict]:
  """kind: "questions" (one value each) or "hard" (several queries or a looked-up fact)."""
  return json.loads(QUESTIONS.read_text())[kind]


@cache
def right_row(trusted_sql: str) -> tuple:
  return tuple(query(trusted_sql)[1][0])


def right_answer(trusted_sql: str):
  return right_row(trusted_sql)[0]


@dataclass
class Graded:
  question: dict
  sql: str | None
  answer: str
  rows: list[tuple] = field(default_factory=list)
  correct: bool = False
  strict: bool = False
  attempts: int = 0
  calls: int = 0
  tokens: int = 0
  seconds: float = 0.0


def grade(question: dict, result, recorder: UsageRecorder, seconds: float) -> Graded:
  """result: anything with .sql, .rows, .answer and .attempts. Right when EVERY value of the trusted row appears in
  the rows the system saw (an agent's rows from all its queries)."""
  right = right_row(question["sql"])
  usage = recorder.usage()
  return Graded(
    question,
    result.sql,
    result.answer,
    result.rows,
    correct=result.sql is not None and all(matches(v, result.rows) for v in right),
    strict=result.sql is not None and all(matches(v, result.rows, normalize=False) for v in right),
    attempts=len(result.attempts),
    calls=usage.calls,
    tokens=usage.input_tokens + usage.output_tokens,
    seconds=seconds,
  )


def evaluate(ask: Callable, questions: list[dict]) -> list[Graded]:
  """ask(question, config) → a result; config carries the usage recorder."""
  graded = []
  for question in questions:
    recorder = UsageRecorder()
    started = time.perf_counter()
    result = ask(question["question"], {"callbacks": [recorder]})
    graded.append(grade(question, result, recorder, time.perf_counter() - started))
  return graded


def summary(name: str, graded: list[Graded]) -> dict:
  n = max(len(graded), 1)
  seconds = sorted(g.seconds for g in graded)
  return {
    "variant": name,
    "correct": round(sum(g.correct for g in graded) / n, 3),
    "strict": round(sum(g.strict for g in graded) / n, 3),
    "failed": sum(g.sql is None for g in graded),
    "attempts": round(sum(g.attempts for g in graded) / n, 2),
    "calls": round(sum(g.calls for g in graded) / n, 2),
    "tokens": round(sum(g.tokens for g in graded) / n),
    "latency_p50_ms": round(seconds[len(seconds) // 2] * 1000) if seconds else 0,
  }
