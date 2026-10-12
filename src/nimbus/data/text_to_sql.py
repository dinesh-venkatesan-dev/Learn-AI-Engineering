"""Text to SQL as it ships (Data Module 1): question → SQL → guarded run → rows → answer, with self-correction.

  write    the model writes {"sql": ...} from the schema, the glossary and examples (structured output)
  run      the guard checks and runs it read-only (nimbus.data.guard)
  fix      on an error or a refusal, the model sees (question, SQL, error) and writes a new query; a few times at most
  answer   the model explains the rows; with no working query, an honest "I couldn't answer that", and with no rows
           "No matching data was found", both without a model call
Every attempt is kept on the answer and logged.

  answer = TextToSQL(model).ask("Which café had the highest revenue last month?")
"""

import logging
import sqlite3
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from nimbus.data.context import EXAMPLES, GLOSSARY, sql_instructions
from nimbus.data.guard import UnsafeQuery, run_safe
from nimbus.data.sql import schema_text

log = logging.getLogger(__name__)


class SQLQuery(BaseModel):
  sql: str = Field(description="One SQLite SELECT statement")


@dataclass
class Attempt:
  sql: str
  error: str | None = None


@dataclass
class SQLAnswer:
  question: str
  answer: str
  sql: str | None = None
  columns: list[str] = field(default_factory=list)
  rows: list[tuple] = field(default_factory=list)
  attempts: list[Attempt] = field(default_factory=list)

  @property
  def ok(self) -> bool:
    return self.sql is not None


class TextToSQL:
  def __init__(
    self,
    model: BaseChatModel,
    *,
    glossary: str = GLOSSARY,
    examples: str = EXAMPLES,
    sample_rows: int = 3,
    max_attempts: int = 3,
  ):
    self.model = model
    self.write_sql = model.with_structured_output(SQLQuery, method="json_schema")
    self.instructions = sql_instructions(schema_text(sample_rows=sample_rows), glossary, examples)
    self.glossary = glossary
    self.max_attempts = max_attempts

  def write(self, question: str, config=None) -> str:
    return self.write_sql.invoke([SystemMessage(self.instructions), HumanMessage(question)], config).sql

  def fix(self, question: str, sql: str, error: str, config=None) -> str:
    """The error is feedback: a wrong dialect's function, a missing column, a refused column."""
    prompt = (
      f"Question: {question}\nPrevious query: {sql}\nError: {error}\n\nThe previous query failed. Write a corrected "
      "query that answers the question. If the error says some data isn't available, answer as much of the question "
      "as the allowed data can."
    )
    return self.write_sql.invoke([SystemMessage(self.instructions), HumanMessage(prompt)], config).sql

  def ask(self, question: str, first_sql: str | None = None, config=None) -> SQLAnswer:
    """first_sql: start from a given query (labs use it to show a failing first attempt)."""
    result = SQLAnswer(question, answer="")
    sql = first_sql or self.write(question, config)
    for attempt in range(1, self.max_attempts + 1):
      try:
        columns, rows = run_safe(sql)
      except (UnsafeQuery, sqlite3.Error) as error:
        result.attempts.append(Attempt(sql, str(error)))
        log.info("text-to-sql attempt %d failed: %s", attempt, error)
        if attempt < self.max_attempts:
          sql = self.fix(question, sql, str(error), config)
        continue
      result.attempts.append(Attempt(sql))
      result.sql, result.columns, result.rows = sql, columns, rows
      break
    if not result.ok:
      result.answer = f"I couldn't answer that: {result.attempts[-1].error}"
      return result
    if not result.rows:  # decided in code: a small model told "if empty, say so" says it about full results too
      result.answer = "No matching data was found."
      return result
    reply = self.model.invoke(
      [
        SystemMessage("Answer the question in one or two sentences using ONLY the query result. Money is in rupees."),
        HumanMessage(f"Question: {question}\nSQL: {sql}\nColumns: {columns}\nRows: {rows[:50]}"),
      ],
      config,
    )
    result.answer = reply.text
    return result
