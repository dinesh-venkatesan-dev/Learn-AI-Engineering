"""RAG quality: synthetic evaluation questions and answer judges.

Synthetic questions (written once, saved, reused):
    questions = generate_questions(chunks, model)      # each from one chunk, with the quote that answers it
    save_questions(questions, path)                      # every later evaluation runs on exactly these

Judging an answer (one call per question, on a different model than the one answering):
    verdict = judge(judge_model, question, answer, sources, reference)
    verdict.faithfulness   share of the answer's claims supported by the sources (hallucination check)
    verdict.relevant       does the answer address the question?
    verdict.correct        does it agree with the reference answer?
    verdict.context_precision  share of the retrieved sources that were useful
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


class SyntheticQuestion(BaseModel):
  question: str = Field(description="A question a customer or employee would really ask, answered by the text")
  answer: str = Field(description="The short answer, from the text")
  evidence: str = Field(description="The exact words from the text that answer it, copied verbatim (one sentence)")


class Critique(BaseModel):
  specific: bool = Field(description="true if the question is specific and answerable without seeing the text")
  reason: str


@dataclass
class EvalItem:
  question: str
  answer: str
  evidence: str
  source: str


def _norm(text: str) -> str:
  text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
  return re.sub(r"\s+", " ", re.sub(r"[|*#`]", " ", text)).casefold().strip().rstrip(".")


def copies_the_text(question: str, chunk: str, limit: float = 0.8) -> bool:
  """True if most of the question's words come straight from the chunk (a lookup, not a real question)."""
  words = re.findall(r"[a-z0-9]+", question.casefold())
  text = set(re.findall(r"[a-z0-9]+", chunk.casefold()))
  return bool(words) and sum(w in text for w in words) / len(words) > limit


QUESTION_PROMPT = (
  "Write one realistic question that this text fully answers, the way a customer or an employee of Nimbus Coffee "
  "would ask it (in your own words, not the text's). Then give the short answer, and the evidence: a short exact "
  "quote from the text (a few words up to one sentence) that contains the answer.\n\nText:\n{text}"
)
CRITIC_PROMPT = (
  "Is this question specific and self-contained: does it ask one clear thing, without referring to 'this text', "
  "'the document' or 'above'? Questions about Nimbus Coffee's own policies, products and numbers are fine.\n\n"
  "{question}"
)


def generate_questions(chunks: list[Document], model: BaseChatModel) -> tuple[list[EvalItem], list[dict]]:
  """(kept, rejected): one question per chunk, kept only if its evidence is really in the chunk, it doesn't copy
  the chunk, and a critic finds it specific."""
  writer = model.with_structured_output(SyntheticQuestion, method="json_schema")
  critic = model.with_structured_output(Critique, method="json_schema")
  kept, rejected = [], []
  for chunk in chunks:
    q = writer.invoke(QUESTION_PROMPT.format(text=chunk.page_content))
    source = chunk.metadata["source"]
    if _norm(q.evidence) not in _norm(chunk.page_content):
      rejected.append({"question": q.question, "source": source, "reason": "evidence not in the chunk"})
      continue
    if copies_the_text(q.question, chunk.page_content):
      rejected.append({"question": q.question, "source": source, "reason": "copies the text's words"})
      continue
    verdict = critic.invoke(CRITIC_PROMPT.format(question=q.question))
    if not verdict.specific:
      rejected.append({"question": q.question, "source": source, "reason": f"vague: {verdict.reason}"})
      continue
    kept.append(EvalItem(q.question, q.answer, q.evidence, source))
  return kept, rejected


def save_questions(items: list[EvalItem], path: Path) -> None:
  path.write_text(json.dumps([asdict(i) for i in items], indent=1, ensure_ascii=False) + "\n")


def load_synthetic(path: Path) -> list[EvalItem]:
  return [EvalItem(**i) for i in json.loads(path.read_text())]


class Verdict(BaseModel):
  claims: list[str] = Field(description="Each factual claim the answer makes")
  unsupported: list[str] = Field(description="Claims NOT supported by the sources")
  relevant: bool = Field(description="Does the answer address the question?")
  correct: bool = Field(description="Does the answer agree with the reference answer? (true if no reference)")
  useful_sources: list[int] = Field(description="Numbers of the sources that help answer the question")

  @property
  def faithfulness(self) -> float:
    return 1.0 if not self.claims else 1 - len(self.unsupported) / len(self.claims)


def judge_models(settings=None) -> list[BaseChatModel]:
  """The judges, best first: when one model's daily quota runs out, the next one takes over."""
  from nimbus.config import get_settings
  from nimbus.llm import get_chat_model

  s = settings or get_settings()
  return [get_chat_model(s, role="reasoning", model=name, max_retries=1) for name in s.judge_models]


def judge(
  model: BaseChatModel | list[BaseChatModel],
  question: str,
  answer: str,
  sources: str,
  n_sources: int,
  reference: str | None,
  rubric: str = "",
) -> Verdict:
  """All four judgements in one structured call (cheaper than four, and they share the same reading). `model` can
  be a list (judge_models()): each is tried in turn. `rubric` adds task-specific grading rules."""
  # Judges occasionally return invalid JSON (Groq answers 400 json_validate_failed, even three times running):
  # retry, then fall back to tool-calling structured output, which the provider validates differently
  attempts = [
    m.with_structured_output(Verdict, method=method).with_retry(stop_after_attempt=2)
    for m in (model if isinstance(model, list) else [model])
    for method in ("json_schema", "function_calling")
  ]
  structured = attempts[0].with_fallbacks(attempts[1:])
  return structured.invoke(
    f"You are grading a question-answering system.\n\nQuestion: {question}\n\nReference answer: {reference or '(none)'}"
    f"\n\nSources given to the system ({n_sources}):\n{sources}\n\nThe system's answer:\n{answer}\n\n"
    "List the answer's claims and those the sources don't support; say whether it addresses the question and "
    "agrees with the reference; list which sources are useful for this question." + (f"\n{rubric}" if rubric else "")
  )
