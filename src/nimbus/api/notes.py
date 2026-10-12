"""What the notes endpoints do besides storing: text from uploads, summaries written in the background, which notes
a chat question is about, and links removed from the model's answers (a note can carry instructions to leak data
through a link or an image)."""

import logging
import re

from langchain_core.language_models import BaseChatModel
from sqlmodel import Session, select

from nimbus.api.models import Note

log = logging.getLogger("nimbus.api.notes")
URL = re.compile(r"!?\[[^\]]*\]\(\s*https?://[^)]*\)|https?://\S+")  # markdown links/images and bare URLs
TEXT_TYPES = {"text/plain", "text/markdown"}
UPLOAD_TYPES = TEXT_TYPES | {"application/pdf"}


def extract_text(content_type: str, data: bytes) -> str:
  """PDFs opened from the bytes in memory (and closed afterwards); text files decoded."""
  if content_type == "application/pdf":
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as document:
      return "\n".join(page.get_text() for page in document).strip()
  return data.decode("utf-8", errors="replace").strip()


async def summarize(model: BaseChatModel, text: str) -> str:
  reply = await model.ainvoke(
    [("system", "Summarise the note in one sentence of at most 25 words."), ("human", text[:6000])]
  )
  return reply.text.strip()


async def summarize_in_background(engine, model: BaseChatModel, note_id: int) -> None:
  """Runs AFTER the response is sent. A failure here can't change the response, so it's logged."""
  try:
    with Session(engine) as session:
      note = session.get(Note, note_id)
      if note is None:
        return
      note.summary = await summarize(model, note.content)
      session.add(note)
      session.commit()
  except Exception:
    log.exception("summary failed for note %s", note_id)


STOP = {"the", "and", "can", "for", "how", "what", "when", "who", "why", "are", "our", "you", "your", "does", "get"}


def keywords(text: str) -> set[str]:
  """Words that say what a text is about: no stop words, plural s dropped ("refunds" matches "refund")."""
  words = re.findall(r"[a-z0-9]+", text.lower())
  return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if len(w) > 2 and w not in STOP}


def relevant_notes(session: Session, owner_id: int, question: str, limit: int = 3) -> list[Note]:
  """The user's notes that share the most keywords with the question (newest first on ties)."""
  wanted = keywords(question)
  notes = session.exec(select(Note).where(Note.owner_id == owner_id).order_by(Note.id.desc())).all()
  scored = [(len(wanted & keywords(f"{n.title} {n.content}")), n) for n in notes]
  return [n for score, n in sorted(scored, key=lambda pair: -pair[0]) if score > 0][:limit]


def notes_prompt(question: str, notes: list[Note], history: list[tuple[str, str]] | None = None) -> list:
  context = "\n\n".join(f"[{n.title}]\n{n.content[:2000]}" for n in notes) or "(no matching notes)"
  return [
    (
      "system",
      "Answer ONLY from the user's notes below; if they don't say, say you don't know. Name the note you used. "
      f"At most three sentences.\n\nNotes:\n{context}",
    ),
    *(history or []),
    ("human", question),
  ]


def strip_links(text: str) -> str:
  """Model output is untrusted: a link or an image URL in an answer can carry the user's data to someone else."""
  return URL.sub("[link removed]", text)


class LinkFilter:
  """strip_links for a token stream: text is released up to the last whitespace, so a URL is always seen whole."""

  def __init__(self) -> None:
    self.buffer = ""

  def feed(self, token: str) -> str:
    self.buffer += token
    cut = max(self.buffer.rfind(" "), self.buffer.rfind("\n"))
    if cut < 0 or "](" in self.buffer[cut:]:  # wait for the whole word (and a markdown link's closing bracket)
      return ""
    ready, self.buffer = self.buffer[: cut + 1], self.buffer[cut + 1 :]
    return strip_links(ready)

  def flush(self) -> str:
    ready, self.buffer = self.buffer, ""
    return strip_links(ready)
