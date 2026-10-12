"""Loaders for the formats every company is full of: Word, Excel, PowerPoint and email.

Each keeps the structure a reader relies on (sections, rows with their headers, slides with speaker notes, an email's
headers and attachments), because a chunk that lost its header or its section title can't answer the question.
"""

import csv
import io
import re
from collections.abc import Iterator
from email import policy
from email.parser import BytesParser
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from openpyxl import load_workbook
from pptx import Presentation


def markdown_table(rows: list[list[str]]) -> str:
  """Rows (the first is the header) as a Markdown table: every chunk of it still says what each column means."""
  header, *body = rows
  lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
  return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in body])


class DocxLoader(BaseLoader):
  """One Document per section (a Heading 1/2 starts one), tables as Markdown, the heading in metadata."""

  def __init__(self, path: Path, source: str):
    self.path, self.source = path, source

  def lazy_load(self) -> Iterator[Document]:
    docx = DocxDocument(str(self.path))
    paragraphs = {p._p: p for p in docx.paragraphs}
    tables = {t._tbl: t for t in docx.tables}
    title = next((p.text for p in docx.paragraphs if p.text.strip()), self.path.stem)
    heading, parts, number = "Introduction", [], 0
    for block in docx.element.body.iterchildren():  # paragraphs AND tables, in document order
      if block in tables:
        parts.append(markdown_table([[c.text.strip() for c in row.cells] for row in tables[block].rows]))
        continue
      paragraph = paragraphs.get(block)
      if paragraph is None or not paragraph.text.strip():
        continue
      style = paragraph.style.name if paragraph.style is not None else ""
      if style.startswith("Heading") or style == "Title":
        if parts:
          number += 1
          yield self._section(title, heading, parts, number)
        heading, parts = paragraph.text.strip(), []
      else:
        parts.append(paragraph.text.strip())
    if parts:
      yield self._section(title, heading, parts, number + 1)

  def _section(self, title: str, heading: str, parts: list[str], number: int) -> Document:
    text = f"## {heading}\n\n" + "\n\n".join(parts)  # the heading stays in the text: it's what the section is about
    meta = {"source": self.source, "page": number, "file_type": "docx", "title": title, "section": heading}
    return Document(text, metadata=meta)


def _sheet_rows(sheet) -> tuple[list[str], list[tuple]]:
  """(header, data rows). The header is the first row whose cells are all filled: title rows above it are skipped."""
  rows = [r for r in sheet.iter_rows(values_only=True) if any(v is not None for v in r)]
  at = next((i for i, r in enumerate(rows) if all(v is not None for v in r)), 0)
  return [str(v) for v in rows[at]], rows[at + 1 :]


class XlsxLoader(BaseLoader):
  """One Document per row of every sheet, each cell labelled with its column header."""

  def __init__(self, path: Path, source: str):
    self.path, self.source = path, source

  def lazy_load(self) -> Iterator[Document]:
    book = load_workbook(self.path, read_only=True, data_only=True)  # data_only: formula RESULTS, not formulas
    try:
      for sheet in book.worksheets:
        header, rows = _sheet_rows(sheet)
        for number, row in enumerate(rows, start=1):
          cells = "; ".join(f"{h}: {v}" for h, v in zip(header, row, strict=False) if v is not None)
          meta = {"source": self.source, "page": number, "file_type": "xlsx", "sheet": sheet.title, "row": number}
          yield Document(f"{sheet.title}: {cells}", metadata=meta)
    finally:
      book.close()  # a read-only workbook keeps the file open until closed


class PptxLoader(BaseLoader):
  """One Document per slide: title, text, tables and the speaker notes (often what the presenter actually said)."""

  def __init__(self, path: Path, source: str):
    self.path, self.source = path, source

  def lazy_load(self) -> Iterator[Document]:
    for number, slide in enumerate(Presentation(str(self.path)).slides, start=1):
      parts = []
      for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
          parts.append(shape.text_frame.text.strip())
        if shape.has_table:
          parts.append(markdown_table([[c.text.strip() for c in row.cells] for row in shape.table.rows]))
      notes = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
      if notes:
        parts.append(f"Speaker notes: {notes}")
      title = slide.shapes.title.text.strip() if slide.shapes.title is not None else ""
      meta = {"source": self.source, "page": number, "file_type": "pptx", "slide": number, "title": title}
      yield Document("\n\n".join(parts), metadata=meta)


QUOTED_REPLY = re.compile(r"\n+On .+ wrote:\n.*", re.DOTALL)  # "On <date>, <person> wrote:" and everything after


class EmlLoader(BaseLoader):
  """The new text of an email (quoted replies cut) plus each attachment, with the headers as metadata."""

  def __init__(self, path: Path, source: str):
    self.path, self.source = path, source

  def lazy_load(self) -> Iterator[Document]:
    mail = BytesParser(policy=policy.default).parsebytes(self.path.read_bytes())
    meta = {
      "source": self.source,
      "file_type": "eml",
      "from": str(mail["From"]),
      "to": str(mail["To"]),
      "subject": str(mail["Subject"]),
      "date": str(mail["Date"]),
    }
    body = mail.get_body(preferencelist=("plain", "html"))
    text = QUOTED_REPLY.sub("", body.get_content() if body is not None else "").strip()
    yield Document(f"Email: {mail['Subject']}\nFrom: {mail['From']}\n\n{text}", metadata={**meta, "page": 1})
    for number, attachment in enumerate(mail.iter_attachments(), start=2):
      name = attachment.get_filename() or f"attachment-{number}"
      content = attachment.get_content()
      if isinstance(content, bytes):
        continue  # binary attachments (images, PDFs) go through their own loaders in production
      if name.endswith(".csv"):
        content = "\n".join(
          "; ".join(f"{k}: {v}" for k, v in row.items()) for row in csv.DictReader(io.StringIO(content))
        )
      yield Document(
        f"Attachment {name} of the email '{mail['Subject']}':\n{content}",
        metadata={**meta, "page": number, "attachment": name},
      )
