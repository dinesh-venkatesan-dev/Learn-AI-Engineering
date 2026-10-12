"""Step 1 — Load: find the supported files and turn each one into LangChain Documents.

A Document is text (`page_content`) plus `metadata`. The metadata recorded here (source, page, file type) travels with
every chunk into the database, so answers can always say where a fact came from.
"""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pymupdf4llm
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from nimbus.rag.ingestion.ocr import Engine, ImageLoader, ocr_pages, scanned_pages
from nimbus.rag.ingestion.office import DocxLoader, EmlLoader, PptxLoader, XlsxLoader

log = logging.getLogger(__name__)

# v1 (RAG Module 1): text formats and PDFs. v2 (RAG Module 2): every format the company stores, except images.
FILE_TYPES = {".pdf": "pdf", ".md": "markdown", ".txt": "text", ".html": "html", ".htm": "html"}
FILE_TYPES_V2 = FILE_TYPES | {
  ".csv": "csv",
  ".json": "json",
  ".py": "code",
  ".docx": "docx",
  ".xlsx": "xlsx",
  ".pptx": "pptx",
  ".eml": "eml",
}
# v3 (RAG Module 3): + images, and OCR for scanned PDF pages.
FILE_TYPES_V3 = FILE_TYPES_V2 | {".png": "image", ".jpg": "image", ".jpeg": "image"}
VERSIONS = {"v1": FILE_TYPES, "v2": FILE_TYPES_V2, "v3": FILE_TYPES_V3}


class PdfLoader(BaseLoader):
  """One Document per page, as Markdown, so headings and tables keep their structure.

  With an OCR engine, pages without a text layer (scans) are rendered and OCR'd: a hybrid loader."""

  def __init__(self, path: Path, source: str, ocr: Engine | None = None):
    self.path, self.source, self.ocr = path, source, ocr

  def lazy_load(self) -> Iterator[Document]:
    scanned = set(scanned_pages(self.path)) if self.ocr else set()
    ocr_text = ocr_pages(self.path, sorted(scanned), self.ocr) if scanned else {}
    # use_ocr=False: PyMuPDF4LLM reads the text layer; scanned pages are OCR'd above with our engine and cache.
    for n, page in enumerate(pymupdf4llm.to_markdown(str(self.path), page_chunks=True, use_ocr=False)):
      meta = {"source": self.source, "page": page["metadata"]["page_number"], "file_type": "pdf"}
      if n in ocr_text:
        yield Document(ocr_text[n], metadata={**meta, "extraction": "ocr", "ocr_engine": self.ocr})
      else:
        yield Document(page["text"], metadata=meta)


class TextFileLoader(BaseLoader):
  """Markdown, plain text, HTML and code: one Document per file. HTML stays raw here; cleaning extracts its text."""

  def __init__(self, path: Path, source: str, file_type: str):
    self.path, self.source, self.file_type = path, source, file_type

  def lazy_load(self) -> Iterator[Document]:
    text = self.path.read_text(encoding="utf-8", errors="replace")  # a stray bad byte must not fail the file
    yield Document(text, metadata={"source": self.source, "page": 1, "file_type": self.file_type})


class CsvLoader(BaseLoader):
  """One Document per row, every value labelled with its column: "item: Oat Latte\nprice_rupees: 220"."""

  def __init__(self, path: Path, source: str):
    self.path, self.source = path, source

  def lazy_load(self) -> Iterator[Document]:
    with self.path.open(newline="", encoding="utf-8") as f:
      for number, row in enumerate(csv.DictReader(f), start=1):
        text = "\n".join(f"{k}: {v}" for k, v in row.items())
        yield Document(text, metadata={"source": self.source, "page": number, "file_type": "csv", "row": number})


def record_text(record: dict) -> str:
  """A JSON record as readable lines (lists joined, nested objects flattened): what a person would write."""
  lines = []
  for key, value in record.items():
    if isinstance(value, list):
      value = "; ".join(", ".join(f"{k}: {x}" for k, x in v.items()) if isinstance(v, dict) else str(v) for v in value)
    elif isinstance(value, dict):
      value = "; ".join(f"{k}: {v}" for k, v in value.items())
    lines.append(f"{key.replace('_', ' ')}: {value}")
  return "\n".join(lines)


class JsonLoader(BaseLoader):
  """A list of records → one Document per record; any other JSON → one Document."""

  def __init__(self, path: Path, source: str):
    self.path, self.source = path, source

  def lazy_load(self) -> Iterator[Document]:
    data = json.loads(self.path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else [data]
    for number, record in enumerate(records, start=1):
      text = record_text(record) if isinstance(record, dict) else json.dumps(record, ensure_ascii=False)
      yield Document(text, metadata={"source": self.source, "page": number, "file_type": "json", "record": number})


def discover(folder: Path, version: str = "v2") -> tuple[list[Path], list[Path]]:
  """(supported, unsupported) files under `folder`, recursively, in a stable order. Hidden files are skipped.

  Unsupported files are returned, not silently ignored: a file nobody can search is a gap worth reporting.
  """
  files = sorted(
    p
    for p in folder.rglob("*")
    if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)
  )
  types = VERSIONS[version]
  return (
    [p for p in files if p.suffix.lower() in types],
    [p for p in files if p.suffix.lower() not in types],
  )


def loader_for(path: Path, root: Path, ocr: Engine | None = None) -> BaseLoader:
  """`ocr`: None (v1, v2: no OCR), "tesseract" or "vision" (v3: images and scanned PDF pages are read)."""
  # The source id is the path relative to the data folder: stable across machines, unique within the corpus.
  source = path.relative_to(root).as_posix()
  file_type = FILE_TYPES_V3[path.suffix.lower()]
  if file_type == "pdf":
    return PdfLoader(path, source, ocr)
  if file_type == "image":
    return ImageLoader(path, source, ocr or "tesseract")
  structured = {
    "csv": CsvLoader,
    "json": JsonLoader,
    "docx": DocxLoader,
    "xlsx": XlsxLoader,
    "pptx": PptxLoader,
    "eml": EmlLoader,
  }
  if file_type in structured:
    return structured[file_type](path, source)
  return TextFileLoader(path, source, file_type)  # markdown, text, HTML and code
