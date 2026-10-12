"""OCR: text from images and from scanned PDF pages, with a local engine (Tesseract) or a vision model.

    text = read_image(image, engine="tesseract")      # free, local, private; weak on poor photos and charts
    text = read_image(image, engine="vision")         # reads photos, tables and charts; costs a request per page

Results are cached by the image's hash: OCR is slow (and a vision model costs a request), and the same scan is
ingested again and again. Every OCR'd Document records how its text was obtained (`extraction`, `ocr_engine`),
because OCR text can contain errors a reader should know about.
"""

import base64
import hashlib
import io
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pymupdf
import pytesseract
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from PIL import Image, ImageFilter, ImageOps

from nimbus.config import PROJECT_DIR

log = logging.getLogger(__name__)

Engine = Literal["tesseract", "vision"]
CACHE_DIR = PROJECT_DIR / ".cache" / "ocr"
TRANSCRIBE = (
  "Transcribe all text in this image exactly as written, as Markdown. Keep line order; use a Markdown table for "
  "tabular rows; for a chart, write its title and each label with its value as a table. Do not add, fix or "
  "summarise anything. Reply with the transcription only: no introduction, no code fences."
)


def open_image(path: Path) -> Image.Image:
  """Load an image fully and close the file (Image.open alone keeps it open)."""
  with Image.open(path) as image:
    image.load()
    return image.copy()


def preprocess(image: Image.Image, enlarge: int = 3) -> Image.Image:
  """Enlarge → even out the lighting → stretch the contrast: what rescues a small, shadowed phone photo."""
  gray = ImageOps.grayscale(image)
  big = gray.resize((gray.width * enlarge, gray.height * enlarge), Image.Resampling.LANCZOS)
  background = np.array(big.filter(ImageFilter.GaussianBlur(25)), dtype=float)  # the lighting, without the text
  flat = np.clip(np.array(big, dtype=float) / np.maximum(background, 1) * 255, 0, 255).astype("uint8")
  return ImageOps.autocontrast(Image.fromarray(flat), cutoff=1)


def needs_preprocessing(image: Image.Image) -> bool:
  """Small or low-contrast images (phone photos, thumbnails) read far better after preprocessing."""
  lo, hi = ImageOps.grayscale(image).getextrema()
  return min(image.size) < 600 or hi - lo < 200


def image_message(image: Image.Image, instruction: str) -> HumanMessage:
  buffer = io.BytesIO()
  image.convert("RGB").save(buffer, format="PNG")
  return HumanMessage(
    content_blocks=[
      {"type": "text", "text": instruction},
      {"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode(), "mime_type": "image/png"},
    ]
  )


def _key(image: Image.Image, engine: str) -> str:
  return hashlib.sha256(image.tobytes() + f"{image.size}{engine}".encode()).hexdigest()[:24]


def read_image(image: Image.Image, engine: Engine = "tesseract", cache: bool = True) -> str:
  """The text of an image. Cached on disk by content hash + engine."""
  path = CACHE_DIR / f"{_key(image, engine)}.json"
  if cache and path.exists():
    return json.loads(path.read_text())["text"]
  if engine == "vision":
    from nimbus.llm import get_chat_model  # imported here: Tesseract-only ingestion needs no model client

    text = get_chat_model().invoke([image_message(image, TRANSCRIBE)]).text.strip()
    text = re.sub(r"^```[a-z]*\n|\n?```$", "", text).strip()  # models add code fences even when told not to
  else:
    text = pytesseract.image_to_string(preprocess(image) if needs_preprocessing(image) else image)
  if cache:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"engine": engine, "text": text}))
  return text


def render(page: pymupdf.Page, dpi: int = 300) -> Image.Image:
  """A PDF page drawn as an image; 300 dpi is the usual resolution for OCR."""
  pixmap = page.get_pixmap(dpi=dpi)
  return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


class ImageLoader(BaseLoader):
  """An image file → one Document of OCR text."""

  def __init__(self, path: Path, source: str, engine: Engine = "tesseract"):
    self.path, self.source, self.engine = path, source, engine

  def lazy_load(self) -> Iterator[Document]:
    text = read_image(open_image(self.path), self.engine)
    meta = {"source": self.source, "page": 1, "file_type": "image", "extraction": "ocr", "ocr_engine": self.engine}
    yield Document(text, metadata=meta)


def scanned_pages(path: Path, min_chars: int = 20) -> list[int]:
  """Page numbers (0-based) with no usable text layer: they hold an image of text."""
  with pymupdf.open(path) as pdf:
    return [n for n, page in enumerate(pdf) if len(page.get_text().strip()) < min_chars]


def ocr_pages(path: Path, pages: list[int], engine: Engine = "tesseract") -> dict[int, str]:
  with pymupdf.open(path) as pdf:
    images = {n: render(pdf[n]) for n in pages}
  return {n: read_image(image, engine) for n, image in images.items()}
