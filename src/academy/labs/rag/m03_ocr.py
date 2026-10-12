"""RAG · Module 3 labs: OCR with Tesseract (preprocessing, layout, confidence), scanned PDFs, vision-model OCR,
structured extraction from document images, and layout-aware PDF parsing. Then: loaders v3 against v2."""

import base64
import difflib
import io
import json
import time

import pymupdf
import pytesseract
from PIL import Image

from academy.labs.rag import corpus
from academy.labs.runtime import (
  Input,
  Metric,
  callout,
  code,
  columns,
  example,
  images,
  lab,
  scorecard,
  table,
)
from nimbus.config import get_settings
from nimbus.llm import get_chat_model
from nimbus.rag.evaluation import evaluate
from nimbus.rag.ingestion.ocr import image_message, open_image, preprocess, read_image, render

DATA = get_settings().data_dir


def _uri(image: Image.Image, width: int = 360) -> str:
  """A small PNG data URI, to show the image the lab worked on."""
  copy = image.convert("RGB")
  if copy.width > width:
    copy = copy.resize((width, round(copy.height * width / copy.width)))
  buffer = io.BytesIO()
  copy.save(buffer, format="PNG")
  return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _norm(text: str) -> str:
  return " ".join(text.replace("|", " ").replace("*", " ").replace("#", " ").replace("`", " ").split())


def _invented(reference: str, text: str) -> list[str]:
  """Words in a transcription that aren't on the clean receipt (Markdown table headers it was asked to add aside)."""
  words = lambda t: ["".join(c for c in w if c.isalnum() or c in ".-/") for w in _norm(t).casefold().split()]  # noqa: E731
  known = set(words(reference)) | {"item", "items", "price", "qty", "quantity", "amount", "description"}
  return list(dict.fromkeys(w for w in words(text) if w and w not in known and any(c.isalnum() for c in w)))


def _quality(reference: str, text: str) -> float:
  return round(difflib.SequenceMatcher(None, _norm(reference), _norm(text)).ratio(), 3)


# ── Lab 1: OCR basics ───────────────────────────────────────────────────────────
# Each learning point is a small function: the "What it covers" tab shows its body, and the lab runs it.


def ocr_a_clean_image():
  import pytesseract

  from nimbus.rag.ingestion.ocr import open_image

  clean = open_image(DATA / "receipt.png")
  text = pytesseract.image_to_string(clean)  # pixels in, characters out (Tesseract, the classic open-source engine)
  return clean, text


def ocr_a_phone_photo():
  import pytesseract

  from nimbus.rag.ingestion.ocr import open_image

  photo = open_image(DATA / "receipt_photo.png")  # small, grey, shadowed: a real-world photo
  raw_text = pytesseract.image_to_string(photo)
  return photo, raw_text


def preprocess_then_ocr(photo, enlarge):
  import numpy as np
  import pytesseract
  from PIL import Image, ImageFilter, ImageOps

  gray = ImageOps.grayscale(photo)
  big = gray.resize((gray.width * enlarge, gray.height * enlarge), Image.Resampling.LANCZOS)  # 1. enlarge
  background = np.array(big.filter(ImageFilter.GaussianBlur(25)), dtype=float)  # the lighting, without the text
  flat = np.clip(np.array(big, dtype=float) / np.maximum(background, 1) * 255, 0, 255).astype("uint8")  # 2. even it out
  processed = ImageOps.autocontrast(Image.fromarray(flat), cutoff=1)  # 3. stretch the contrast
  text = pytesseract.image_to_string(processed)
  return processed, text


def a_naive_threshold(processed, threshold):
  import pytesseract

  binary = processed.point(lambda p: 255 if p > threshold else 0)  # pure black and white
  text = pytesseract.image_to_string(binary)
  return binary, text


def ocr_text_as_a_document(text):
  from langchain_core.documents import Document

  doc = Document(
    text,
    metadata={"source": "receipt_photo.png", "extraction": "ocr", "ocr_engine": "tesseract"},  # HOW it was read
  )
  return doc


@lab(
  "rag.ocr-basics",
  "OCR basics: a clean image, a phone photo, and preprocessing",
  inputs=[
    Input("enlarge", "Enlarge factor", "number", default=3, min=1, max=5, help="1 skips enlarging"),
    Input("threshold", "Black/white threshold", "number", default=160, min=60, max=230),
  ],
)
def ocr_basics(enlarge: int, threshold: int):
  """The same receipt as a clean image and as a phone photo; the photo raw, preprocessed, and thresholded. Quality
  is similarity to the clean image's OCR (0–1)."""
  clean, reference = ocr_a_clean_image()
  photo, raw_text = ocr_a_phone_photo()
  started = time.perf_counter()
  processed, fixed_text = preprocess_then_ocr(photo, enlarge)
  seconds = time.perf_counter() - started
  binary, binary_text = a_naive_threshold(processed, threshold)
  doc = ocr_text_as_a_document(fixed_text)
  rows = [
    {"variant": "clean image", "quality": 1.0, "total_found": "✓" if "787.50" in reference else "✗"},
    {
      "variant": "phone photo, raw",
      "quality": _quality(reference, raw_text),
      "total_found": "✓" if "787.50" in raw_text else "✗",
    },
    {
      "variant": f"photo, preprocessed (×{enlarge})",
      "quality": _quality(reference, fixed_text),
      "total_found": "✓" if "787.50" in fixed_text else "✗",
    },
    {
      "variant": f"+ black/white at {threshold}",
      "quality": _quality(reference, binary_text),
      "total_found": "✓" if "787.50" in binary_text else "✗",
    },
  ]
  return [
    images(
      [
        {"src": _uri(clean, 240), "caption": "receipt.png (clean)"},
        {"src": _uri(photo, 240), "caption": f"receipt_photo.png ({photo.width}×{photo.height})"},
        {"src": _uri(processed, 240), "caption": "preprocessed"},
        {"src": _uri(binary, 240), "caption": f"threshold {threshold}"},
      ],
      title="What Tesseract was given",
    ),
    scorecard(
      rows,
      [
        Metric("quality", "quality", "higher", "ratio", "Similarity to the clean image's OCR text"),
        Metric("total_found", "total 787.50 read", "none", "text"),
      ],
      title=f"The same receipt, four ways (preprocessing + OCR {seconds * 1000:,.0f} ms)",
    ),
    columns(
      [code(raw_text or "(nothing)", "text", "photo, raw"), code(fixed_text, "text", "photo, preprocessed")],
      title="OCR text",
    ),
    callout(
      f"Raw, the photo gave quality {rows[1]['quality']:.2f}; preprocessed, {rows[2]['quality']:.2f}; a hard "
      f"black/white threshold on top, {rows[3]['quality']:.2f}. "
      + (
        "But overall quality isn't the fact you need: here the preprocessed text misreads the total while scoring "
        "best overall. Check the facts questions depend on (totals, dates, ids), not just a similarity score. "
        if rows[2]["total_found"] == "✗"
        else "Preprocessing recovered the facts too. "
      )
      + "Production preprocesses only small or low-contrast images (it checks first) and records `extraction: ocr` "
      "in metadata, so answers from OCR text can be flagged.",
      "decision",
      "Preprocess, then OCR",
    ),
  ] + [
    example(
      "rag.l13.1",
      "Tesseract: pixels in, characters out",
      ocr_a_clean_image,
      "`pytesseract` calls the Tesseract program installed on this machine (free, local).",
      [code(reference, "text")],
    ),
    example(
      "rag.l13.2",
      "A clean image: nearly perfect",
      ocr_a_clean_image,
      "Every item, price and the total come out right.",
      [
        table(
          ["Expression", "Value"],
          [
            ["'787.50' in text", "787.50" in reference],
            ["lines", len([x for x in reference.splitlines() if x.strip()])],
          ],
        )
      ],
    ),
    example(
      "rag.l13.3",
      "A phone photo: OCR falls apart",
      ocr_a_phone_photo,
      f"{photo.width}×{photo.height} pixels, grey and shadowed.",
      [code(raw_text or "(nothing recognised)", "text"), table(["quality"], [[rows[1]["quality"]]])],
    ),
    example(
      "rag.l13.4",
      "Enlarge → even out the lighting → stretch the contrast",
      preprocess_then_ocr,
      "Dividing by a blurred copy removes the shadow; autocontrast makes the text black and the paper white.",
      [code(fixed_text, "text"), table(["quality"], [[rows[2]["quality"]]])],
    ),
    example(
      "rag.l13.5",
      "A naive threshold can make it worse",
      a_naive_threshold,
      "Forcing every pixel to black or white loses the grey edges of thin strokes.",
      [table(["threshold", "quality"], [[threshold, rows[3]["quality"]]])],
    ),
    example(
      "rag.l13.6",
      "OCR output becomes a normal Document",
      ocr_text_as_a_document,
      "Record how the text was obtained: OCR text can contain errors a reader should know about.",
      [table(["metadata"], [[json.dumps(doc.metadata)]])],
    ),
  ]


# ── Lab 2: OCR layout ───────────────────────────────────────────────────────────
def words_with_boxes(image):
  import pytesseract

  data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
  keys = ("text", "conf", "left", "top", "width", "height", "block_num", "line_num")
  words = [dict(zip(keys, row, strict=True)) for row in zip(*(data[k] for k in keys), strict=True) if row[0].strip()]
  return words  # each word: its text, a confidence 0–100, and its box on the page


def unsure_words(words, below):
  unsure = [(w["text"], w["conf"]) for w in words if w["conf"] < below]  # candidates for review or a second opinion
  return unsure


def rebuild_lines(words):
  lines = {}
  for w in words:
    lines.setdefault((w["block_num"], w["line_num"]), []).append(w["text"])  # the OCR's own reading order
  return {key: " ".join(ws) for key, ws in lines.items()}


def page_segmentation_modes(image):
  import pytesseract

  # --psm tells Tesseract what layout to expect: 4 = a single column of text of variable sizes,
  # 6 = one uniform block, 11 = sparse text (find as much as possible, no order)
  texts = {psm: pytesseract.image_to_string(image, config=f"--psm {psm}") for psm in (4, 6, 11)}
  return texts


def detect_orientation(image):
  import pytesseract

  rotated = image.rotate(90, expand=True, fillcolor="white")  # a page scanned sideways
  osd = pytesseract.image_to_osd(rotated, output_type=pytesseract.Output.DICT)
  upright = rotated.rotate(-osd["rotate"], expand=True, fillcolor="white")
  return rotated, osd, upright


@lab(
  "rag.ocr-layout",
  "OCR layout: words, boxes, confidence and page segmentation",
  inputs=[Input("below", "Flag words with confidence below", "number", default=80, min=10, max=99)],
)
def ocr_layout(below: int):
  """Word boxes and confidence on the clean receipt and the preprocessed photo; layout modes; rotation."""
  clean = open_image(DATA / "receipt.png")
  photo = preprocess(open_image(DATA / "receipt_photo.png"))
  clean_words = words_with_boxes(clean)
  photo_words = words_with_boxes(photo)
  flagged = unsure_words(photo_words, below)
  lines = rebuild_lines(clean_words)
  texts = page_segmentation_modes(clean)
  rotated, osd, upright = detect_orientation(clean)
  item_lines = ("2 x Cappuccino 360.00", "1 x Oat Latte 220.00", "1 x Vegan Brownie 170.00")
  psm_rows = [
    {
      "variant": f"--psm {psm}",
      "item_rows_whole": sum(row in " ".join(t.split()).replace("1x", "1 x") for row in item_lines),
      "lines": len([x for x in t.splitlines() if x.strip()]),
    }
    for psm, t in texts.items()
  ]
  rows = [
    {
      "variant": "clean receipt",
      "words": len(clean_words),
      "avg_conf": round(sum(w["conf"] for w in clean_words) / len(clean_words)),
      "flagged": len(unsure_words(clean_words, below)),
    },
    {
      "variant": "phone photo (preprocessed)",
      "words": len(photo_words),
      "avg_conf": round(sum(w["conf"] for w in photo_words) / len(photo_words)),
      "flagged": len(flagged),
    },
  ]
  return [
    scorecard(
      rows,
      [
        Metric("words", "words", "none", "int"),
        Metric("avg_conf", "average confidence", "higher", "int"),
        Metric("flagged", f"words below {below}", "lower", "int"),
      ],
      title="Confidence: how sure Tesseract is about each word",
    ),
    table(["Word", "Confidence"], flagged[:20], title=f"Photo words below {below}: candidates for review"),
    scorecard(
      psm_rows,
      [
        Metric("item_rows_whole", "item rows read whole (of 3)", "higher", "int"),
        Metric("lines", "lines", "none", "int"),
      ],
      title="Page segmentation modes on the clean receipt",
    ),
    images(
      [
        {"src": _uri(rotated, 200), "caption": "scanned sideways"},
        {"src": _uri(upright, 160), "caption": f"rotated back {osd['rotate']}°"},
      ],
      title="Orientation detection",
    ),
    callout(
      f"Average confidence drops from {rows[0]['avg_conf']} on the clean image to {rows[1]['avg_conf']} on the "
      f"photo, and {len(flagged)} photo words fall below {below}"
      + (
        f", including the misread total {next(w for w, c in flagged if w.startswith('787'))!r} at confidence "
        f"{next(c for w, c in flagged if w.startswith('787'))}"
        if any(w.startswith("787") for w, _ in flagged)
        else ""
      )
      + ". Route documents with unsure words to review or a vision model instead of trusting them. The layout mode "
      "changes how lines come out: test it on your own documents. Detect rotation before OCR: a sideways page reads "
      "as garbage.",
      "decision",
      "Use what OCR knows about itself",
    ),
  ] + [
    example(
      "rag.l14.1",
      "image_to_data: words with boxes and confidence",
      words_with_boxes,
      "Every word with its position on the page and a confidence score from 0 to 100.",
      [
        table(
          ["text", "conf", "left", "top", "width", "height"],
          [[w[k] for k in ("text", "conf", "left", "top", "width", "height")] for w in clean_words[:8]],
        )
      ],
    ),
    example(
      "rag.l14.2",
      "Find the words the engine is unsure about",
      unsure_words,
      f"{len(flagged)} words on the preprocessed photo fall below {below}.",
      [table(["Word", "Confidence"], flagged[:12])],
    ),
    example(
      "rag.l14.3",
      "Rebuild lines from block and line numbers",
      rebuild_lines,
      "Grouping words by `(block_num, line_num)` gives the OCR's reading order back.",
      [table(["block, line", "Text"], [[f"{k[0]}, {k[1]}", v] for k, v in list(lines.items())[4:10]])],
    ),
    example(
      "rag.l14.4",
      "Page segmentation modes (--psm)",
      page_segmentation_modes,
      "The same image, three layout assumptions.",
      [columns([code(t.strip(), "text", f"--psm {p}") for p, t in texts.items()])],
    ),
    example(
      "rag.l14.5",
      "Orientation detection (OSD)",
      detect_orientation,
      "`image_to_osd` reports how far the page is rotated; rotate it back before reading.",
      [
        table(
          ["Key", "Value"],
          [
            ["rotate", osd["rotate"]],
            ["orientation_conf", round(osd["orientation_conf"], 2)],
            ["first line after fixing", pytesseract.image_to_string(upright).strip().splitlines()[0]],
          ],
        )
      ],
    ),
  ]


# ── Lab 3: scanned PDFs ─────────────────────────────────────────────────────────
SCANNED = DATA / "scanned_notice.pdf"
TESSDATA = "/opt/homebrew/share/tessdata"  # where Homebrew installs Tesseract's language files


def a_scan_has_no_text():
  import pymupdf

  with pymupdf.open("data/nimbus/scanned_notice.pdf") as pdf:
    pages = [(len(page.get_text().strip()), len(page.get_images())) for page in pdf]  # (characters, images)
  return pages


def detect_scanned_pages():
  from pathlib import Path

  from nimbus.rag.ingestion.ocr import scanned_pages

  # Production: a page with fewer than 20 characters of text holds an image of text
  scanned = {
    name: scanned_pages(Path("data/nimbus") / name) for name in ("scanned_notice.pdf", "annual_report_2025.pdf")
  }
  return scanned


def render_and_ocr():
  import pymupdf
  import pytesseract
  from PIL import Image

  with pymupdf.open("data/nimbus/scanned_notice.pdf") as pdf:
    pixmap = pdf[0].get_pixmap(dpi=300)  # 300 dpi: the usual resolution for OCR
  image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
  text = pytesseract.image_to_string(image)
  return image, text


def hybrid_loader():
  from pathlib import Path

  from nimbus.rag.ingestion.loaders import PdfLoader

  docs = []
  for name in ("scanned_notice.pdf", "annual_report_2025.pdf"):
    # The text layer when a page has one, OCR only when it doesn't
    docs += PdfLoader(Path("data/nimbus") / name, name, ocr="tesseract").load()
  return docs


def pymupdf_built_in_ocr():
  import pymupdf

  with pymupdf.open("data/nimbus/scanned_notice.pdf") as pdf:
    page = pdf[1]
    textpage = page.get_textpage_ocr(dpi=300, full=True, tessdata=TESSDATA)
    text = page.get_text(textpage=textpage)  # PyMuPDF calls Tesseract for you
  return text


@lab("rag.scanned-pdfs", "Scanned PDFs: find pages with no text, OCR only those")
def scanned_pdfs():
  """The scanned notice and the annual report through a plain loader and the hybrid loader."""
  pages = a_scan_has_no_text()
  scanned = detect_scanned_pages()
  image, page_text = render_and_ocr()
  started = time.perf_counter()
  docs = hybrid_loader()
  seconds = time.perf_counter() - started
  built_in = pymupdf_built_in_ocr()
  from nimbus.rag.ingestion.loaders import PdfLoader

  plain = PdfLoader(SCANNED, SCANNED.name).load()
  checks = {"14 March 2026": "roastery closure (q41)", "25 March 2026": "fire drill (q42)", "Kavya Nair": "fire warden"}
  ocr_all = " ".join(d.page_content for d in docs if d.metadata["source"] == SCANNED.name)
  rows = [
    {
      "variant": "PDF loader, text layer only",
      "chars": sum(len(d.page_content.strip()) for d in plain),
      **{k: "✓" if k in " ".join(d.page_content for d in plain) else "✗" for k in checks},
    },
    {
      "variant": "hybrid loader (Tesseract)",
      "chars": len(ocr_all),
      **{k: "✓" if k in ocr_all else "✗" for k in checks},
    },
  ]
  return [
    scorecard(
      rows,
      [Metric("chars", "characters", "higher", "int"), *[Metric(k, v, "none", "text") for k, v in checks.items()]],
      title=f"scanned_notice.pdf (hybrid loader over both PDFs: {seconds:.1f} s)",
    ),
    table(
      ["Source", "page", "extraction", "characters"],
      [
        [d.metadata["source"], d.metadata["page"], d.metadata.get("extraction", "text_layer"), len(d.page_content)]
        for d in docs
      ],
      title="Hybrid loader: per page",
    ),
    images([{"src": _uri(image, 420), "caption": "scanned_notice.pdf page 1, rendered at 300 dpi"}]),
    callout(
      "A scanned PDF loads as empty pages: no error, just nothing to search. The hybrid loader checks each page and "
      "OCRs only those without a text layer, so digital PDFs keep their exact text. But Tesseract misreads this scan's "
      "date ('14 March' as '44 March'): the closure question stays unanswerable. Lab 4 tries a vision model.",
      "decision",
      "Scanned PDFs",
    ),
  ] + [
    example(
      "rag.l15.1",
      "A scan holds page images, not text",
      a_scan_has_no_text,
      "Each page has an image and (almost) no characters: normal loaders return empty text.",
      [table(["Page", "characters", "images"], [[i + 1, c, n] for i, (c, n) in enumerate(pages)])],
    ),
    example(
      "rag.l15.2",
      "Detecting pages that need OCR",
      detect_scanned_pages,
      "Pages with fewer than 20 characters of text.",
      [table(["File", "pages needing OCR (0-based)"], [[k, str(v)] for k, v in scanned.items()])],
    ),
    example(
      "rag.l15.3",
      "Render at 300 dpi, then OCR",
      render_and_ocr,
      f"Rendered to {image.width}×{image.height} pixels.",
      [code(page_text.strip(), "text")],
    ),
    example(
      "rag.l15.4",
      "A hybrid loader",
      hybrid_loader,
      "Production's PdfLoader with `ocr=\"tesseract\"`: the text layer where it exists, OCR where it doesn't.",
      [
        table(
          ["Source", "page", "extraction"],
          [[d.metadata["source"], d.metadata["page"], d.metadata.get("extraction", "text_layer")] for d in docs],
        )
      ],
    ),
    example(
      "rag.l15.5",
      "PyMuPDF's built-in OCR",
      pymupdf_built_in_ocr,
      "`get_textpage_ocr` renders and OCRs inside PyMuPDF (it needs Tesseract's language files).",
      [code(built_in.strip(), "text")],
    ),
  ]


# ── Lab 4: vision-model OCR ─────────────────────────────────────────────────────
def transcribe_with_a_vision_model(model, image):
  from nimbus.rag.ingestion.ocr import TRANSCRIBE

  # TRANSCRIBE asks for an exact Markdown transcription: line order kept, tables as tables, nothing "fixed"
  reply = model.invoke([image_message(image, TRANSCRIBE)])
  return reply.text.strip()


def _date_line(text: str) -> str:
  return next(
    (x.strip() for x in text.splitlines() if "March 2026 to" in x or "March 2026 for" in x), "(line not found)"
  )


@lab(
  "rag.vision-ocr",
  "Vision-model OCR: Tesseract vs a model that can see",
  uses_model=True,
  calls="3 calls to Mistral (vision)",
)
def vision_ocr():
  """The phone photo, a scanned page and a chart: Tesseract (with preprocessing) against the vision model."""
  model = get_chat_model()
  _, reference = ocr_a_clean_image()
  photo = open_image(DATA / "receipt_photo.png")
  with pymupdf.open(SCANNED) as pdf:
    page = render(pdf[0])
  chart = open_image(DATA / "sales_chart.png")
  tasks = {"phone photo": photo, "scanned page": page, "sales chart": chart}
  results, rows = {}, []
  for name, image in tasks.items():
    t = time.perf_counter()
    tess = read_image(image, "tesseract", cache=False)
    tess_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    vision = transcribe_with_a_vision_model(model, image)
    vision_ms = (time.perf_counter() - t) * 1000
    results[name] = (tess, vision)
    check = {"phone photo": "787.50", "scanned page": "14 March 2026", "sales chart": "62"}[name]
    for engine, text, ms in (("Tesseract", tess, tess_ms), ("vision model", vision, vision_ms)):
      rows.append(
        {
          "variant": f"{name} · {engine}",
          "quality": _quality(reference, text) if name == "phone photo" else None,
          "fact": f"{'✓' if check in text else '✗'} {check}",
          "ms": round(ms),
        }
      )
  chart_tess, chart_vision = results["sales chart"]
  invented = _invented(reference, results["phone photo"][1])
  rescued = [
    name
    for name in tasks
    if next(r for r in rows if r["variant"] == f"{name} · vision model")["fact"].startswith("✓")
    and next(r for r in rows if r["variant"] == f"{name} · Tesseract")["fact"].startswith("✗")
  ]
  return [
    scorecard(
      rows,
      [
        Metric("quality", "quality vs clean OCR", "higher", "ratio"),
        Metric("fact", "the fact a question needs", "none", "text"),
        Metric("ms", "time", "lower", "ms"),
      ],
      title="Three hard images, two engines",
    ),
    columns(
      [
        code(results["scanned page"][0].strip(), "text", "Tesseract"),
        code(results["scanned page"][1], "markdown", "vision model"),
      ],
      title="The scanned page",
    ),
    columns(
      [code(chart_tess.strip(), "text", "Tesseract"), code(chart_vision, "markdown", "vision model")], title="The chart"
    ),
    callout(
      (f"The vision model got the fact Tesseract missed on: {', '.join(rescued)}. " if rescued else "")
      + "On the chart it also returns label → value pairs instead of loose numbers. It costs a request per page, "
      "takes seconds instead of milliseconds, and sends the document to a provider. "
      + (
        f"It also wrote words that aren't on the receipt ({', '.join(invented[:5])}): a vision "
        "model produces plausible text, so keep the image to check against. "
        if invented
        else "Keep the image anyway: a vision model can produce plausible text that isn't there. "
      )
      + "Production makes the engine configuration (NIMBUS_OCR_ENGINE) and caches every result by image hash.",
      "decision",
      "Which engine",
    ),
  ] + [
    example(
      "rag.l16.1",
      "A Markdown transcription from a vision model",
      transcribe_with_a_vision_model,
      "The image goes in as a content block next to the instruction.",
      [code(results["phone photo"][1], "markdown", "phone photo")],
    ),
    example(
      "rag.l16.2",
      "Against Tesseract on the phone photo",
      transcribe_with_a_vision_model,
      "Quality is similarity to the clean image's OCR.",
      [
        table(
          ["Engine", "quality", "time"],
          [[r["variant"], r["quality"], f"{r['ms']} ms"] for r in rows if r["variant"].startswith("phone")],
        )
      ],
    ),
    example(
      "rag.l16.3",
      "The scanned line Tesseract misread",
      transcribe_with_a_vision_model,
      "The roastery closure date.",
      [
        table(
          ["Engine", "Line"],
          [
            ["Tesseract", _date_line(results["scanned page"][0])],
            ["vision model", _date_line(results["scanned page"][1])],
          ],
        )
      ],
    ),
    example(
      "rag.l16.4",
      "The trade-offs",
      transcribe_with_a_vision_model,
      "Quality vs cost, speed and privacy.",
      [
        table(
          ["", "Tesseract", "vision model"],
          [
            ["cost", "free", "a request per page"],
            [
              "speed",
              f"{sum(r['ms'] for r in rows if 'Tesseract' in r['variant'])} ms for 3",
              f"{sum(r['ms'] for r in rows if 'vision' in r['variant']):,} ms for 3",
            ],
            ["privacy", "local", "sent to the provider"],
            ["risk", "garbled text", "plausible 'corrections'"],
            ["words not on the receipt (phone photo)", "—", ", ".join(invented) or "none"],
          ],
        )
      ],
    ),
  ]


# ── Lab 5: structured extraction ────────────────────────────────────────────────
def receipt_schema():
  from pydantic import BaseModel, Field

  class LineItem(BaseModel):
    name: str
    quantity: int
    amount: float = Field(description="Line total in rupees")

  class Receipt(BaseModel):
    store: str
    date: str = Field(description="As printed, e.g. 11/03/2026")
    bill_number: str
    items: list[LineItem]
    subtotal: float
    tax: float
    total: float

  return Receipt


def path_a_vision(model, Receipt, image):  # noqa: N803

  extract = model.with_structured_output(Receipt, method="json_schema")
  receipt = extract.invoke([image_message(image, "Extract the receipt's fields.")])  # the model reads the image
  return receipt


def path_b_ocr_then_text(model, Receipt, image):  # noqa: N803
  import pytesseract

  ocr_text = pytesseract.image_to_string(image)  # local OCR first
  extract = model.with_structured_output(Receipt, method="json_schema")
  receipt = extract.invoke(f"Extract the receipt's fields from this OCR text:\n\n{ocr_text}")  # text only: cheaper
  return receipt


def check_the_numbers(receipt):
  problems = []
  if not receipt.items or receipt.total <= 0:
    problems.append("nothing was extracted")  # all zeros would pass the sums below
  if abs(sum(i.amount for i in receipt.items) - receipt.subtotal) > 0.01:
    problems.append("the items don't add up to the subtotal")
  if abs(receipt.subtotal + receipt.tax - receipt.total) > 0.01:
    problems.append("subtotal + tax ≠ total")
  return problems  # arithmetic catches misreads like 787.50 → 778.50


def receipt_as_a_document(receipt):
  from langchain_core.documents import Document

  doc = Document(
    f"Receipt {receipt.bill_number} from {receipt.store} on {receipt.date}: "
    + ", ".join(f"{i.quantity} x {i.name} ({i.amount:.2f})" for i in receipt.items)
    + f". Total {receipt.total:.2f} rupees including {receipt.tax:.2f} tax.",
    metadata={"source": "receipt.png", "doc_type": "receipt", "total": receipt.total, "date": receipt.date},
  )  # readable text for search, exact fields for filters ("receipts over 500 rupees")
  return doc


@lab(
  "rag.extraction",
  "Structured extraction: a document image → validated fields",
  uses_model=True,
  calls="2 calls to Mistral",
  inputs=[Input("image", "Image", "select", default="receipt.png", options=["receipt.png", "receipt_photo.png"])],
)
def extraction(image: str):
  """Path A (vision model fills the schema) against path B (Tesseract text → text model), then arithmetic checks."""
  model = get_chat_model()
  Receipt = receipt_schema()  # noqa: N806
  picture = open_image(DATA / image)
  t = time.perf_counter()
  a = path_a_vision(model, Receipt, picture)
  a_ms = (time.perf_counter() - t) * 1000
  t = time.perf_counter()
  b = path_b_ocr_then_text(model, Receipt, picture)
  b_ms = (time.perf_counter() - t) * 1000
  truth = {"total": 787.50, "subtotal": 750.0, "tax": 37.5, "items": 3, "bill_number": "INB-20417"}
  rows = []
  for label, r, ms in (("A · vision model reads the image", a, a_ms), ("B · Tesseract → text model", b, b_ms)):
    right = sum(
      [
        abs(r.total - truth["total"]) < 0.01,
        abs(r.subtotal - truth["subtotal"]) < 0.01,
        abs(r.tax - truth["tax"]) < 0.01,
        len(r.items) == truth["items"],
        r.bill_number == truth["bill_number"],
      ]
    )
    problems = check_the_numbers(r)
    rows.append(
      {
        "variant": label,
        "fields_right": f"{right}/5",
        "checks": "✓ consistent" if not problems else "✗ " + "; ".join(problems),
        "ms": round(ms),
      }
    )
  broken = a.model_copy(update={"total": 778.50})
  doc = receipt_as_a_document(a)
  return [
    images([{"src": _uri(picture, 260), "caption": image}]),
    scorecard(
      rows,
      [
        Metric("fields_right", "key fields right", "none", "text", "total, subtotal, tax, item count, bill number"),
        Metric("checks", "arithmetic checks", "none", "text"),
        Metric("ms", "time", "lower", "ms"),
      ],
      title="Two ways to fill the same schema",
    ),
    columns(
      [
        code(a.model_dump_json(indent=2), "json", "A · vision"),
        code(b.model_dump_json(indent=2), "json", "B · OCR → text model"),
      ]
    ),
    callout(
      f"Key fields right: {rows[0]['fields_right']} with the vision model, {rows[1]['fields_right']} from OCR text. "
      "Path B keeps OCR local and sends only text (cheaper, more private) but inherits every OCR error; on a poor "
      "photo it has nothing to work with. Validate whatever comes back: arithmetic (`subtotal + tax = total`) catches "
      "a misread digit, and a check for empty results catches the extraction that silently found nothing.",
      "decision",
      "Extraction",
    ),
  ] + [
    example(
      "rag.l17.1",
      "The fields as a Pydantic schema",
      receipt_schema,
      "Types and descriptions tell the model exactly what to return, and validation rejects anything else.",
      [code(json.dumps(Receipt.model_json_schema()["properties"], indent=2)[:900], "json")],
    ),
    example(
      "rag.l17.2",
      "Path A: the vision model fills the schema",
      path_a_vision,
      "Structured output with an image as input.",
      [code(a.model_dump_json(indent=2), "json")],
    ),
    example(
      "rag.l17.3",
      "Path B: Tesseract text → a text model",
      path_b_ocr_then_text,
      "OCR locally, then structured output from text.",
      [code(b.model_dump_json(indent=2), "json")],
    ),
    example(
      "rag.l17.4",
      "Validate the numbers",
      check_the_numbers,
      "Arithmetic checks catch misreads without knowing the right answer.",
      [
        table(
          ["Receipt", "Problems"],
          [
            ["path A", "; ".join(check_the_numbers(a)) or "none"],
            ["path B", "; ".join(check_the_numbers(b)) or "none"],
            ["A with total misread as 778.50", "; ".join(check_the_numbers(broken))],
          ],
        )
      ],
    ),
    example(
      "rag.l17.5",
      "The result as a RAG Document",
      receipt_as_a_document,
      "Readable text for search, exact fields in metadata for filters.",
      [table(["page_content", "metadata"], [[doc.page_content, json.dumps(doc.metadata)]])],
    ),
  ]


# ── Lab 6: layout-aware parsing ─────────────────────────────────────────────────
def text_with_positions_and_fonts():
  import pymupdf

  runs = []
  with pymupdf.open("data/nimbus/annual_report_2025.pdf") as pdf:
    for block in pdf[0].get_text("dict")["blocks"]:  # blocks → lines → spans (runs of one font)
      for line in block.get("lines", []):
        for span in line["spans"]:
          if span["text"].strip():
            runs.append(
              {
                "text": span["text"],
                "size": round(span["size"], 1),
                "bold": "Bold" in span["font"],
                "y": round(span["bbox"][1]),
              }
            )
  return runs


def headings_from_font_size(runs, page_height):
  body = 11  # the body text size on this report
  lines = []
  for r in runs:
    if r["y"] < 50 or r["y"] > page_height - 40:
      continue  # top/bottom margin: a running header or footer, removed by position
    if r["size"] >= 20:
      lines.append(f"# {r['text']}")
    elif r["size"] > body:
      lines.append(f"## {r['text']}")
    else:
      lines.append(r["text"])
  return "\n".join(lines)


def reading_order_of_two_columns():
  import pymupdf

  with pymupdf.open("data/nimbus/annual_report_2025.pdf") as pdf:
    page = pdf[2]
    height, middle = page.rect.height, page.rect.width / 2
    blocks = [b for b in page.get_text("blocks") if 50 < b[1] < height - 40]  # (x0, y0, x1, y1, text, ...)
  naive = sorted(blocks, key=lambda b: (round(b[1]), b[0]))  # top to bottom across the page: columns interleave
  by_column = sorted(blocks, key=lambda b: (b[0] > middle, b[1]))  # left column first, then right
  return naive, by_column


def pymupdf4llm_does_it_all():
  import pymupdf4llm

  markdown = pymupdf4llm.to_markdown(
    "data/nimbus/annual_report_2025.pdf", pages=[0, 2], header=False, footer=False, use_ocr=False
  )  # headings, tables, reading order, no running header/footer
  return markdown


@lab("rag.layout", "Layout-aware parsing: headings, margins and columns from PDF geometry")
def layout_lab():
  """The annual report's page geometry: font sizes become headings, margins drop headers, columns read in order."""
  runs = text_with_positions_and_fonts()
  with pymupdf.open(DATA / "annual_report_2025.pdf") as pdf:
    height = pdf[0].rect.height
  markdown_by_hand = headings_from_font_size(runs, height)
  naive, by_column = reading_order_of_two_columns()
  llm = pymupdf4llm_does_it_all()
  first = lambda blocks: [" ".join(b[4].split())[:40] for b in blocks]  # noqa: E731
  return [
    table(
      ["text", "size", "bold", "y"],
      [[r["text"][:60], r["size"], r["bold"], r["y"]] for r in runs[:10]],
      title="Page 1 as text runs: position and font",
    ),
    columns(
      [code(markdown_by_hand, "markdown", "by hand: font size + margins"), code(llm[:1400], "markdown", "pymupdf4llm")],
      title="Page 1 (and 3) as Markdown",
    ),
    table(
      ["#", "naive (top to bottom)", "column by column"],
      [[i + 1, a, b] for i, (a, b) in enumerate(zip(first(naive), first(by_column), strict=True))],
      title="Page 3 is two columns: reading order",
    ),
    callout(
      "Layout is in the geometry: bigger fonts are headings, text in the top and bottom margins is a running header "
      "or footer, and a two-column page must be read column by column or sentences from both columns interleave. "
      "pymupdf4llm (production's PDF loader) does all of this; knowing how lets you fix the documents where it "
      "guesses wrong.",
      "decision",
      "Layout",
    ),
  ] + [
    example(
      "rag.l18.1",
      "Text as blocks with positions and fonts",
      text_with_positions_and_fonts,
      "`get_text('dict')`: blocks → lines → spans, each with its bounding box, font and size.",
      [table(["text", "size", "bold", "y"], [[r["text"][:50], r["size"], r["bold"], r["y"]] for r in runs[:8]])],
    ),
    example(
      "rag.l18.2",
      "Headings from font size",
      headings_from_font_size,
      "The largest text becomes `#`, larger-than-body text `##`.",
      [code(markdown_by_hand, "markdown")],
    ),
    example(
      "rag.l18.3",
      "Headers and footers by position",
      headings_from_font_size,
      "Runs in the top 50 and bottom 40 points are dropped: the 'Confidential - Page 1 of 3' line is gone.",
      [table(["Running header in the output?"], [["Confidential" in markdown_by_hand]])],
    ),
    example(
      "rag.l18.4",
      "Two columns: read column by column",
      reading_order_of_two_columns,
      "Sorting blocks top to bottom mixes the columns.",
      [
        table(
          ["#", "naive", "column by column"],
          [[i + 1, a, b] for i, (a, b) in enumerate(zip(first(naive), first(by_column), strict=True))],
        )
      ],
    ),
    example(
      "rag.l18.5",
      "pymupdf4llm: PDF → clean Markdown",
      pymupdf4llm_does_it_all,
      "One call; `header=False, footer=False` drops the running lines.",
      [code(llm[:1200], "markdown")],
    ),
  ]


# ── Production path: loaders v3 (OCR) ───────────────────────────────────────────
OCR_QUESTIONS = ("q41", "q42", "q43", "q44")


def evaluate_ocr(k: int = 5) -> dict:
  import numpy as np

  reports = {}
  for version in ("v2", "v3-tesseract", "v3-vision"):
    chunks = corpus.chunks(500, 80, version)
    vectors = corpus.chunk_vectors(500, 80, version)

    def retrieve(query: str, n: int, chunks=chunks, vectors=vectors):
      q = np.array(corpus.embeddings().embed_query(query))
      return [chunks[i] for i in np.argsort(-(vectors @ q))[:n]]

    reports[version] = evaluate(
      retrieve,
      corpus.questions(),
      [c.page_content for c in chunks],
      k=k,
      indexed_sources=corpus.indexed_sources(500, 80, version),
    )
  return reports


@lab(
  "rag.ocr-eval",
  "Evaluate: loaders v2 vs v3 (Tesseract) vs v3 (vision)",
  uses_model=True,
  calls="5 vision calls the first time (3 images + 2 scanned pages), then cached",
  inputs=[Input("k", "Top k", "number", default=5, min=1, max=10)],
)
def ocr_eval(k: int):
  """The same 45 questions; only the OCR engine differs."""
  started = time.perf_counter()
  reports = evaluate_ocr(k)
  seconds = time.perf_counter() - started
  rows = []
  for version, report in reports.items():
    s = report.summary()
    rows.append(
      {
        "variant": {"v2": "v2 (no OCR)", "v3-tesseract": "v3 · Tesseract", "v3-vision": "v3 · vision model"}[version],
        "files": len(corpus.indexed_sources(500, 80, version)),
        "answerable": s["answerable"],
        "hit": report.at["hit"][k],
        "mrr": s["mrr"],
        "ocr_answerable": sum(r.total_relevant > 0 for r in report.results if r.question.id in OCR_QUESTIONS),
      }
    )
  per_q = [
    [
      r.question.question,
      r.question.source,
      *[("✓" if rep.results[i].total_relevant else "✗") for rep in reports.values()],
    ]
    for i, r in enumerate(reports["v2"].results)
    if r.question.id in OCR_QUESTIONS
  ]
  return [
    scorecard(
      rows,
      [
        Metric("files", "files indexed", "higher", "int"),
        Metric("answerable", "answerable (45)", "higher", "ratio"),
        Metric("hit", f"hit@{k}", "higher", "ratio"),
        Metric("mrr", "MRR", "higher", "ratio"),
        Metric("ocr_answerable", "OCR questions answerable (of 4)", "higher", "int"),
      ],
      title=f"45 questions, k={k} · {seconds:.1f} s (OCR results cached by image hash)",
    ),
    table(
      ["Question", "Source", "v2", "v3 · Tesseract", "v3 · vision"],
      per_q,
      title="The four questions only OCR can answer: answerable?",
    ),
    callout(
      f"Tesseract makes {rows[1]['ocr_answerable']} of the 4 image questions answerable; the vision model "
      f"{rows[2]['ocr_answerable']}. Over all 45 questions answerable goes {rows[0]['answerable']:.2f} → "
      f"{rows[1]['answerable']:.2f} (Tesseract) → {rows[2]['answerable']:.2f} (vision). Production defaults to "
      "Tesseract (free, local, private) and switches per deployment to the vision model "
      "(NIMBUS_OCR_ENGINE=vision) where scans and charts matter; results are cached either way.",
      "decision",
      "Ship loaders v3",
    ),
  ]
