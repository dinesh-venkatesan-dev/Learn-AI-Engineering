from pathlib import Path

from nimbus.rag.ingestion.loaders import discover, loader_for
from nimbus.rag.ingestion.ocr import needs_preprocessing, open_image, read_image, scanned_pages

DATA = Path(__file__).resolve().parents[1] / "data" / "nimbus"


def test_scanned_pages_are_detected_and_digital_ones_are_not():
  assert scanned_pages(DATA / "scanned_notice.pdf") == [0, 1]
  assert scanned_pages(DATA / "annual_report_2025.pdf") == []


def test_only_small_or_flat_images_are_preprocessed():
  assert needs_preprocessing(open_image(DATA / "receipt_photo.png"))
  assert not needs_preprocessing(open_image(DATA / "receipt.png"))


def test_tesseract_reads_the_clean_receipt_and_caches_it(tmp_path, monkeypatch):
  monkeypatch.setattr("nimbus.rag.ingestion.ocr.CACHE_DIR", tmp_path)
  image = open_image(DATA / "receipt.png")
  text = read_image(image, "tesseract")
  assert "TOTAL 787.50" in text
  assert len(list(tmp_path.glob("*.json"))) == 1
  assert read_image(image, "tesseract") == text  # served from the cache


def test_v3_loads_images_and_ocrs_scanned_pages(tmp_path, monkeypatch):
  monkeypatch.setattr("nimbus.rag.ingestion.ocr.CACHE_DIR", tmp_path)
  files, unsupported = discover(DATA, "v3")
  assert not unsupported and any(p.suffix == ".png" for p in files)
  pages = loader_for(DATA / "scanned_notice.pdf", DATA, "tesseract").load()
  assert [p.metadata["extraction"] for p in pages] == ["ocr", "ocr"]
  assert "25 March 2026" in pages[1].page_content
  digital = loader_for(DATA / "annual_report_2025.pdf", DATA, "tesseract").load()
  assert all("extraction" not in p.metadata for p in digital)  # a text layer is never OCR'd
