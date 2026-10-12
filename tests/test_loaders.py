from pathlib import Path

from nimbus.rag.ingestion.loaders import discover, loader_for

DATA = Path(__file__).resolve().parents[1] / "data" / "nimbus"


def test_discover_skips_hidden_and_unsupported_files(tmp_path):
  (tmp_path / "a.md").write_text("# A")
  (tmp_path / "b.png").write_bytes(b"x")
  (tmp_path / ".hidden.md").write_text("secret")
  (tmp_path / "sub").mkdir()
  (tmp_path / "sub" / "c.txt").write_text("C")
  supported, unsupported = discover(tmp_path)
  assert [p.relative_to(tmp_path).as_posix() for p in supported] == ["a.md", "sub/c.txt"]
  assert [p.name for p in unsupported] == ["b.png"]
  v1, unsupported_v1 = discover(tmp_path, "v1")
  assert [p.name for p in v1] == ["a.md", "c.txt"]


def test_text_file_becomes_one_document_with_source_metadata(tmp_path):
  (tmp_path / "notes.md").write_text("# Notes\nHello")
  [doc] = loader_for(tmp_path / "notes.md", tmp_path).load()
  assert doc.page_content == "# Notes\nHello"
  assert doc.metadata == {"source": "notes.md", "page": 1, "file_type": "markdown"}


def test_pdf_becomes_one_document_per_page():
  docs = loader_for(DATA / "annual_report_2025.pdf", DATA).load()
  assert [d.metadata["page"] for d in docs] == [1, 2, 3]
  assert "Letter from the CEO" in docs[0].page_content


def test_v2_loads_rows_records_sections_slides_and_email():
  by_type = {
    name: loader_for(DATA / name, DATA).load()
    for name in (
      "menu.csv",
      "orders.json",
      "supplier_contract.docx",
      "q1_2026_review.pptx",
      "green_valley_price_notice.eml",
    )
  }
  assert (
    len(by_type["menu.csv"]) == 8
    and "item: Oat Latte\ncategory: hot\nprice_rupees: 220" in by_type["menu.csv"][2].page_content
  )
  assert by_type["orders.json"][2].page_content.startswith("order id: NC-1003")
  assert all(d.page_content.startswith("## ") and d.metadata["section"] for d in by_type["supplier_contract.docx"])
  assert any("Speaker notes:" in d.page_content for d in by_type["q1_2026_review.pptx"])
  email = by_type["green_valley_price_notice.eml"]
  assert (
    email[0].metadata["subject"].startswith("Re: Revised green bean prices") and " wrote:" not in email[0].page_content
  )
