from langchain_core.documents import Document

from nimbus.rag.ingestion.cleaning import clean_file, html_to_text, normalize, remove_repeated_lines


def test_html_keeps_main_content_and_drops_boilerplate():
  html = (
    "<html><body><nav>Home | Shop</nav><script>track()</script>"
    "<main><h1>FAQ</h1><p>Delivery takes 2 days.</p></main><footer>© 2026</footer></body></html>"
  )
  text = html_to_text(html)
  assert "Delivery takes 2 days." in text
  assert "Home" not in text and "track" not in text and "2026" not in text


def test_repeated_page_headers_and_footers_are_removed():
  pages = [f"ACME Report\nReal content {i}\nConfidential - Page {i} of 3" for i in range(1, 4)]
  cleaned = remove_repeated_lines(pages)
  assert cleaned == ["Real content 1", "Real content 2", "Real content 3"]


def test_single_page_is_left_alone():
  assert remove_repeated_lines(["Only page"]) == ["Only page"]


def test_normalize_whitespace_and_bold_markers():
  assert normalize("  **Bold**   text\n\n\n\nnext ") == "Bold text\n\nnext"


def test_empty_pages_are_dropped_and_metadata_kept():
  docs = [
    Document("Header\nText", metadata={"source": "a.pdf", "page": 1, "file_type": "pdf"}),
    Document("Header\n", metadata={"source": "a.pdf", "page": 2, "file_type": "pdf"}),
  ]
  cleaned = clean_file(docs)
  assert [d.page_content for d in cleaned] == ["Text"]
  assert cleaned[0].metadata == {"source": "a.pdf", "page": 1, "file_type": "pdf"}
