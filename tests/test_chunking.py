from langchain_core.documents import Document

from nimbus.rag.ingestion.chunking import build_splitter, split_documents

TEXT = "# Policy\n\n" + "\n\n".join(f"## Section {i}\n" + f"Sentence {i} has some words. " * 12 for i in range(6))


def test_chunks_respect_the_size_limit_and_keep_metadata():
  doc = Document(TEXT, metadata={"source": "policy.md", "page": 1, "file_type": "markdown"})
  chunks = split_documents([doc], build_splitter(300, 50), min_chars=20)
  assert len(chunks) > 1
  assert all(len(c.page_content) <= 300 for c in chunks)
  assert all(c.metadata["source"] == "policy.md" for c in chunks)
  assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_start_index_points_at_the_chunk_in_the_source():
  doc = Document(TEXT, metadata={"source": "policy.md", "page": 1, "file_type": "markdown"})
  for c in split_documents([doc], build_splitter(300, 50), min_chars=20):
    start = c.metadata["start_index"]
    assert TEXT[start : start + len(c.page_content)] == c.page_content


def test_tiny_fragments_are_dropped():
  doc = Document("# Title\n\nShort.", metadata={"source": "t.md", "page": 1, "file_type": "markdown"})
  assert split_documents([doc], build_splitter(100, 10), min_chars=40) == []


def test_sections_split_at_headings_and_record_the_path():
  from nimbus.rag.ingestion.chunking import split_by_sections

  doc = Document(
    "# Handbook\n\n## Leave\nTwenty-four days of paid leave.\n\n## Travel\nHotels up to 6,000 rupees.",
    metadata={"source": "h.md"},
  )
  chunks = split_by_sections([doc], build_splitter(300, 0), 10)
  assert [c.metadata["section"] for c in chunks] == ["Handbook › Leave", "Handbook › Travel"]
  assert all(c.metadata["source"] == "h.md" for c in chunks)


def test_context_headers_name_source_and_section():
  from nimbus.rag.ingestion.chunking import add_context_headers

  [chunk] = add_context_headers(
    [Document("Hotels up to 6,000 rupees.", metadata={"source": "h.md", "section": "Travel"})]
  )
  assert chunk.page_content == "[h.md › Travel]\nHotels up to 6,000 rupees."


def test_llm_context_needs_a_model():
  import pytest

  from nimbus.rag.ingestion.chunking import make_chunks

  with pytest.raises(ValueError, match="needs a chat model"):
    make_chunks({"a.md": [Document("text " * 50, metadata={"source": "a.md"})]}, 100, 0, 10, context="llm")
