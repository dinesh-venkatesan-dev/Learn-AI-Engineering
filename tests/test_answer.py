from langchain_core.documents import Document

from nimbus.rag.answer import Citation, fit_budget, format_sources, order_for_attention, verify

CHUNKS = [
  Document("Every employee gets 24 days of paid annual leave per year.", metadata={"source": "handbook.md", "page": 1}),
  Document("| Milestone | Share |\n|---|---|\n| On delivery | 60% |", metadata={"source": "contract.docx"}),
]


def test_quotes_are_verified_against_the_cited_source():
  verified, rejected = verify(
    [
      Citation(source=1, quote="24 days of **paid** annual leave"),  # Markdown doesn't matter
      Citation(source=2, quote="On delivery | 60%"),  # nor do table pipes
      Citation(source=1, quote="On delivery | 60%"),  # the wrong source
      Citation(source=3, quote="anything"),  # a source that doesn't exist
    ],
    CHUNKS,
  )
  assert [c.source for c in verified] == [1, 2]
  assert len(rejected) == 2


def test_best_chunks_go_to_the_start_and_the_end():
  ranked = [Document(str(i)) for i in range(1, 6)]  # 1 = best
  assert [d.page_content for d in order_for_attention(ranked)] == ["1", "3", "5", "4", "2"]


def test_budget_keeps_the_best_that_fit_and_at_least_one():
  docs = [Document("word " * 200), Document("word " * 200), Document("word " * 200)]
  assert len(fit_budget(docs, 300)) == 1
  assert len(fit_budget(docs, 10)) == 1  # never empty


def test_sources_are_numbered_with_where_they_come_from():
  text = format_sources(CHUNKS)
  assert text.startswith("[1] (handbook.md, page 1)")
  assert "[2] (contract.docx)" in text
